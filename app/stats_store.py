"""Token usage statistics store for the NInfer launcher.

Ingests engine request-log JSONL (--request-log-jsonl) incrementally into a local
SQLite database, rolls up per-day aggregates, and estimates cost.

Metric conventions follow public serving dashboards: vLLM Prometheus metrics
(prompt_tokens_total / generation_tokens_total / time_to_first_token_seconds),
OpenRouter usage pages (tokens by day, requests, cost), SGLang /metrics.

Local cost model:
  api_cost = ((prompt - cache_hit)/1e6) * input_price
           + (cache_hit/1e6) * cache_input_price
           + (completion/1e6) * output_price
  cache-hit prompt tokens are billed at the (discounted) cached-input price;
  the remainder of prompt tokens at the regular input price.
  power_cost = (prefill_s + decode_s) * gpu_tdp_w / 3.6e6 * electricity_price_per_kwh
All prices are user-editable; defaults approximate Qwen API tiers (CNY per 1M tokens).

Pure stdlib (sqlite3/json/pathlib); no third-party dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

_CST = timezone(timedelta(hours=8))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    day TEXT NOT NULL,
    hour INTEGER NOT NULL DEFAULT 0,
    event TEXT NOT NULL,
    protocol TEXT, model TEXT, stream INTEGER,
    finish_reason TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    computed_prefill_tokens INTEGER NOT NULL DEFAULT 0,
    prefix_cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    vision_tokens INTEGER NOT NULL DEFAULT 0,
    ttft_ms REAL, prefill_s REAL, decode_s REAL, total_s REAL, prepare_s REAL,
    speculative_backend TEXT, draft_window INTEGER, spec_rounds INTEGER,
    drafted_tokens INTEGER NOT NULL DEFAULT 0, accepted_tokens INTEGER NOT NULL DEFAULT 0,
    server_instance_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingest_state (path TEXT PRIMARY KEY, offset INTEGER NOT NULL);
"""

_DEFAULT_PRICES = {
    "input_price_per_m": 2.0,      # CNY per 1M input tokens (Qwen-plus tier)
    "cache_input_price_per_m": 0.2,  # CNY per 1M cached-input tokens (≈10% of input, industry convention)
    "output_price_per_m": 8.0,     # CNY per 1M output tokens
    "gpu_tdp_w": 350.0,            # RTX 5090 TDP in watts
    "electricity_price_per_kwh": 0.6,  # CNY per kWh
}

_EVENT_FILTER = {"request_done"}


@dataclass
class HourStats:
    """某自然日单小时聚合（图表 tooltip / 逐小时细条用）。"""
    hour: int
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    computed_prefill_tokens: int = 0
    avg_ttft_ms: float | None = None


@dataclass
class DayStats:
    day: str
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    computed_prefill_tokens: int = 0
    cache_hit_tokens: int = 0
    avg_ttft_ms: float | None = None
    avg_decode_tps: float | None = None
    spec_accept_rate: float | None = None  # 0..1 or None
    api_cost_cny: float = 0.0
    power_cost_cny: float = 0.0
    spec_drafted: int = 0
    spec_accepted: int = 0


class StatsStore:
    """Incremental JSONL ingestion + daily rollups + cost estimation."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        for k, v in _DEFAULT_PRICES.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO config(key,value) VALUES(?,?)", (k, str(v)))
        self._conn.commit()

    # ------------------------------------------------------------------ prices
    @property
    def prices(self) -> dict[str, float]:
        rows = self._conn.execute("SELECT key,value FROM config").fetchall()
        out = dict(_DEFAULT_PRICES)
        for r in rows:
            try:
                out[r["key"]] = float(r["value"])
            except ValueError:
                pass
        return out

    def set_prices(self, **kw: float) -> None:
        for k, v in kw.items():
            if k in _DEFAULT_PRICES:
                self._conn.execute(
                    "INSERT INTO config(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, str(float(v))))
        self._conn.commit()

    # ----------------------------------------------------------------- ingest
    def ingest_jsonl(self, path: str | Path) -> int:
        """Read new complete lines after the recorded offset; insert request_done events.

        Returns the number of newly ingested request_done records.
        Incomplete trailing line (engine still flushing) is left for next pass.
        """
        p = Path(path)
        if not p.is_file():
            return 0
        size = p.stat().st_size
        row = self._conn.execute(
            "SELECT offset FROM ingest_state WHERE path=?", (str(p),)).fetchone()
        offset = row["offset"] if row else 0
        if size < offset:  # file truncated/rotated: start over
            offset = 0
        if size == offset:
            return 0

        count = 0
        with open(p, "rb") as f:
            f.seek(offset)
            data = f.read(size - offset)
        # keep trailing partial line (no newline) for next ingest
        cut = data.rfind(b"\n")
        if cut < 0:
            return 0
        complete, tail = data[: cut + 1], data[cut + 1:]
        lines = complete.splitlines()
        now_day = datetime.now(_CST).strftime("%Y-%m-%d")
        batches: list[tuple] = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(rec, dict) or rec.get("event") not in _EVENT_FILTER:
                continue
            ev = self._parse_event(rec)
            if ev is not None:
                batches.append(ev)
                count += 1
        if batches:
            self._conn.executemany(
                """INSERT INTO events(ts_ms,day,hour,event,protocol,model,stream,
                   finish_reason,prompt_tokens,completion_tokens,computed_prefill_tokens,
                   prefix_cache_hit_tokens,vision_tokens,ttft_ms,prefill_s,decode_s,total_s,
                   prepare_s,speculative_backend,draft_window,spec_rounds,drafted_tokens,
                   accepted_tokens,server_instance_id)
                   VALUES(:ts_ms,:day,:hour,:event,:protocol,:model,:stream,:finish_reason,
                   :prompt_tokens,:completion_tokens,:computed_prefill_tokens,
                   :prefix_cache_hit_tokens,:vision_tokens,:ttft_ms,:prefill_s,:decode_s,
                   :total_s,:prepare_s,:speculative_backend,:draft_window,:spec_rounds,
                   :drafted_tokens,:accepted_tokens,:server_instance_id)""", batches)
        self._conn.execute(
            "INSERT INTO ingest_state(path,offset) VALUES(?,?) "
            "ON CONFLICT(path) DO UPDATE SET offset=excluded.offset",
            (str(p), offset + len(complete)))
        self._conn.commit()
        return count

    @staticmethod
    def _parse_event(rec: dict[str, Any]) -> dict[str, Any] | None:
        ts_ms = rec.get("timestamp_unix_ms")
        if not isinstance(ts_ms, (int, float)) or ts_ms <= 0:
            return None
        dt = datetime.fromtimestamp(ts_ms / 1000.0, _CST)
        res = rec.get("result") or {}
        tim = rec.get("timings_seconds") or {}
        spec = rec.get("speculative") or {}
        req = rec.get("request") or {}
        prep = rec.get("preparation_seconds") or {}

        def _i(v: Any) -> int:
            return int(v) if isinstance(v, (int, float)) and v >= 0 else 0

        def _f(v: Any) -> float | None:
            return float(v) if isinstance(v, (int, float)) and v >= 0 else None

        tps = tim.get("total")
        dec = tim.get("decode")
        completion = _i(res.get("completion_tokens"))
        return {
            "ts_ms": int(ts_ms),
            "day": dt.strftime("%Y-%m-%d"),
            "hour": dt.hour,
            "event": rec.get("event", "request_done"),
            "protocol": req.get("protocol"),
            "model": req.get("model"),
            "stream": 1 if req.get("stream") else 0,
            "finish_reason": res.get("finish_reason"),
            "prompt_tokens": _i(res.get("prompt_tokens")),
            "completion_tokens": completion,
            "computed_prefill_tokens": _i(res.get("computed_prefill_tokens")),
            "prefix_cache_hit_tokens": _i(res.get("prefix_cache_hit_tokens")),
            "vision_tokens": _i(prep.get("vision_tokens")),
            "ttft_ms": (_f(tim.get("ttft")) or 0.0) * 1000.0 or None,
            "prefill_s": _f(tim.get("prefill")),
            "decode_s": _f(tim.get("decode")),
            "total_s": _f(tps),
            "prepare_s": _f(tim.get("prepare")),
            "speculative_backend": spec.get("backend"),
            "draft_window": _i(spec.get("draft_window")) or None,
            "spec_rounds": _i(spec.get("rounds")) or None,
            "drafted_tokens": _i(spec.get("drafted_tokens")),
            "accepted_tokens": _i(spec.get("accepted_tokens")),
            "server_instance_id": rec.get("server_instance_id"),
        }

    def hourly_stats(self, day: str) -> dict[int, HourStats]:
        """某自然日按小时聚合；无数据的整点不在结果中。"""
        rows = self._conn.execute(
            "SELECT hour, COUNT(*) n, COALESCE(SUM(prompt_tokens),0) p, "
            "COALESCE(SUM(completion_tokens),0) c, COALESCE(SUM(prefix_cache_hit_tokens),0) ch, "
            "COALESCE(SUM(computed_prefill_tokens),0) cf, AVG(ttft_ms) ttft "
            "FROM events WHERE day=? AND event='request_done' GROUP BY hour", (day,)).fetchall()
        return {r["hour"]: HourStats(
            hour=r["hour"], requests=int(r["n"]), prompt_tokens=int(r["p"]),
            completion_tokens=int(r["c"]), cache_hit_tokens=int(r["ch"]),
            computed_prefill_tokens=int(r["cf"]), avg_ttft_ms=r["ttft"])
            for r in rows}

    # -------------------------------------------------------------- rollups
    def daily_stats(self, days: int = 14) -> list[DayStats]:
        pr = self.prices
        end = datetime.now(_CST)
        start = (end - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            """SELECT day,
               COUNT(*) AS requests,
               COALESCE(SUM(prompt_tokens),0) AS prompt,
               COALESCE(SUM(completion_tokens),0) AS completion,
               COALESCE(SUM(computed_prefill_tokens),0) AS prefill,
               COALESCE(SUM(prefix_cache_hit_tokens),0) AS cache,
               AVG(ttft_ms) AS ttft,
               SUM(CASE WHEN decode_s>0 THEN decode_s ELSE 0 END) AS dec_s,
               SUM(CASE WHEN completion_tokens>0 AND decode_s>0 THEN completion_tokens ELSE 0 END) AS dec_tok,
               SUM(prefill_s) AS pre_s,
               COALESCE(SUM(drafted_tokens),0) AS drafted,
               COALESCE(SUM(accepted_tokens),0) AS accepted
               FROM events WHERE day>=? GROUP BY day ORDER BY day""", (start,)).fetchall()
        by_day = {r["day"]: r for r in rows}
        out: list[DayStats] = []
        cur = datetime.strptime(start, "%Y-%m-%d")
        for _ in range(days):
            d = cur.strftime("%Y-%m-%d")
            r = by_day.get(d)
            s = DayStats(day=d)
            if r:
                dec_s, dec_tok = r["dec_s"] or 0, r["dec_tok"] or 0
                s.requests, s.prompt_tokens, s.completion_tokens = r["requests"], r["prompt"], r["completion"]
                s.total_tokens = s.prompt_tokens + s.completion_tokens
                s.computed_prefill_tokens, s.cache_hit_tokens = r["prefill"], r["cache"]
                s.avg_ttft_ms = r["ttft"]
                s.avg_decode_tps = dec_tok / dec_s if dec_s > 0 else None
                s.spec_drafted, s.spec_accepted = r["drafted"], r["accepted"]
                s.spec_accept_rate = r["accepted"] / r["drafted"] if r["drafted"] else None
                cache_hit = min(s.cache_hit_tokens, s.prompt_tokens)
                s.api_cost_cny = ((s.prompt_tokens - cache_hit) * pr["input_price_per_m"]
                                  + cache_hit * pr["cache_input_price_per_m"]
                                  + s.completion_tokens * pr["output_price_per_m"]) / 1e6
                busy_h = ((r["dec_s"] or 0) + (r["pre_s"] or 0)) / 3600.0
                s.power_cost_cny = busy_h * pr["gpu_tdp_w"] / 1000.0 * pr["electricity_price_per_kwh"]
            out.append(s)
            cur += timedelta(days=1)
        return out

    def protocol_stats(self, days: int = 14) -> list[dict]:
        """窗口内按协议聚合：请求数/prompt/completion/缓存命中/成本（口径与 daily_stats 一致）。"""
        pr = self.prices
        end = datetime.now(_CST)
        start = (end - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            """SELECT COALESCE(protocol,'unknown') AS protocol,
               COUNT(*) AS requests,
               COALESCE(SUM(prompt_tokens),0) AS prompt,
               COALESCE(SUM(completion_tokens),0) AS completion,
               COALESCE(SUM(prefix_cache_hit_tokens),0) AS cache
               FROM events WHERE day>=? GROUP BY protocol ORDER BY requests DESC""",
            (start,)).fetchall()
        out = []
        for r in rows:
            cache_hit = min(r["cache"] or 0, r["prompt"] or 0)
            api = ((r["prompt"] - cache_hit) * pr["input_price_per_m"]
                   + cache_hit * pr["cache_input_price_per_m"]
                   + (r["completion"] or 0) * pr["output_price_per_m"]) / 1e6
            out.append({"protocol": r["protocol"], "requests": r["requests"],
                        "prompt_tokens": r["prompt"], "completion_tokens": r["completion"],
                        "cache_hit_tokens": cache_hit, "api_cost_cny": api})
        return out

    def totals(self) -> dict[str, float]:
        r = self._conn.execute(
            """SELECT COUNT(*) c, COALESCE(SUM(prompt_tokens),0) p,
               COALESCE(SUM(completion_tokens),0) c_t, MIN(day) first_day
               FROM events""").fetchone()
        return {"requests": r["c"], "prompt_tokens": r["p"],
                "completion_tokens": r["c_t"], "total_tokens": r["p"] + r["c_t"],
                "first_day": r["first_day"]}

    def reset(self) -> None:
        self._conn.execute("DELETE FROM events")
        self._conn.execute("DELETE FROM ingest_state")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _trim(x: float, nd: int) -> str:
    """小数去尾零：20.0 -> '20'；316.8 -> '316.8'。"""
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") or "0"


def fmt_tokens(n: float | int | None) -> str:
    """易读 token 数（中文习惯量级）：9,800 / 20万 / 316.8万 / 2.3亿。"""
    if n is None:
        return "—"
    n = int(float(n))
    a = abs(n)
    if a < 10_000:
        return f"{n:,}"
    if a < 100_000_000:
        return f"{_trim(n / 10_000, 1)}万"
    return f"{_trim(n / 100_000_000, 2)}亿"

def fmt_tokens_en(n: float | int | None) -> str:
    """易读 token 数（英文单位，用于明细表/图表刻度）：9,800 / 20.5k / 3.17M。"""
    if n is None:
        return "—"
    n = int(float(n))
    a = abs(n)
    if a < 1_000:
        return f"{n:,}"
    if a < 1_000_000:
        return f"{_trim(n / 1_000, 1)}k"
    if a < 1_000_000_000:
        return f"{_trim(n / 1_000_000, 2)}M"
    return f"{_trim(n / 1_000_000_000, 2)}B"

