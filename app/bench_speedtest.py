# -*- coding: utf-8 -*-
"""
bench_speedtest.py — 基准测试引擎与 UI 组件（移植自 bench/llm_speedtest-main 的测试逻辑）。

- 引擎部分（纯 Python + requests）：prompt 生成与 token 校准、输入长度阶梯、并发流式
  请求测量（TTFT/ITL/prefill/decode）、按真实墙上时间聚合吞吐。
- UI 部分（PyQt6）：参数表单 + 结果表格 + 进度日志 + 运行/停止。

零新增依赖、无额外进程/服务；API 地址与模型名由启动器按配置自动预填。
测量口径与 llm_speedtest index.html 的 testOpenAI/executeAndMeasureRequest 对齐：
  - prefill 吞吐 = Σprompt_tokens / (最晚首token时刻 − 最早开始时刻)
  - decode 吞吐  = Σoutput_tokens / (最早首token → 最晚结束)
  - token 数优先取响应 usage 字段，缺失时本地估算并标注警告
  - 共享 keep-alive 连接池 + 每长度点前置轻量探测请求（同浏览器口径）
  - 每次运行用随机盐值派生 prompt 种子，避免跨运行的前缀 KV 缓存命中导致
    prefill 速度虚高（NInfer 支持 prefix reuse）
"""

import html as _html
import json
import math
import os
import random
import re
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from collections import deque

import i18n
import palette

i18n.bootstrap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
t = i18n.t

# 配色 token（与启动器共用 app/palette.py；由启动器 apply_theme 同步 _UI）
_TOK = palette.tokens()


def refresh_tokens():
    """启动器切换主题/配色后调用，刷新本组件用到的颜色。"""
    global _TOK
    _TOK = palette.tokens()

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------- 常量（与原工具一致）

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

DEFAULT_BENCHMARK_SUFFIX = (
    "\nBased on the words above, write a short philosophical essay discussing the meaning of "
    "existence, the nature of consciousness, and humanity's place in the universe. "
    "Use clear, coherent sentences."
)

ASCII_FILLER_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"

PROMPT_TOKEN_WORDS = [
    "a", "the", "and", "of", "to", "in", "is", "it", "that", "for", "with", "as",
    "on", "by", "from", "this", "be", "are", "or", "not", "we", "you", "they",
    "can", "will", "if", "all", "one", "time", "world", "life", "work", "data",
    "model", "token", "text", "idea", "mind", "story", "light", "space", "future",
    "human", "system", "simple", "clear", "reason", "change", "value", "truth",
]

SHORT_PROMPT_MAX_LENGTH = 46
BENCHMARK_SUFFIX_TOKEN_LENGTH = SHORT_PROMPT_MAX_LENGTH + 1  # 47
TOKEN_SANITY_MIN_TOKENS = 128
TOKEN_SANITY_MAX_RELATIVE_DIFF = 0.8


def has_positive_number(value):
    """对应 JS hasPositiveNumber。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0

# ---------------------------------------------------------------- prompt 生成


def estimate_token_count(text):
    """与 index.html estimateTokenCount 相同的启发式：CJK 每字 1 token，
    ASCII 字母数字串每 12 字符 1 token，其余非标点字符各 1 token。"""
    if not text:
        return 0
    token_count = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        cp = ord(ch)
        if (0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
                or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF):
            token_count += 1
            i += 1
            continue
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            start = i
            while i < n and (text[i].isascii() and (text[i].isalnum() or text[i] == "_")):
                i += 1
            token_count += max(1, math.ceil((i - start) / 12))
            continue
        token_count += 1
        i += 1
    return max(1, token_count)


def build_token_word_sequence(token_count, seed=0):
    """按 seed 生成 token_count 个单 token 英文词序列（每个词恰为 1 token）。"""
    count = max(int(token_count) if token_count else 0, 0)
    if count <= 0:
        return ""
    rng = random.Random(seed)
    return " ".join(rng.choice(PROMPT_TOKEN_WORDS) for _ in range(count))


def generate_prompt(length, seed=0):
    """生成长度约等于 length tokens 的 prompt。短 prompt 走纯词序列，
    长 prompt = 词序列前缀 + 固定后缀指令。"""
    length = max(int(length), 0)
    if length <= SHORT_PROMPT_MAX_LENGTH:
        return build_token_word_sequence(length, seed)
    suffix_tokens = estimate_token_count(DEFAULT_BENCHMARK_SUFFIX.strip())
    prefix_tokens = max(length - suffix_tokens, 0)
    prefix = build_token_word_sequence(prefix_tokens, seed)
    if prefix:
        return prefix + DEFAULT_BENCHMARK_SUFFIX
    return DEFAULT_BENCHMARK_SUFFIX


def make_ladder(min_length, max_length, step, multiplier):
    """长度阶梯：multiplier==1 线性，否则指数序列 step*mult^n 过滤到 [min, max]。"""
    lengths = []
    if multiplier == 1:
        cur = min_length
        while cur <= max_length:
            lengths.append(cur)
            cur += step
    else:
        idx = 0
        while True:
            cur = int(round(step * (multiplier ** idx)))
            if cur > max_length:
                break
            if cur >= min_length:
                lengths.append(cur)
            idx += 1
    return lengths


def make_session(concurrency):
    """共享 keep-alive 连接池（与浏览器连接复用口径一致），线程安全。"""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=8,
                          pool_maxsize=max(concurrency, 8) + 4)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session



def calibrate_prompt(session, cfg):
    """warmup 请求校准「估算 vs 实际」prompt token 偏移（照搬 sendWarmupRequest）。
    失败返回 None（不阻断测试）。成功后等待 800ms 让服务端稳定（与原实现一致）。"""
    prompt = generate_prompt(96, random.randint(1, 1_000_000))
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8,
        "stream": False,
    }
    try:
        resp = session.post(cfg.api_url, json=body, timeout=max(cfg.timeout_s, 15.0))
        if resp.status_code != 200:
            resp.close()
            return None
        data = resp.json()
        resp.close()
        actual = None
        if data.get("usage"):
            actual = normalize_usage_token_stats(data["usage"])["prompt_tokens"]
        elif data.get("prompt_eval_count") is not None:
            actual = to_token_number(data["prompt_eval_count"])
        if not has_positive_number(actual):
            return None
        estimated = estimate_token_count(DEFAULT_SYSTEM_PROMPT) + estimate_token_count(prompt)
        offset = int(actual) - estimated
        time.sleep(0.8)  # 与原实现一致的 warmup 后稳定等待
        return {"token_offset": offset}
    except Exception:
        return None




# ---------------------------------------------------------------- Token 统计工具（照搬 index.html）

def to_token_number(value):
    """对应 JS toTokenNumber。"""
    if value is None or isinstance(value, bool) or value == '':
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return int(num)  # Math.trunc（对正整数与 int() 等价）


def pick_best_token_value(*values):
    """对应 JS pickBestTokenValue。"""
    parsed = [v for v in map(to_token_number, values) if v is not None]
    if not parsed:
        return None
    positives = [v for v in parsed if v > 0]
    return max(positives) if positives else max(parsed)


def normalize_usage_token_stats(usage):
    """对应 JS normalizeUsageTokenStats（含 reasoning token 处理）。"""
    prompt_tokens = pick_best_token_value(
        usage.get("prompt_tokens"), usage.get("input_tokens"), usage.get("prompt_eval_count"))
    completion_tokens = pick_best_token_value(
        usage.get("completion_tokens"), usage.get("output_tokens"), usage.get("eval_count"))
    reasoning_top = pick_best_token_value(usage.get("reasoning_tokens"))
    reasoning_nested = pick_best_token_value(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
    total_tokens = pick_best_token_value(usage.get("total_tokens"))
    reasoning_tokens = reasoning_top if reasoning_top is not None else reasoning_nested

    output_tokens = None
    if completion_tokens is not None:
        output_tokens = completion_tokens
        if reasoning_top is not None and reasoning_top > 0:
            if prompt_tokens is not None and total_tokens is not None:
                if total_tokens == prompt_tokens + completion_tokens + reasoning_top:
                    output_tokens = completion_tokens + reasoning_top
                elif total_tokens == prompt_tokens + completion_tokens:
                    output_tokens = completion_tokens
                else:
                    output_tokens = max(completion_tokens, completion_tokens + reasoning_top)
            else:
                output_tokens = completion_tokens + reasoning_top
        elif output_tokens <= 0 and reasoning_tokens is not None and reasoning_tokens > 0:
            output_tokens = reasoning_tokens
    elif reasoning_tokens is not None:
        output_tokens = reasoning_tokens
    elif prompt_tokens is not None and total_tokens is not None and total_tokens >= prompt_tokens:
        output_tokens = total_tokens - prompt_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def resolve_token_count_with_sanity_check(kind, api_tokens, estimated_tokens):
    """对应 JS resolveTokenCountWithSanityCheck。"""
    api_value = to_token_number(api_tokens)
    estimated_value = to_token_number(estimated_tokens)
    if not has_positive_number(api_value):
        return {
            "tokens": estimated_value if has_positive_number(estimated_value) else 0,
            "source": "Local Estimation" if has_positive_number(estimated_value) else "Unknown",
            "warning": None,
        }
    if has_positive_number(estimated_value):
        relative_diff = abs(api_value - estimated_value) / max(estimated_value, 1)
        max_tokens = max(api_value, estimated_value)
        if max_tokens >= TOKEN_SANITY_MIN_TOKENS and relative_diff >= TOKEN_SANITY_MAX_RELATIVE_DIFF:
            return {
                "tokens": estimated_value,
                "source": "API Stats Anomaly",
                "warning": {
                    "type": "api_token_mismatch", "kind": kind,
                    "apiTokens": api_value, "estimatedTokens": estimated_value,
                    "relativeDiff": relative_diff,
                },
            }
    return {"tokens": api_value, "source": "API", "warning": None}


def has_llamacpp_style_timings(usage_info):
    """对应 JS hasLlamaCppStyleTimings。"""
    timings = (usage_info or {}).get("timings")
    if not isinstance(timings, dict):
        return False
    return any(has_positive_number(float(timings[k]))
               for k in ("prompt_ms", "predicted_ms", "prompt_n", "predicted_n")
               if k in timings)


def get_average_network_latency_ms(latency_samples):
    """对应 JS getAverageNetworkLatencyMs（≥3 个样本时去掉最大最小值再平均）。"""
    valid = [v for v in latency_samples if math.isfinite(v) and v > 0]
    if not valid:
        return 0
    if len(valid) >= 3:
        valid = sorted(valid)[1:-1]
    return sum(valid) / len(valid)


# ---------------------------------------------------------------- 网络延迟采样（照搬 index.html）

def resolve_model_catalog_endpoint(api_url):
    """对应 JS resolveModelCatalogEndpoint：推导非推理类延迟探测端点。"""
    normalized = (api_url or '').strip().rstrip('/')
    if not normalized:
        return ''
    if normalized.endswith('/v1/chat/completions'):
        return normalized[:-len('/v1/chat/completions')] + '/v1/models'
    if normalized.endswith('/v1/models'):
        return normalized
    return ''


def measure_network_latency_sample(session, cfg):
    """对应 JS measureNetworkLatencySample：GET 模型目录端点测 RTT，不触发推理。"""
    endpoint = resolve_model_catalog_endpoint(cfg.api_url)
    if not endpoint:
        return None
    probe_timeout_s = min(max(cfg.timeout_s, 3.0), 8.0)
    start = time.perf_counter()
    try:
        resp = session.get(endpoint, timeout=probe_timeout_s)
        resp.text  # 读完整响应
        resp.close()
        return (time.perf_counter() - start) * 1000.0
    except Exception:
        return None


def collect_network_latency_sample(samples, session, cfg):
    """对应 JS collectNetworkLatencySample：采样入列（最多20个），返回当前平均值。"""
    try:
        sample = measure_network_latency_sample(session, cfg)
        if sample is not None and math.isfinite(sample) and sample > 0:
            samples.append(sample)
            if len(samples) > 20:
                samples.popleft()
    except Exception:
        pass
    return get_average_network_latency_ms(samples)


# ---------------------------------------------------------------- 流式测量（照搬 executeAndMeasureRequest）

def execute_and_measure_request(session, cfg, prompt, prompt_length,
                                 prompt_token_estimate=None,
                                 network_latency_ms=0, network_latency_sample_count=0):
    """单次流式请求的完整测量（时间戳、usage/timing 提取、token 校验、速度计算）。
    失败重试语义与 createFetchPromiseWithRetry(…, 3, 1500) 一致：共 3 次尝试。"""
    start_time = time.perf_counter()
    first_token_time = None
    output_content = ''
    reasoning_content = ''
    actual_output_tokens_from_usage = None
    actual_prompt_tokens_from_usage = None
    server_prefill_time_ms = None
    server_decode_time_ms = None
    usage_info = None
    cached_tokens_total = 0

    if has_positive_number(prompt_token_estimate):
        actual_prompt_tokens_from_usage = prompt_token_estimate

    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": cfg.output_len,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "presence_penalty": cfg.presence_penalty,
        "frequency_penalty": cfg.frequency_penalty,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response = None
    last_err = None
    for attempt in range(3):
        try:
            response = session.post(cfg.api_url, json=body, stream=True,
                                    timeout=cfg.timeout_s)
            break
        except Exception as e:
            last_err = e
            if attempt == 2:
                raise RuntimeError(t("E_req_fail", e=last_err))
            time.sleep(1.5)

    if response.status_code != 200:
        error_text = ''
        try:
            error_text = response.text[:200]
        finally:
            response.close()
        raise RuntimeError(t("E_http", s=response.status_code, e=error_text))

    deadline = start_time + cfg.timeout_s
    lines_iter = response.iter_lines(decode_unicode=True)
    while True:
        try:
            raw = next(lines_iter)
        except StopIteration:
            break
        if time.perf_counter() > deadline:
            response.close()
            raise TimeoutError(t("E_timeout"))
        if not raw or not raw.startswith('data:'):
            continue
        payload = raw[5:].strip()
        if payload == '[DONE]':
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # Ollama 兼容分支（与 JS 一致，保留）
        if data.get('prompt_eval_count') is not None:
            p = data.get('prompt_eval_count') or 0
            c = data.get('eval_count') or 0
            actual_output_tokens_from_usage = c
            if has_positive_number(p):
                actual_prompt_tokens_from_usage = p
            usage_info = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
            if data.get('prompt_eval_duration'):
                server_prefill_time_ms = data['prompt_eval_duration'] / 1e6
            if data.get('eval_duration'):
                server_decode_time_ms = data['eval_duration'] / 1e6

        usage = data.get('usage')
        if usage:
            stats = normalize_usage_token_stats(usage)
            actual_output_tokens_from_usage = stats["output_tokens"]
            if has_positive_number(stats["prompt_tokens"]):
                actual_prompt_tokens_from_usage = stats["prompt_tokens"]
            # 前缀缓存命中 token 数（优化 prefix reuse 的关键观测指标）
            details = usage.get("prompt_tokens_details") or {}
            cached = to_token_number(details.get("cached_tokens"))
            cached_tokens_total = cached if cached and cached > 0 else 0
            usage_info = dict(usage)
            if data.get('timings'):
                usage_info['timings'] = dict(data['timings'])
            if usage.get('prompt_eval_duration'):
                server_prefill_time_ms = usage['prompt_eval_duration'] / 1e6
            elif usage.get('prompt_eval_time'):
                server_prefill_time_ms = usage['prompt_eval_time']
            if usage.get('eval_duration'):
                server_decode_time_ms = usage['eval_duration'] / 1e6
            elif usage.get('eval_time'):
                server_decode_time_ms = usage['eval_time']

        timings = data.get('timings')
        if isinstance(timings, dict):
            if not server_prefill_time_ms:
                if timings.get('prompt_eval_duration'):
                    server_prefill_time_ms = timings['prompt_eval_duration'] / 1e6
                elif timings.get('prompt_ms'):
                    server_prefill_time_ms = timings['prompt_ms']
            if not server_decode_time_ms:
                if timings.get('eval_duration'):
                    server_decode_time_ms = timings['eval_duration'] / 1e6
                elif timings.get('predicted_ms'):
                    server_decode_time_ms = timings['predicted_ms']

        content_text = None
        is_reasoning = False
        if data.get('response') is not None:
            content_text = data['response']
        choices = data.get('choices') or []
        choice = choices[0] if choices else None
        if choice:
            delta = choice.get('delta') or {}
            if delta.get('reasoning_content'):
                content_text = delta['reasoning_content']; is_reasoning = True
            elif delta.get('reasoning'):
                content_text = delta['reasoning']; is_reasoning = True
            elif delta.get('content'):
                content_text = delta['content']
            elif (choice.get('message') or {}).get('content'):
                content_text = choice['message']['content']
            elif choice.get('text'):
                content_text = choice['text']
        elif data.get('message'):
            if data['message'].get('thinking'):
                content_text = data['message']['thinking']; is_reasoning = True
            elif data['message'].get('content'):
                content_text = data['message']['content']

        if content_text:
            if is_reasoning:
                reasoning_content += content_text
            else:
                output_content += content_text
            if first_token_time is None:
                first_token_time = time.perf_counter()

        if data.get('done') is True:
            break

    end_time = time.perf_counter()
    response.close()

    if first_token_time is None:
        if not (reasoning_content + output_content):
            raise RuntimeError(t("E_notok"))
        first_token_time = end_time

    # token 数解析（与 JS 相同：usage 优先 + sanity check 回退本地估算）
    local_prompt_estimate = prompt_token_estimate if has_positive_number(prompt_token_estimate) else prompt_length
    token_warnings = []
    prompt_resolution = resolve_token_count_with_sanity_check(
        'prompt', actual_prompt_tokens_from_usage, local_prompt_estimate)
    actual_prompt_tokens = (prompt_resolution["tokens"]
                            if has_positive_number(prompt_resolution["tokens"])
                            else local_prompt_estimate)
    if prompt_resolution["warning"]:
        token_warnings.append(prompt_resolution["warning"])

    local_output_estimate = (estimate_token_count(reasoning_content)
                             + estimate_token_count(output_content)
                             if (reasoning_content or output_content) else None)
    if actual_output_tokens_from_usage and actual_output_tokens_from_usage > 0:
        output_resolution = resolve_token_count_with_sanity_check(
            'decode', actual_output_tokens_from_usage, local_output_estimate)
        received_tokens = output_resolution["tokens"]
        token_source = ('API Stats Anomaly' if output_resolution["source"] == 'API Stats Anomaly'
                        else 'API')
        if output_resolution["warning"]:
            token_warnings.append(output_resolution["warning"])
    elif reasoning_content or output_content:
        received_tokens = local_output_estimate
        token_source = 'Local Estimation'
    else:
        received_tokens = 0
        token_source = 'Unknown'
    if prompt_resolution["warning"] and token_source == 'API':
        token_source = 'API Stats Anomaly'

    # 时间来源选择（服务器 timing 优先，否则客户端计时并扣除网络延迟）
    latency_adjustment_ms = network_latency_ms if has_positive_number(network_latency_ms) else 0
    client_prefill_time_ms = max((first_token_time - start_time) * 1000.0 - latency_adjustment_ms, 1)
    client_output_time_ms = max((end_time - first_token_time) * 1000.0, 1)
    use_server_prefill = has_positive_number(server_prefill_time_ms)
    use_server_decode = has_positive_number(server_decode_time_ms)
    prefill_time_ms = server_prefill_time_ms if use_server_prefill else client_prefill_time_ms
    output_time_ms = server_decode_time_ms if use_server_decode else client_output_time_ms
    prefill_time_source = ('server' if use_server_prefill
                           else ('client_latency_adjusted' if latency_adjustment_ms > 0 else 'client'))
    output_time_source = 'server' if use_server_decode else 'client'

    prefill_speed = actual_prompt_tokens / (prefill_time_ms / 1000.0)
    output_speed = received_tokens / (output_time_ms / 1000.0) if received_tokens > 0 else 0
    ttft_ms = (first_token_time - start_time) * 1000.0
    itl_mean = output_time_ms / (received_tokens - 1) if received_tokens > 1 else 0

    return {
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "first_token_timestamp": first_token_time,
        "ttft_ms": ttft_ms,
        "itl_mean": itl_mean,
        "prefill_time_ms": prefill_time_ms,
        "output_time_ms": output_time_ms,
        "prefill_speed": prefill_speed,
        "output_speed": output_speed,
        "prompt_tokens": actual_prompt_tokens,
        "output_tokens": received_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "server_prefill_time_ms": server_prefill_time_ms,
        "server_decode_time_ms": server_decode_time_ms,
        "prefill_time_source": prefill_time_source,
        "output_time_source": output_time_source,
        "network_latency_ms": latency_adjustment_ms,
        "network_latency_sample_count": network_latency_sample_count,
        "client_prefill_duration_ms": client_prefill_time_ms,
        "client_decode_duration_ms": client_output_time_ms,
        "usage_info": usage_info,
        "cached_tokens": cached_tokens_total,
        "token_source": token_source,
        "token_warnings": token_warnings,
    }


def calculate_aggregate_throughput_metrics(results):
    """照搬 JS calculateAggregateThroughputMetrics（含 llamacpp timing 异常回退逻辑）。"""
    empty = {
        "total_prompt_tokens": 0, "total_output_tokens": 0,
        "prefill_duration_ms": 1, "decode_duration_ms": 1,
        "client_prefill_duration_ms": 1, "client_decode_duration_ms": 1,
        "client_total_duration_ms": 1, "client_has_phase_boundary": False,
        "prefill_source": 'client', "decode_source": 'client', "warnings": [],
    }
    if not results:
        return empty

    total_prompt = sum(r.get("actual_prompt_tokens") or 0 for r in results)
    total_output = sum(r.get("output_tokens") or 0 for r in results)
    warnings = []
    avg_latency = get_average_network_latency_ms(
        [r.get("network_latency_ms", 0) for r in results])

    start_times = [r["start_timestamp"] for r in results if math.isfinite(r["start_timestamp"])]
    end_times = [r["end_timestamp"] for r in results if math.isfinite(r["end_timestamp"])]
    boundary_times = [r.get("first_token_timestamp") for r in results
                      if r.get("first_token_timestamp") is not None
                      and math.isfinite(r["first_token_timestamp"])]
    has_client_timing_window = (len(start_times) == len(results)
                                and len(end_times) == len(results))
    has_client_phase_boundary = (has_client_timing_window
                                 and len(boundary_times) == len(results))
    min_start = min(start_times) if start_times else 0
    max_end = max(end_times) if end_times else min_start + 1
    max_boundary = max(boundary_times) if has_client_phase_boundary else max_end
    min_decode_start = min(boundary_times) if has_client_phase_boundary else None

    client_total = (max((max_end - min_start) * 1000.0 - avg_latency, 1)
                    if has_client_timing_window else None)
    client_prefill = (max((max_boundary - min_start) * 1000.0 - avg_latency, 1)
                      if has_client_timing_window else None)
    client_decode = (max((max_end - min_decode_start) * 1000.0, 1)
                     if has_client_phase_boundary else None)
    client_prefill_for_use = client_prefill if client_prefill is not None else 1
    client_decode_for_use = (client_decode if client_decode is not None
                             else client_total if client_total is not None else 1)
    client_total_for_use = (client_total if client_total is not None
                            else max(client_prefill_for_use + client_decode_for_use, 1))

    use_server_prefill = all(
        has_positive_number(r.get("server_prefill_time_ms"))
        and (r.get("prefill_time_source") in (None, 'server'))
        for r in results)
    server_prefill_durations = [r["server_prefill_time_ms"] for r in results
                                if has_positive_number(r.get("server_prefill_time_ms"))]
    if use_server_prefill:
        prefill_duration_ms = max(server_prefill_durations) if server_prefill_durations else client_prefill_for_use
        prefill_source = 'server'
    else:
        prefill_duration_ms = client_prefill_for_use
        prefill_source = ('client_latency_adjusted' if avg_latency > 0 else 'client')

    use_server_decode = all(
        has_positive_number(r.get("server_decode_time_ms"))
        and (r.get("output_time_source") in (None, 'server'))
        for r in results)
    server_decode_durations = [r["server_decode_time_ms"] for r in results
                               if has_positive_number(r.get("server_decode_time_ms"))]
    if use_server_decode:
        decode_duration_ms = max(server_decode_durations) if server_decode_durations else client_decode_for_use
        decode_source = 'server'
    else:
        decode_duration_ms = client_decode_for_use
        decode_source = 'client'

    # llamacpp 并发 timing 异常检测（与 JS 完全一致）
    has_concurrent_llamacpp = (len(results) > 1
                               and any(has_llamacpp_style_timings(r.get("usage_info")) for r in results))
    server_total = sum(server_prefill_durations) + sum(server_decode_durations)
    inferred_parallelism = 0
    if (has_concurrent_llamacpp and has_client_phase_boundary
            and server_total > 0 and client_total_for_use > 0):
        inferred_parallelism = min(len(results),
                                   max(1, server_total / client_total_for_use))
    has_limited = inferred_parallelism > 0 and inferred_parallelism < len(results) * 0.8
    if has_limited and use_server_prefill and server_prefill_durations:
        prefill_duration_ms = client_prefill_for_use
        prefill_source = 'client_llamacpp_timing_anomaly'
        warnings.append({
            "type": "api_timing_mismatch", "kind": "prefill_timing",
            "reason": "llamacpp_limited_parallel_slots",
            "apiTimeMs": round(max(server_prefill_durations), 2),
            "clientTimeMs": round(client_prefill_for_use, 2),
            "inferredParallelism": round(inferred_parallelism, 2),
            "configuredConcurrency": len(results),
        })
    if has_limited and use_server_decode and server_decode_durations:
        decode_duration_ms = client_total_for_use
        decode_source = 'client_llamacpp_timing_anomaly'
        warnings.append({
            "type": "api_timing_mismatch", "kind": "decode_timing",
            "reason": "llamacpp_limited_parallel_slots",
            "apiTimeMs": round(max(server_decode_durations), 2),
            "clientTimeMs": round(client_total_for_use, 2),
            "inferredParallelism": round(inferred_parallelism, 2),
            "configuredConcurrency": len(results),
        })

    return {
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "prefill_duration_ms": max(prefill_duration_ms, 1),
        "decode_duration_ms": max(decode_duration_ms, 1),
        "client_prefill_duration_ms": client_prefill,
        "client_decode_duration_ms": client_decode,
        "client_total_duration_ms": client_total,
        "client_has_phase_boundary": has_client_phase_boundary,
        "prefill_source": prefill_source,
        "decode_source": decode_source,
        "warnings": warnings,
    }


class BenchmarkConfig:
    def __init__(self, api_url, model, min_length, max_length, step, multiplier,
                 output_len, concurrency, timeout_s,
                 temperature=0.0, top_p=1.0, presence_penalty=0.0, frequency_penalty=0.0):
        self.api_url = api_url
        self.model = model
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.step = int(step)
        self.multiplier = float(multiplier)
        self.output_len = int(output_len)
        self.concurrency = int(concurrency)
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.presence_penalty = float(presence_penalty)
        self.frequency_penalty = float(frequency_penalty)


def run_benchmark(cfg, log_cb, row_cb, stop_event):
    """照搬 JS testOpenAI 的执行流程：warmup 校准 → 延迟采样 → 逐长度点并发测量与聚合。"""
    lengths = make_ladder(cfg.min_length, cfg.max_length, cfg.step, cfg.multiplier)
    if not lengths:
        log_cb(t("L_err_ladder_empty"))
        return
    if len(lengths) > 256:
        log_cb(t("L_err_ladder_many", n=len(lengths)))
        return
    log_cb(t("L_testpts", pts=", ".join(map(str, lengths)), c=cfg.concurrency)
           + t("L_outlen", o=cfg.output_len, t=f"{cfg.timeout_s:.0f}"))

    # 每次运行随机盐值：prompt 内容跨运行不同，避免前缀 KV 缓存命中导致 prefill 虚高
    salt = random.randrange(1 << 30)
    session = make_session(cfg.concurrency)
    latency_samples = deque()

    # warmup 校准（sendWarmupRequest）
    log_cb(t("L_warmup"))
    calibration = calibrate_prompt(session, cfg)
    if calibration is None:
        log_cb(t("L_warn_warmup"))
    else:
        log_cb(t("L_calib", o=calibration["token_offset"]))

    # 首轮两次延迟采样（与 JS 一致）
    collect_network_latency_sample(latency_samples, session, cfg)
    collect_network_latency_sample(latency_samples, session, cfg)

    sys_tokens = estimate_token_count(DEFAULT_SYSTEM_PROMPT)
    stats = {"min_pre": float('inf'), "max_pre": 0.0,
             "min_dec": float('inf'), "max_dec": 0.0,
             "sum_pre": 0.0, "sum_dec": 0.0, "n": 0}

    total = len(lengths)
    for idx, length in enumerate(lengths, 1):
        if stop_event.is_set():
            log_cb(t("L_stop_aborted"))
            break
        network_latency_ms = collect_network_latency_sample(latency_samples, session, cfg)
        log_cb(t("L_point_head", i=idx, n=total, L=length)
               + t("L_latency", x=f"{network_latency_ms:.1f}", cnt=len(latency_samples)))

        user_len = length
        if calibration and length > SHORT_PROMPT_MAX_LENGTH:
            user_len = max(BENCHMARK_SUFFIX_TOKEN_LENGTH,
                           round(length - sys_tokens - calibration["token_offset"]))

        results, failed_msgs = [None] * cfg.concurrency, []
        workers = []
        for i in range(cfg.concurrency):
            prompt = generate_prompt(user_len, seed=salt + i + 1)
            estimate = apply_prompt_calibration(
                estimate_token_count(DEFAULT_SYSTEM_PROMPT) + estimate_token_count(prompt),
                calibration)

            def worker(prompt=prompt, estimate=estimate):
                try:
                    return execute_and_measure_request(
                        session, cfg, prompt, user_len, estimate,
                        network_latency_ms, len(latency_samples))
                except Exception as e:
                    log_cb(t("L_warn_fail", t=type(e).__name__, e=e))
                    return None

            # 线程化并发执行（与 Promise.allSettled 相同语义）
            th = threading.Thread(target=_collect_result, args=(results, i, worker), daemon=True)
            th.start()
            workers.append(th)
        for th in workers:
            th.join()

        ok = [r for r in results if r is not None]
        if not ok:
            log_cb(t("L_allfail", i=idx, n=total, L=length))
            row_cb({"length": length, "status": t("L_zero_fail", c=cfg.concurrency), "error": True})
            continue

        avg_output = sum(r["output_tokens"] for r in ok) / len(ok)
        avg_ttft = sum(r["ttft_ms"] for r in ok) / len(ok)
        avg_itl = sum(r["itl_mean"] for r in ok) / len(ok)
        avg_actual_prompt = sum((r["actual_prompt_tokens"] or length) for r in ok) / len(ok)

        agg = calculate_aggregate_throughput_metrics(ok)
        prefill_speed = (agg["total_prompt_tokens"] / (agg["prefill_duration_ms"] / 1000.0)
                         if agg["prefill_duration_ms"] > 0 else 0)
        decode_speed = (agg["total_output_tokens"] / (agg["decode_duration_ms"] / 1000.0)
                        if agg["decode_duration_ms"] > 0 else 0)
        total_cached = sum(r.get("cached_tokens", 0) for r in ok)
        cache_pct = (total_cached / agg["total_prompt_tokens"] * 100.0
                     if agg["total_prompt_tokens"] > 0 else 0.0)

        stats["min_pre"] = min(stats["min_pre"], prefill_speed)
        stats["max_pre"] = max(stats["max_pre"], prefill_speed)
        stats["min_dec"] = min(stats["min_dec"], decode_speed)
        stats["max_dec"] = max(stats["max_dec"], decode_speed)
        stats["sum_pre"] += prefill_speed
        stats["sum_dec"] += decode_speed
        stats["n"] += 1

        token_warnings = [w for r in ok for w in r["token_warnings"]]
        status_text = t("L_conc_ok", a=len(ok), b=cfg.concurrency)
        if token_warnings:
            status_text += " " + t("L_api_anomaly")
        warn_src = (f" [Prefill={agg['prefill_source']}, Decode={agg['decode_source']}]"
                    if agg['prefill_source'] != 'client' or agg['decode_source'] != 'client' else "")
        log_cb(t("L_pt_actual", i=idx, n=total, L=int(round(avg_actual_prompt)))
               + f"prefill={prefill_speed:.2f} tok/s ({agg['prefill_duration_ms']:.2f}ms), "
                 f"decode={decode_speed:.2f} tok/s ({agg['decode_duration_ms']:.2f}ms), "
                 f"TTFT={avg_ttft:.2f}ms, ITL={avg_itl:.2f}ms, "
                 + t("L_cache", a=total_cached, b=agg["total_prompt_tokens"],
                     p=f"{cache_pct:.1f}", w=warn_src))
        row_cb({
            "length": int(round(avg_actual_prompt)),
            "prefill_ms": agg["prefill_duration_ms"],
            "prefill": prefill_speed,
            "out_tokens": avg_output,
            "output_ms": agg["decode_duration_ms"],
            "decode": decode_speed,
            "cache": f"{cache_pct:.1f}%",
            "status": status_text,
            "error": False,
        })

    if stats["n"] > 0:
        log_cb(t("L_sum_pre", a=f"{stats['min_pre']:.2f}",
                 b=f"{stats['sum_pre']/stats['n']:.2f}", c=f"{stats['max_pre']:.2f}")
               + t("L_sum_dec", a=f"{stats['min_dec']:.2f}",
                   b=f"{stats['sum_dec']/stats['n']:.2f}", c=f"{stats['max_dec']:.2f}",
                   n=stats["n"]))
    log_cb(t("L_done") + (t("L_manual_stop") if stop_event.is_set() else ""))


def _collect_result(results, index, worker):
    results[index] = worker()


def apply_prompt_calibration(estimated_prompt_tokens, calibration):
    """对应 JS applyPromptCalibration。"""
    estimate = max(int(estimated_prompt_tokens or 0), 1)
    if not calibration or not math.isfinite(calibration.get("token_offset", 0)):
        return estimate
    return max(1, estimate + int(calibration["token_offset"]))




# ---------------------------------------------------------------- UI 组件


class _SignalBridge(QObject):
    sig_row = pyqtSignal(object)
    sig_log = pyqtSignal(str)
    sig_done = pyqtSignal(str)


class SpeedTestWidget(QWidget):
    """Benchmark tab 内容：参数表单 + 运行/停止 + 结果表格 + 日志。"""

    COLUMNS_KEYS = ("b_th_in", "b_th_pre_ms", "b_th_pre_tps", "b_th_out",
                    "b_th_dec_ms", "b_th_dec_tps", "b_th_cache", "b_th_stat")
    COLUMNS = [t(k) for k in COLUMNS_KEYS]

    def __init__(self, parent=None, make_defaults=None):
        super().__init__(parent)
        # make_defaults() -> (api_url, model)，运行时从启动器配置读取
        self._make_defaults = make_defaults or (lambda: ("http://127.0.0.1:8080/v1/chat/completions", ""))
        self._bridge = _SignalBridge(self)
        self._stop_event = threading.Event()
        self._running = False
        self._build_ui()
        self._bridge.sig_row.connect(self._on_row)
        self._bridge.sig_log.connect(self._on_log)
        self._bridge.sig_done.connect(self._on_done)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form_w = QGroupBox(t("bench_params"))
        self._form_grp = form_w
        form = QFormLayout(form_w)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.ed_api = QLineEdit()
        self.ed_model = QLineEdit()
        sync_btn = QPushButton(t("bench_sync"))
        self._btn_sync = sync_btn

        self.cb_conc = QComboBox(); self.cb_conc.setEditable(True)
        self.cb_conc.addItems(["1", "2", "4", "8", "16", "32", "64"]); self.cb_conc.setCurrentText("2")
        self.cb_minlen = QComboBox(); self.cb_minlen.setEditable(True)
        self.cb_minlen.addItems(["128", "256", "512", "1024"]); self.cb_minlen.setCurrentText("512")
        self.cb_maxlen = QComboBox(); self.cb_maxlen.setEditable(True)
        self.cb_maxlen.addItems(["512", "1024", "2048", "4096", "8192", "16384", "32768"])
        self.cb_maxlen.setCurrentText("8192")
        self.cb_step = QComboBox(); self.cb_step.setEditable(True)
        self.cb_step.addItems(["128", "256", "512", "1024"]); self.cb_step.setCurrentText("128")
        self.cb_mult = QComboBox(); self.cb_mult.setEditable(True)
        self.cb_mult.addItems(["1", "2", "4"]); self.cb_mult.setCurrentText("2")
        self.cb_olen = QComboBox(); self.cb_olen.setEditable(True)
        self.cb_olen.addItems(["64", "128", "256", "512", "1024"]); self.cb_olen.setCurrentText("256")
        self.cb_timeout = QComboBox(); self.cb_timeout.setEditable(True)
        self.cb_timeout.addItems(["30", "60", "120", "300"]); self.cb_timeout.setCurrentText("120")

        api_row = QHBoxLayout()
        api_row.addWidget(self.ed_api, 1)
        api_row.addWidget(sync_btn)
        self._api_row = api_row
        form.addRow(t("bench_apiurl"), api_row)
        form.addRow(t("bench_model"), self.ed_model)
        form.addRow(t("bench_conc"), self.cb_conc)
        form.addRow(t("bench_minlen"), self.cb_minlen)
        form.addRow(t("bench_maxlen"), self.cb_maxlen)
        form.addRow(t("bench_step"), self.cb_step)
        form.addRow(t("bench_mult"), self.cb_mult)
        form.addRow(t("bench_outlen"), self.cb_olen)
        form.addRow(t("bench_timeout"), self.cb_timeout)
        sync_btn.clicked.connect(self._sync_defaults)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton(t("bench_run"))
        self.btn_run.setStyleSheet(
            ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
             "stop:0 %s, stop:1 %s);color:%s;font-weight:bold;"
             "border:none;border-radius:7px;padding:8px 20px;}"
             "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
             "stop:0 %s, stop:1 %s);}"
             "QPushButton:disabled{background:%s;}") % (
                _TOK["accent_lo"], _TOK["accent_hi"], _TOK["text_on_accent"],
                _TOK["accent_hover_lo"], _TOK["accent_hover_hi"], _TOK["text_off"]))
        self.btn_stop = QPushButton(t("bench_stop"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
             "stop:0 %s, stop:1 %s);color:%s;font-weight:bold;"
             "border:none;border-radius:7px;padding:8px 20px;}"
             "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
             "stop:0 %s, stop:1 %s);}"
             "QPushButton:disabled{background:%s;}") % (
                _TOK["danger_border"], _TOK["danger"], _TOK["text_on_accent"],
                _TOK["danger_border"], _TOK["danger"], _TOK["text_off"]))
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(28)
        hf = QFont(); hf.setBold(True)
        self.table.horizontalHeader().setFont(hf)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setMinimumHeight(160)

        root.addWidget(form_w)
        root.addLayout(btn_row)
        root.addWidget(self.table, 3)
        self._log_lbl = QLabel(t("bench_log"))
        root.addWidget(self._log_lbl)
        root.addWidget(self.log_view, 2)

        self.btn_run.clicked.connect(self.run)
        self.btn_stop.clicked.connect(self.stop)
        self._sync_defaults()

    def _sync_defaults(self):
        url, model = self._make_defaults()
        self.ed_api.setText(url)
        self.ed_model.setText(model)
        self._append_log(t("bench_synced", u=url, m=model))

    def retranslate(self):
        """语言切换后重刷本组件的静态文本（表单标签/按钮/表头/日志标题）。"""
        self._form_grp.setTitle(t("bench_params"))
        form = self._form_grp.layout()
        lbl = form.labelForField(self._api_row)
        if lbl is not None:
            lbl.setText(t("bench_apiurl"))
        for w, key in ((self.ed_model, "bench_model"), (self.cb_conc, "bench_conc"),
                       (self.cb_minlen, "bench_minlen"), (self.cb_maxlen, "bench_maxlen"),
                       (self.cb_step, "bench_step"), (self.cb_mult, "bench_mult"),
                       (self.cb_olen, "bench_outlen"), (self.cb_timeout, "bench_timeout")):
            lb = form.labelForField(w)
            if lb is not None:
                lb.setText(t(key))
        self._btn_sync.setText(t("bench_sync"))
        self._log_lbl.setText(t("bench_log"))
        if self.btn_run.isEnabled():
            self.btn_run.setText(t("bench_run"))
        if self.btn_stop.isEnabled():
            self.btn_stop.setText(t("bench_stop"))
        self.table.setHorizontalHeaderLabels([t(k) for k in self.COLUMNS_KEYS])

    # ---------------------------------------------------------- 运行控制

    def _combo_num(self, cb, name, lo, hi):
        try:
            v = float(cb.currentText().strip())
            iv = int(v) if v == int(v) else v
            if not (lo <= iv <= hi):
                raise ValueError
            return iv
        except ValueError:
            self._append_log(t("bench_err_param", n=name, v=cb.currentText(), lo=lo, hi=hi))
            return None

    def run(self):
        if self._running:
            return
        conc = self._combo_num(self.cb_conc, t("bench_conc"), 1, 128)
        mn = self._combo_num(self.cb_minlen, t("bench_minlen"), 1, 1000000)
        mx = self._combo_num(self.cb_maxlen, t("bench_maxlen"), 1, 1000000)
        step = self._combo_num(self.cb_step, t("bench_step"), 1, 1000000)
        mult = self._combo_num(self.cb_mult, t("bench_mult_name"), 1, 100)
        olen = self._combo_num(self.cb_olen, t("bench_outlen"), 1, 100000)
        tout = self._combo_num(self.cb_timeout, t("bench_timeout_name"), 1, 3600)
        if None in (conc, mn, mx, step, mult, olen, tout):
            return
        if mn > mx:
            self._append_log(t("bench_err_minmax"))
            return

        cfg = BenchmarkConfig(
            api_url=self.ed_api.text().strip(),
            model=self.ed_model.text().strip(),
            min_length=mn, max_length=mx, step=step, multiplier=mult,
            output_len=olen, concurrency=conc, timeout_s=tout,
        )
        if not cfg.api_url or not cfg.model:
            self._append_log(t("bench_err_empty"))
            return

        self._running = True
        self._stop_event.clear()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.table.setRowCount(0)
        self.log_view.clear()

        def worker():
            try:
                run_benchmark(cfg,
                              log_cb=lambda t: self._bridge.sig_log.emit(t),
                              row_cb=lambda r: self._bridge.sig_row.emit(r),
                              stop_event=self._stop_event)
            except Exception as e:
                self._bridge.sig_log.emit(t("bench_exc", t=type(e).__name__, e=e))
            finally:
                self._bridge.sig_done.emit("")

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self._running:
            self._stop_event.set()
            self._append_log(t("bench_stopping"))

    # ---------------------------------------------------------- 槽函数

    def _cell(self, text, color=None, bold=False):
        item = QTableWidgetItem(text)
        if color:
            item.setForeground(QColor(color))
        if bold:
            f = item.font(); f.setBold(True); item.setFont(f)
        return item

    def _on_row(self, row):
        r = self.table.rowCount()
        self.table.insertRow(r)
        if row.get("error"):
            self.table.setItem(r, 0, self._cell(str(row["length"]), bold=True))
            self.table.setItem(r, 7, self._cell(row["status"], _TOK["danger_text"], bold=True))
            return
        # 颜色语义：速度=主角数字（蓝/绿加粗），长度加粗，其余保持默认
        self.table.setItem(r, 0, self._cell(str(row["length"]), bold=True))
        self.table.setItem(r, 1, self._cell(f"{row['prefill_ms']:.2f}"))
        self.table.setItem(r, 2, self._cell(f"{row['prefill']:.2f}", _TOK["accent_text"], bold=True))
        self.table.setItem(r, 3, self._cell(f"{row['out_tokens']:.0f}"))
        self.table.setItem(r, 4, self._cell(f"{row['output_ms']:.2f}"))
        self.table.setItem(r, 5, self._cell(f"{row['decode']:.2f}", _TOK["ok"], bold=True))
        # 缓存命中：0% 正常；>0 琥珀；≥50% 红（严重污染基准）
        cache_txt = row.get("cache", "-")
        try:
            pct = float(cache_txt.rstrip('%'))
        except ValueError:
            pct = 0.0
        cache_color = _TOK["ok"] if pct == 0 else (_TOK["warn_text"] if pct < 50 else _TOK["danger_text"])
        self.table.setItem(r, 6, self._cell(cache_txt, cache_color,
                                            bold=(pct >= 50)))
        # 状态：全部成功绿，部分成功琥珀，全失败红
        status = row["status"]
        if status.startswith("0/"):
            s_color = _TOK["danger_text"]
        elif t("res_fail") in status:
            s_color = _TOK["warn_text"]
        else:
            s_color = _TOK["ok"]
        self.table.setItem(r, 7, self._cell(status, s_color))
        for c in range(1, 7):
            it = self.table.item(r, c)
            if it:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for c in range(len(self.COLUMNS)):
            self.table.resizeColumnToContents(c)

    def _on_log(self, text):
        self._append_log(text)

    def _on_done(self, _):
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # 汇总行（速度列：Prefill=列2，Decode=列5）
        pre, dec = [], []
        for r in range(self.table.rowCount()):
            try:
                pre.append(float(self.table.item(r, 2).text()))
                dec.append(float(self.table.item(r, 5).text()))
            except (ValueError, AttributeError, TypeError):
                continue
        if pre:
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = QTableWidgetItem(t("res_agg"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            font = item.font(); font.setBold(True)
            item.setFont(font)
            self.table.setItem(r, 0, item)
            self.table.setItem(r, 2, self._cell(f"{sum(pre)/len(pre):.1f}", _TOK["accent_text"], bold=True))
            self.table.setItem(r, 5, self._cell(f"{sum(dec)/len(dec):.1f}", _TOK["ok"], bold=True))
            self.table.setItem(r, 7, self._cell(t("res_avg", n=len(pre)), _TOK["text_muted"]))

    def _colorize_log(self, text):
        """按语义给日志行着色：错误红/警告琥珀/汇总蓝/完成绿；结果行数字高亮。"""
        esc = _html.escape(text)
        if text.startswith(t("p_err")):
            return '<span style="color:' + _TOK["danger_text"] + ';font-weight:bold">' + esc + '</span>'
        if text.startswith(t("p_warn")):
            return '<span style="color:' + _TOK["warn_text"] + '">' + esc + '</span>'
        if text.startswith(t("p_sum")):
            return '<span style="color:' + _TOK["accent_text"] + ';font-weight:bold">' + esc + '</span>'
        if text.startswith(t("p_done")) or text.startswith(t("p_stop")):
            return '<span style="color:' + _TOK["ok"] + '">' + esc + '</span>'
        if "prefill=" in text and "decode=" in text:
            esc = re.sub(r'(prefill=\d+\.\d+) tok/s',
                         r'<span style="color:' + _TOK["accent_text"] + r';font-weight:bold">\1</span> tok/s', esc)
            esc = re.sub(r'(decode=\d+\.\d+) tok/s',
                         r'<span style="color:' + _TOK["ok"] + r';font-weight:bold">\1</span> tok/s', esc)
            esc = re.sub(r'(TTFT=\d+\.\d+ms)',
                         r'<span style="color:' + _TOK["warn_text"] + r'">\1</span>', esc)
            esc = re.sub(r'(ITL=\d+\.\d+ms)',
                         r'<span style="color:' + _TOK["warn_text"] + r'">\1</span>', esc)
            m = re.search(r'(?:缓存命中|cache hit) (\d+)/(\d+) \((\d+\.\d)%\)',
                          esc, re.IGNORECASE)
            if m:
                pct = float(m.group(3))
                c = _TOK["ok"] if pct == 0 else (_TOK["warn_text"] if pct < 50 else _TOK["danger_text"])
                head = esc[:m.start()] + m.group(0)[: m.start(1) - m.start()]
                tail = esc[m.end():]
                esc = (head + m.group(1) + "/" + m.group(2) + " ("
                       + f'<span style="color:{c};font-weight:bold">{m.group(3)}</span>%)'
                       + tail)
        return esc

    def _append_log(self, text):
        self.log_view.append(self._colorize_log(text))
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
