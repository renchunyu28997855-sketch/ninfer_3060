"""Surgical re-pack of the ternary (t2_g128_fp16) core of the v3 artifact.

Base file   : E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-fixed.ninfer
              (author's file, with the 3 hadamard-sign aux objects fixed to bf16 +-1)
Output      : E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer

Every object whose layout is row-split-k128-v1 (the PQ2_0 ternary objects) is
re-generated from the GGUF via pack.py's own producers (byte-exact plane
assembly + author's inverse checks); all other objects (int side, resources,
JSON dir) are left untouched. Object offsets/sizes are unchanged, so the
directory JSON stays valid.
"""
import io
import os
import json
import struct
import sys
import types

GGUF = r"E:\download\123\model\Ternary-Bonsai-2-27B-PQ2_0(1).gguf"
SRC = r"E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-fixed.ninfer"
DST = r"E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer"

# --- import pack.py without running its template requirement --------------
stub_torch = types.ModuleType("torch")
class _T:  # dummy for TypeAlias usage
    pass
stub_torch.Tensor = _T
for _a in ("bfloat16","float32","int32","int64","float16","uint8","int8","float64"):
    setattr(stub_torch, _a, _T)
sys.modules.setdefault("torch", stub_torch)

sys.path.insert(0, r"E:\download\123\ninfer-ternary-bonsai-ada-master")
sys.path.insert(0, r"E:\download\123\model\tools")
import pack  # noqa: E402

# --- shipped dir -----------------------------------------------------------
data = open(SRC, "rb").read(405504)
json_bytes = struct.unpack("<Q", data[8:16])[0]
j = json.loads(data[32:32 + json_bytes])
objects = {o["id"]: o for o in j["objects"]}
by_name_target = {}   # semantic name -> object id (from bindings)
for k, v in j["bindings"].items():
    if isinstance(v, str):
        by_name_target[k] = v

# --- Packer without template ----------------------------------------------
g = pack.Gguf(GGUF)
p = pack.Packer.__new__(pack.Packer)
p.g = g
p.objects = []            # unused by the producers we call
p.identity = None
p._by_name = {}
p.kind = {}
p._signs = None
pack.Packer.fmt_of = pack.Packer.fmt_of  # no-op, keep methods

# which semantic names produce t2 payloads here
t2_objs = [oid for oid, o in objects.items() if o.get("format") == "t2_g128_fp16"]
print(f"t2 objects in shipped dir: {len(t2_objs)}")

# map each physical t2 object -> packer producer name -----------------------
def fqn(layer, sub):
    return f"text/layers/{layer}/{sub}"

producer_for_obj = {}
# aggregate part refs per physical object ACROSS all bindings, then resolve groups
agg = {}
for k, v in j["bindings"].items():
    if not isinstance(v, dict):
        continue
    if "parts" in v:
        for pp in v["parts"]:
            agg.setdefault(pp["object"], set()).add(k)
    elif v.get("object"):
        agg.setdefault(v["object"], set()).add(k)
for oid, keyset in agg.items():
    if oid not in objects or objects[oid].get("format") != "t2_g128_fp16":
        continue
    keys = sorted(keyset)
    whole = [k for k in keys
             if isinstance(j["bindings"][k], dict) and "parts" not in j["bindings"][k]]
    if whole:
        producer_for_obj.setdefault(oid, whole[0])
        continue
    p0 = keys[0].split("/")
    assert keys[0].startswith("text/layers"), keys
    L = int(p0[2])
    ops = sorted({x.split("/")[-1] for x in keys})
    if ops == ["key", "query"]:
        sub = "gdn/query_key" if objects[oid]["shape"] == [4096, 5120] else "attention/query_key"
    elif ops == ["value", "z"]:
        sub = "gdn/value_z"
    elif ops == ["gate", "up"]:
        sub = "mlp/gate_up"
    elif ops == ["gate", "value"]:
        sub = "attention/gate_value"
    else:
        raise SystemExit(f"unresolved part group {ops} for {oid}")
    producer_for_obj.setdefault(oid, fqn(L, sub))

# sanity: every t2 object resolved, none left as proposal/head
skip = set()
for oid in t2_objs:
    if oid not in producer_for_obj:
        # no binding reaches it -> leave untouched (e.g. MTP proposal/head)
        skip.add(oid)
print(f"splicable: {len(t2_objs) - len(skip)}, skipped (untouched): {sorted(skip)}")

# Build payloads for every splicable object -------------------------------
PAYLOAD_START = 405504          # absolute file offset of payload region
splices = {}                    # ABSOLUTE file offset -> bytes
done = set()
report = []
for oid, name in producer_for_obj.items():
    if oid in skip: continue
    o = objects[oid]
    try:
        payload = p.produce(name)
    except Exception as e:
        report.append((oid, name, f"PRODUCE FAILED: {e!r}"))
        continue
    if payload is None:
        report.append((oid, name, "SKIP borrow/unknown producer (left untouched)"))
        continue
    if len(payload) != o["bytes"]:
        report.append((oid, name, f"SIZE MISMATCH {len(payload)} != {o['bytes']}"))
        continue
    splices[PAYLOAD_START + o["offset"]] = payload
    done.add(oid)
    report.append((oid, name, f"ok {len(payload)} B"))

# Deterministic in-place rebuild: exact base copy + seek/write, no streaming.
if os.path.exists(DST):
    os.remove(DST)
import shutil
shutil.copyfile(SRC, DST)
f = open(DST, "r+b")
nwritten = 0
for off in sorted(splices):
    f.seek(off)
    assert f.tell() == off
    f.write(splices[off])
    nwritten += 1
f.close()
out_fsize = os.path.getsize(DST)
assert out_fsize == os.path.getsize(SRC), f"size mismatch {out_fsize}"
print(f"rebuilt {DST}: {out_fsize} bytes (in-place spliced {nwritten} objects)")

fails = [r for r in report if "FAILED" in r[2] or "MISMATCH" in r[2]]
skipped = [r for r in report if r[2].startswith("SKIP")]
print(f"report: {len(report)-len(fails)-len(skipped)} ok, {len(fails)} failed, {len(skipped)} skipped")
for r in fails[:20]:
    print("  FAIL", r)
print("OK" if not fails and len(done) == len(t2_objs) - len(skip) else "INCOMPLETE")
