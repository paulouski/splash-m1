"""Full-vocab KL(ref || ours) and top-1 agreement between two logits dumps.

    kl_compare.py --dir DIR [--ref ref] [--ours ours] [--top 10] [--json OUT] [--tokens CORPUS]
    kl_compare.py --selftest

Per segment <seg>, DIR holds <seg>.<tag>.f32 (rows of raw fp32 logits, or log-probs;
both sides are renormalised) and <seg>.<tag>.json with n_vocab, and for sparse dumps
"positions" (the row p scores token p+1). Rows are matched by position; a dump without
"positions" holds every position 0..n-1. --tokens CORPUS prints token text ids of the
worst rows from CORPUS/<seg>.tokens.txt.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

BLOCK = 64


def load(directory, seg, tag):
    meta = json.loads((directory / f"{seg}.{tag}.json").read_text())
    vocab = meta["n_vocab"]
    raw = np.memmap(directory / f"{seg}.{tag}.f32", dtype=np.float32, mode="r")
    if raw.size % vocab:
        raise SystemExit(f"{seg}.{tag}.f32: size is not a multiple of n_vocab={vocab}")
    rows = raw.reshape(-1, vocab)
    positions = meta.get("positions", list(range(meta.get("n", len(rows)))))
    if len(positions) != len(rows):
        raise SystemExit(
            f"{seg}.{tag}: {len(rows)} rows but {len(positions)} positions"
        )
    return rows, dict(zip(positions, range(len(rows)), strict=True)), vocab


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def compare_rows(ref, ours):
    """Per row KL(ref||ours), ref argmax, ours argmax, ref log-prob cost of ours' pick."""
    lr, lo = log_softmax(ref), log_softmax(ours)
    kl = (np.exp(lr) * (lr - lo)).sum(axis=1)
    rt, ot = lr.argmax(axis=1), lo.argmax(axis=1)
    gap = lr[np.arange(len(rt)), rt] - lr[np.arange(len(rt)), ot]
    return kl, rt, ot, gap


def compare_segment(directory, seg, ref_tag, ours_tag):
    ref, ref_at, v_ref = load(directory, seg, ref_tag)
    ours, ours_at, v_ours = load(directory, seg, ours_tag)
    if v_ref != v_ours:
        raise SystemExit(f"{seg}: n_vocab differs (ref {v_ref}, ours {v_ours})")
    positions = sorted(set(ref_at) & set(ours_at))
    out = {"kl": [], "ref_top": [], "ours_top": [], "gap": []}
    for i in range(0, len(positions), BLOCK):
        chunk = positions[i : i + BLOCK]
        r = np.stack([ref[ref_at[p]] for p in chunk])
        o = np.stack([ours[ours_at[p]] for p in chunk])
        for key, values in zip(out, compare_rows(r, o), strict=True):
            out[key].append(values)
    return positions, {k: np.concatenate(v) for k, v in out.items()}


def summarize(positions, rows):
    kl = rows["kl"]
    return {
        "n": len(kl),
        "top1": float((rows["ref_top"] == rows["ours_top"]).mean()),
        "kl_mean": float(kl.mean()),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
    }


def run(directory, ref_tag, ours_tag, top, corpus=None):
    results, all_rows = {}, []
    for meta in sorted(directory.glob(f"*.{ours_tag}.json")):
        seg = meta.name[: -len(f".{ours_tag}.json")]
        positions, rows = compare_segment(directory, seg, ref_tag, ours_tag)
        results[seg] = summarize(positions, rows)
        worst = np.argsort(-rows["kl"])[:top]
        ids = (
            list(map(int, (corpus / f"{seg}.tokens.txt").read_text().split()))
            if corpus
            else None
        )
        results[seg]["worst"] = [
            {
                "position": positions[i],
                "kl": float(rows["kl"][i]),
                "ref_top": int(rows["ref_top"][i]),
                "ours_top": int(rows["ours_top"][i]),
                "ref_gap": float(rows["gap"][i]),
                "next_token": ids[positions[i] + 1]
                if ids and positions[i] + 1 < len(ids)
                else None,
            }
            for i in worst
        ]
        all_rows.append(rows)
    merged = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    results["ALL"] = summarize(None, merged)
    return results


def table(results):
    print("| segment | n | top1 | KL mean | KL p99 | KL max |")
    print("|---|---|---|---|---|---|")
    for seg, r in results.items():
        print(
            f"| {seg} | {r['n']} | {r['top1'] * 100:.2f}% | {r['kl_mean']:.3e} | "
            f"{r['kl_p99']:.3e} | {r['kl_max']:.3e} |"
        )
    for seg, r in results.items():
        for w in r.get("worst", []):
            print(
                f"worst {seg} pos={w['position']} kl={w['kl']:.3e} ref_top={w['ref_top']} "
                f"ours_top={w['ours_top']} ref_gap={w['ref_gap']:.3f} next_token={w['next_token']}"
            )


def selftest():
    rng = np.random.default_rng(0)
    vocab, n = 500, 24
    z = rng.normal(0, 3, (n, vocab)).astype(np.float32)
    ident, _, _, _ = compare_rows(z, z + 5.0)  # logits offset-invariant
    assert ident.max() < 1e-9
    # analytic: ours = ref with token 0 shifted by d
    shifted = z.copy()
    shifted[:, 0] += 2.0
    kl, rt, ot, gap = compare_rows(z, shifted)
    p = np.exp(log_softmax(z))
    q = np.exp(log_softmax(shifted))
    want = (p * (np.log(p) - np.log(q))).sum(axis=1)
    assert np.allclose(kl, want) and (kl > 0).all()
    # forced flip: ours promotes a non-top token above the top
    flip = z.copy()
    flip[0, 7] = z[0].max() + 1.0
    _, rt, ot, gap = compare_rows(z, flip)
    assert ot[0] == 7 and rt[0] != 7 and gap[0] > 0 and (ot[1:] == rt[1:]).all()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        z.tofile(d / "s.ref.f32")
        (d / "s.ref.json").write_text(json.dumps({"n": n, "n_vocab": vocab}))
        sparse = flip[
            [3, 9, 0]
        ]  # sparse ours: unsorted positions, one flip at position 0
        sparse.tofile(d / "s.ours.f32")
        (d / "s.ours.json").write_text(
            json.dumps({"n_vocab": vocab, "positions": [3, 9, 0]})
        )
        r = run(d, "ref", "ours", 2)
    s = r["s"]
    assert s["n"] == 3 and abs(s["top1"] - 2 / 3) < 1e-9, s
    assert s["worst"][0]["position"] == 0 and s["worst"][0]["ours_top"] == 7
    assert s["kl_mean"] > 0 and r["ALL"]["n"] == 3
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--ref", default="ref")
    ap.add_argument("--ours", default="ours")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tokens", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    results = run(a.dir, a.ref, a.ours, a.top, a.tokens)
    table(results)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
