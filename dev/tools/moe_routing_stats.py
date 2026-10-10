#!/usr/bin/env python3
"""Per-layer expert frequency and router-weight coverage from SPLASH_MOE_STATS dumps.

usage: moe_routing_stats.py english=FILE code=FILE chat=FILE ...
Record layout (Runtime.mm recordMoeStats): header "MOES", version, layers, routes; then per row
kind u32 ('P'/'D'), token u32, request u64, position u64, layers x routes x (expert u32, weight f32).
"""
import sys
import numpy as np

EXPERTS = 512
PARAMS_PER_EXPERT = 3 * 2560 * 640  # gate, up, down


def load(path):
    raw = open(path, "rb").read()
    magic, _, layers, routes = np.frombuffer(raw[:16], "<u4")
    assert magic == 0x534F454D
    rec = np.dtype([("kind", "<u4"), ("tok", "<u4"), ("req", "<u8"), ("pos", "<u8"),
                    ("r", [("id", "<u4"), ("w", "<f4")], (layers, routes))])
    a = np.frombuffer(raw[16:], rec)
    # warmup requests carry ids above 2**63
    return a[a["req"] < 2**63]


def mass(a):
    """[layers][experts] count and router-weight mass."""
    ids, w = a["r"]["id"], a["r"]["w"]
    layers = ids.shape[1]
    cnt = np.zeros((layers, EXPERTS)); ms = np.zeros((layers, EXPERTS))
    for l in range(layers):
        cnt[l] = np.bincount(ids[:, l].ravel(), minlength=EXPERTS)
        ms[l] = np.bincount(ids[:, l].ravel(), weights=w[:, l].ravel(), minlength=EXPERTS)
    return cnt, ms


def hot_count(ms, frac):
    s = np.sort(ms, axis=1)[:, ::-1]
    c = np.cumsum(s, axis=1) / s.sum(axis=1, keepdims=True)
    return (c < frac).sum(axis=1) + 1


def hot_set(ms, frac):
    order = np.argsort(-ms, axis=1)
    n = hot_count(ms, frac)
    return [set(order[l, :n[l]]) for l in range(ms.shape[0])]


def jaccard(x, y):
    return np.mean([len(a & b) / len(a | b) for a, b in zip(x, y)])


def main():
    sets = dict(s.split("=", 1) for s in sys.argv[1:])
    data = {k: load(v) for k, v in sets.items()}
    allrec = np.concatenate(list(data.values()))
    print("## Data")
    print("| domain | rows | prefill | decode |\n|---|---|---|---|")
    for k, a in list(data.items()) + [("all", allrec)]:
        print(f"| {k} | {len(a)} | {(a['kind'] == 80).sum()} | {(a['kind'] == 68).sum()} |")
    layers = allrec["r"]["id"].shape[1]
    stats = {k: mass(a) for k, a in data.items()}
    cnt, ms = mass(allrec)
    ms_n = ms / ms.sum(axis=1, keepdims=True)

    print("\n## Coverage: experts per layer (of 512) covering X% of router-weight mass (all data)")
    print("| X | mean | min | max | layers 0-3 | layers 22-25 | layers 44-47 |\n|---|---|---|---|---|---|---|")
    for f in (0.5, 0.8, 0.9, 0.95, 0.99):
        n = hot_count(ms, f)
        print(f"| {int(f*100)}% | {n.mean():.0f} | {n.min()} | {n.max()} | {list(n[0:4])} | {list(n[22:26])} | {list(n[44:48])} |")
    print("\nExperts never selected per layer: mean %.1f, min %d, max %d" % (
        (cnt == 0).sum(1).mean(), (cnt == 0).sum(1).min(), (cnt == 0).sum(1).max()))

    print("\n## Per layer experts for 50/80/90/95% mass (all data)")
    print("| layer | 50 | 80 | 90 | 95 | unused |\n|---|---|---|---|---|---|")
    h = {f: hot_count(ms, f) for f in (0.5, 0.8, 0.9, 0.95)}
    for l in range(layers):
        print(f"| {l} | {h[0.5][l]} | {h[0.8][l]} | {h[0.9][l]} | {h[0.95][l]} | {(cnt[l] == 0).sum()} |")

    print("\n## Overlap of the 80%-mass hot sets between domains (mean Jaccard over layers; expected under independence in brackets)")
    names = list(data)
    sets80 = {k: hot_set(stats[k][1], 0.8) for k in names}
    print("| a | b | Jaccard | hot a | hot b | frac of a also hot in b |\n|---|---|---|---|---|---|")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ja = jaccard(sets80[a], sets80[b])
            na = np.mean([len(x) for x in sets80[a]]); nb = np.mean([len(x) for x in sets80[b]])
            inter = np.mean([len(x & y) / len(x) for x, y in zip(sets80[a], sets80[b])])
            print(f"| {a} | {b} | {ja:.3f} | {na:.0f} | {nb:.0f} | {inter:.3f} |")
    union = [set().union(*[sets80[k][l] for k in names]) for l in range(layers)]
    print(f"\nUnion of the domains' 80% hot sets: mean {np.mean([len(u) for u in union]):.0f} experts per layer")
    # mass of domain b covered by hot set of a
    print("\nMass of domain B's tokens routed to experts hot (80%) for A: ")
    print("| A -> B | mass covered | per-layer min |\n|---|---|---|")
    for a in names:
        for b in names:
            if a == b: continue
            cov = []
            for l in range(layers):
                m = stats[b][1][l]; cov.append(m[list(sets80[a][l])].sum() / m.sum())
            print(f"| {a} -> {b} | {np.mean(cov):.3f} | {np.min(cov):.3f} |")

    print("\n## Stability: halves of the data (hot-set Jaccard and cross coverage)")
    print("| split | Jaccard 80% hot sets | mass of half B covered by hot(A) 80% | experts 80% A / B |\n|---|---|---|---|")
    def halves(name, A, B):
        mA, mB = mass(A)[1], mass(B)[1]
        sA, sB = hot_set(mA, 0.8), hot_set(mB, 0.8)
        cov = np.mean([mB[l][list(sA[l])].sum() / mB[l].sum() for l in range(layers)])
        print(f"| {name} | {jaccard(sA, sB):.3f} | {cov:.3f} | {np.mean([len(x) for x in sA]):.0f} / {np.mean([len(x) for x in sB]):.0f} |")
    for k, a in data.items():
        n = len(a) // 2
        halves(f"{k}: first/second half", a[:n], a[n:])
        halves(f"{k}: even/odd rows", a[::2], a[1::2])
        for kind, label in ((80, "prefill"), (68, "decode")):
            pass
    pre = allrec[allrec["kind"] == 80]; dec = allrec[allrec["kind"] == 68]
    halves("all: prefill vs decode", pre, dec)

    print("\n## Expert-size estimate (params per expert %d, %d layers x %d experts = %.1fB params)" % (
        PARAMS_PER_EXPERT, layers, EXPERTS, layers * EXPERTS * PARAMS_PER_EXPERT / 1e9))
    cur = 22649241600 + 2 * 5570560000 + 2 * 3027763200
    print(f"Current file (IQ4_NL down, IQ1_S/IQ2_XXS gate/up): {cur/2**30:.2f} GiB ({8*cur/(layers*EXPERTS*PARAMS_PER_EXPERT):.2f} bpw)")
    print("\n| hot mass | hot bpw | rest bpw | hot experts (mean/layer) | GiB | avg bpw |\n|---|---|---|---|---|---|")
    for f in (0.5, 0.8, 0.9, 0.95):
        n = hot_count(ms, f)
        for hb, rb in ((4.5, 2.0), (4.5, 2.5), (5.5, 2.5), (4.5, 3.0)):
            params_hot = n.sum() * PARAMS_PER_EXPERT
            params_all = layers * EXPERTS * PARAMS_PER_EXPERT
            bits = params_hot * hb + (params_all - params_hot) * rb
            print(f"| {int(f*100)}% | {hb} | {rb} | {n.mean():.0f} | {bits/8/2**30:.1f} | {bits/params_all:.2f} |")
    print("\nTwo-tier by cumulative mass share, all experts at the same bpw for reference: 2.0 -> %.1f GiB, 2.5 -> %.1f GiB, 3.0 -> %.1f GiB, 4.5 -> %.1f GiB" % tuple(
        layers * EXPERTS * PARAMS_PER_EXPERT * b / 8 / 2**30 for b in (2.0, 2.5, 3.0, 4.5)))
    # error-weighted view: mass-weighted average bpw seen by tokens
    print("\nMass-weighted bpw (the bits a routed token sees, hot 80% at 4.5, rest at 2.0 / 2.5):")
    n = hot_count(ms, 0.8)
    print("hot share of mass = %.3f; token-seen bpw = %.2f / %.2f" % (
        0.8, 0.8 * 4.5 + 0.2 * 2.0, 0.8 * 4.5 + 0.2 * 2.5))


if __name__ == "__main__":
    main()
