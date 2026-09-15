"""Collect every OCIM run's evaluation into one table (HTER / AUC-ROC / EER / Acc).

    python summarize_ocim.py
    python summarize_ocim.py --roots logs_loo_ocim logs_loo_ocim_kd --out logs_loo_ocim/RESULTS.md
"""
import os
import csv
import glob
import json
import argparse


def _rows_from_eval(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return [{"log_dir": os.path.dirname(path), "error": str(e)}]

    kd = bool(d.get("kd", False))
    sources = "".join(d.get("sources", []))
    common = {"kd": "kd" if kd else "", "sources": sources, "log_dir": d.get("log_dir") or os.path.dirname(path)}

    out = []
    per_target = d.get("per_target")
    if per_target:
        for tgt, pt in per_target.items():
            m = pt.get("metrics", {})
            out.append({**common, "target": tgt, "n": pt.get("n_test"), "hter": m.get("hter"),
                        "auc_roc": m.get("auc_roc"), "eer": m.get("eer"), "acc": m.get("test_acc")})
    else:
        m = d.get("metrics", {})
        out.append({**common, "target": d.get("target"), "n": d.get("n_test"), "hter": m.get("hter"),
                    "auc_roc": m.get("auc_roc"), "eer": m.get("eer"), "acc": m.get("test_acc")})
    return out


def collect(roots):
    rows = []
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "**", "loo_ocim_eval_test.json"), recursive=True)):
            rows.extend(_rows_from_eval(p))
    return sorted(rows, key=lambda x: (x.get("kd", ""), str(x.get("sources", "")), str(x.get("target", ""))))


def fmt_table(rows):
    hdr = ["kd", "sources", "target", "n", "HTER%", "AUC", "EER", "Acc%"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for x in rows:
        if x.get("error"):
            lines.append(f"| _(error)_ | | | | | | | {x['error'][:40]} |")
            continue
        h, a, e, ac = x.get("hter"), x.get("auc_roc"), x.get("eer"), x.get("acc")
        lines.append("| " + " | ".join(str(v) for v in [
            x.get("kd", ""), x.get("sources") or "-", x.get("target", ""), x.get("n", ""),
            f"{h*100:.2f}" if isinstance(h, (int, float)) else "-",
            f"{a:.4f}" if isinstance(a, (int, float)) else "-",
            f"{e:.4f}" if isinstance(e, (int, float)) else "-",
            f"{ac:.2f}" if isinstance(ac, (int, float)) else "-",
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["logs_loo_ocim", "logs_loo_ocim_kd"])
    ap.add_argument("--out", default="logs_loo_ocim/RESULTS.md")
    ap.add_argument("--csv", default="logs_loo_ocim/RESULTS.csv")
    args = ap.parse_args()

    rows = collect(args.roots)
    table = fmt_table(rows)
    print(table)
    print(f"\n{len(rows)} result rows")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from datetime import datetime
    with open(args.out, "w") as f:
        f.write(f"# OCIM results\n\n_generated {datetime.now().isoformat(timespec='seconds')} "
                f"by summarize_ocim.py — roots: {', '.join(args.roots)}_\n\n")
        f.write(table + "\n")
    print(f"wrote {args.out}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kd", "sources", "target", "n", "hter", "auc_roc", "eer", "acc", "log_dir"])
        for x in rows:
            if x.get("error"):
                continue
            w.writerow([x.get("kd"), x.get("sources") or "-", x.get("target"), x.get("n"),
                        x.get("hter"), x.get("auc_roc"), x.get("eer"), x.get("acc"), x.get("log_dir")])
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
