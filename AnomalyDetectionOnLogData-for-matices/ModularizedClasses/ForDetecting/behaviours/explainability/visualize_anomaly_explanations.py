import argparse
import os
import re
from typing import List

import pandas as pd


def _safe_name(s: str) -> str:
    if s is None:
        return "na"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def load_explanations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize optional columns
    for col in ["UserID", "Date"]:
        if col not in df.columns:
            df[col] = None
    return df


def print_terminal_summary(df: pd.DataFrame, topn: int = 10) -> None:
    n = len(df)
    print(f"Total anomalies in report: {n}")
    if n == 0:
        return

    # Top-N by total_error
    top = df.sort_values("total_error", ascending=False).head(topn)
    print("\nTop anomalies by total_error:")
    for _, r in top.iterrows():
        u = r.get("UserID", "-")
        d = r.get("Date", "-")
        print(
            f" idx={int(r['idx']) if 'idx' in r else '-'} | UserID={u} | Date={d} | total_error={r['total_error']:.4f} | "
            f"top1={r.get('top1_feature', '-')}/{r.get('top1_pct', 0):.2%}"
        )

    # Feature frequencies in top1
    if "top1_feature" in df.columns:
        freq = df["top1_feature"].value_counts()
        print("\nTop1 feature frequency:")
        for k, v in freq.items():
            print(f" {k}: {v}")

    # Aggregate contribution percent across top1-3
    contrib = {}
    for rank in (1, 2, 3):
        fcol = f"top{rank}_feature"
        pcol = f"top{rank}_pct"
        if fcol in df.columns and pcol in df.columns:
            for _, r in df[[fcol, pcol]].dropna().iterrows():
                contrib[r[fcol]] = contrib.get(r[fcol], 0.0) + float(r[pcol])
    if contrib:
        s = pd.Series(contrib).sort_values(ascending=False)
        print("\nAggregate contribution percent (sum of top1-3 pct):")
        for k, v in s.items():
            print(f" {k}: {v:.2%}")


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:
        print(f"Matplotlib not available, skipping plots. Reason: {e}")
        return None


def plot_global_contrib(df: pd.DataFrame, out_dir: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    contrib = {}
    for rank in (1, 2, 3):
        fcol = f"top{rank}_feature"
        pcol = f"top{rank}_pct"
        if fcol in df.columns and pcol in df.columns:
            for _, r in df[[fcol, pcol]].dropna().iterrows():
                contrib[r[fcol]] = contrib.get(r[fcol], 0.0) + float(r[pcol])
    if not contrib:
        print("No contribution columns found for plotting.")
        return
    s = pd.Series(contrib).sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    s.plot(kind="bar")
    plt.title("Anomaly explanations — global contribution (sum pct of top1-3)")
    plt.ylabel("Aggregate percent (fraction)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "anomaly_explanations_global_pct.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_topn_anomalies(df: pd.DataFrame, out_dir: str, topn: int = 6):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)

    cols = [
        ("top1_feature", "top1_pct"),
        ("top2_feature", "top2_pct"),
        ("top3_feature", "top3_pct"),
    ]
    need_cols: List[str] = [c for pair in cols for c in pair]
    for c in need_cols:
        if c not in df.columns:
            print("Missing columns for per-anomaly plots; skipping.")
            return

    top = df.sort_values("total_error", ascending=False).head(topn)
    for _, r in top.iterrows():
        features = [r[a] for a, _ in cols]
        pcts = [float(r[b]) for _, b in cols]
        title = (
            f"idx={int(r['idx']) if 'idx' in r else '-'} | UserID={r.get('UserID','-')} | "
            f"Date={r.get('Date','-')} | total_error={r['total_error']:.4f}"
        )
        plt.figure(figsize=(6, 3.5))
        plt.bar(features, pcts)
        plt.ylim(0, max(0.5, max(pcts) * 1.2))
        plt.ylabel("Percent of error (fraction)")
        plt.title(title)
        plt.tight_layout()
        fname = (
            f"anomaly_{_safe_name(str(r.get('idx','na')))}_"
            f"{_safe_name(str(r.get('UserID','na')))}_"
            f"{_safe_name(str(r.get('Date','na')))}.png"
        )
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and summarize anomaly explanations CSV."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__),
            "anomaly_explanations.csv",
        ),
        help="Path to anomaly_explanations.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "viz"),
        help="Directory to save plots",
    )
    parser.add_argument("--topn", type=int, default=10, help="Top-N rows for summaries/plots")
    parser.add_argument("--no-plots", action="store_true", help="Only print terminal summary")
    args = parser.parse_args()

    df = load_explanations(args.csv)
    print_terminal_summary(df, topn=args.topn)

    if not args.no_plots:
        plot_global_contrib(df, args.out)
        plot_topn_anomalies(df, args.out, topn=min(args.topn, 12))


if __name__ == "__main__":
    main()

