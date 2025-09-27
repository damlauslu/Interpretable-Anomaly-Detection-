import argparse
import os
import sys
from typing import Dict, List


def _ensure_repo_on_path():
    # Add repo root to sys.path so we can import ModularizedClasses.* reliably
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:
        print(f"Matplotlib not available, skipping plots. Reason: {e}")
        return None


def load_feature_phrase_map() -> Dict[str, str]:
    _ensure_repo_on_path()
    try:
        from ModularizedClasses.ForDetecting.TestingModel import _feature_phrase_map  # type: ignore
    except Exception as e:
        print(f"Could not import _feature_phrase_map: {e}")
        return {}
    try:
        mp = _feature_phrase_map()
        if not isinstance(mp, dict):
            raise TypeError("_feature_phrase_map() did not return a dict")
        return mp
    except Exception as e:
        print(f"Failed to call _feature_phrase_map: {e}")
        return {}


def plot_bipartite(feature_map: Dict[str, str], out_dir: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return

    os.makedirs(out_dir, exist_ok=True)

    features: List[str] = list(feature_map.keys())  # preserve insertion order
    phrases: List[str] = [feature_map[k] for k in features]

    n = len(features)
    if n == 0:
        print("No features to plot.")
        return

    # Positions for manual bipartite drawing
    ys = [1.0 - (i / max(1, n - 1)) for i in range(n)] if n > 1 else [0.5]

    plt.figure(figsize=(10, max(4, 0.5 * n)))
    # Draw lines connecting feature -> phrase
    for y in ys:
        plt.plot([0.15, 0.85], [y, y], color="#999999", linewidth=1.2, alpha=0.7)

    # Feature labels (left)
    for y, f in zip(ys, features):
        plt.text(0.14, y, f, ha="right", va="center", fontsize=10, color="#2F4F4F")

    # Phrase labels (right)
    for y, p in zip(ys, phrases):
        plt.text(0.86, y, p, ha="left", va="center", fontsize=10, color="#111111")

    plt.title("Feature ↔ Phrase Map", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_phrase_map_bipartite.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_table(feature_map: Dict[str, str], out_dir: str):
    plt = _try_import_matplotlib()
    if plt is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    items = list(feature_map.items())
    if not items:
        print("No features to plot.")
        return

    # Build table data
    cols = ["feature", "phrase"]
    cell_text = [[k, v] for k, v in items]

    # Dynamic figure height
    rows = len(items)
    fig_h = max(3, 0.4 * rows)
    plt.figure(figsize=(12, fig_h))

    # Turn off axes
    plt.axis("off")

    # Create table
    table = plt.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.title("Feature → Phrase Table", fontsize=12, pad=12)

    out_path = os.path.join(out_dir, "feature_phrase_map_table.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the feature phrase map from TestingModel._feature_phrase_map().")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "viz"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--style",
        choices=["both", "bipartite", "table"],
        default="both",
        help="Which visualization(s) to generate",
    )
    args = parser.parse_args()

    fmap = load_feature_phrase_map()
    if not fmap:
        print("Feature phrase map is empty; nothing to visualize.")
        return 1

    if args.style in ("both", "bipartite"):
        plot_bipartite(fmap, args.out)
    if args.style in ("both", "table"):
        plot_table(fmap, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

