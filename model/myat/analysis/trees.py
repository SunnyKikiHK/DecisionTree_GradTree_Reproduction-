"""Draw a GradTree and a CART tree trained on the same trial, side by side (like paper Figure 1).

Both models keep their tuned settings except the maximum depth, which defaults to 3 so the trees
stay readable. Thresholds are converted back to the original units with the trial's saved
quantile transformer, so a split reads "wall-motion-score < 14" rather than "< 0.37".

Run from the repository root:
    python -m model.myat.analysis.trees                                    # Echocardiogram, seed 42, depth 3
    python -m model.myat.analysis.trees --dataset iris --depth tuned       # each model's tuned depth

Writes figures/trees_<dataset>_seed<seed>_depth<depth>.png and the same trees as text rules in
results/trees_<dataset>_seed<seed>_depth<depth>.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from model.myat.baselines.config import CART_TUNED
from model.myat.gradtree.classifier import GradTreeClassifier
from model.myat.gradtree.config import DATASETS, MODEL_SEED, gradtree_settings
from model.myat.gradtree.data import PREPROCESS_DIR, load_split

from .common import AXIS, COLORS, INK, INK_SECONDARY, LABELS, MUTED, RESULTS_DIR, SURFACE, save_figure, use_style


def gradtree_nodes(node) -> dict:
    """HardTree node -> plain dict. GradTree sends a sample left when x < threshold."""
    if node.is_leaf:
        value = node.value
        return {"class": int(value > 0) if np.ndim(value) == 0 else int(np.argmax(value)), "samples": node.n_samples}
    return {"feature": node.feature, "threshold": node.threshold, "op": "<", "samples": node.n_samples,
            "left": gradtree_nodes(node.left), "right": gradtree_nodes(node.right)}


def cart_nodes(tree, index: int = 0) -> dict:
    """sklearn tree -> plain dict. sklearn sends a sample left when x <= threshold."""
    t = tree.tree_
    samples = int(t.n_node_samples[index])
    if t.children_left[index] == t.children_right[index]:
        return {"class": int(tree.classes_[np.argmax(t.value[index][0])]), "samples": samples}
    return {"feature": int(t.feature[index]), "threshold": float(t.threshold[index]), "op": "≤", "samples": samples,
            "left": cart_nodes(tree, t.children_left[index]), "right": cart_nodes(tree, t.children_right[index])}


def to_original_units(tree: dict, quantile_transformer, n_features: int) -> None:
    """Replace every threshold by the original value that the quantile transform maps to it."""
    if "feature" not in tree:
        return
    row = np.zeros((1, n_features))
    row[0, tree["feature"]] = tree["threshold"]
    tree["raw_threshold"] = float(quantile_transformer.inverse_transform(row)[0, tree["feature"]])
    to_original_units(tree["left"], quantile_transformer, n_features)
    to_original_units(tree["right"], quantile_transformer, n_features)


def number(value: float) -> str:
    """3 significant digits without scientific notation (0.0000133, not 1.33e-05)."""
    return np.format_float_positional(value, precision=3, unique=False, fractional=False, trim="-")


def count(tree: dict) -> int:
    return 1 if "feature" not in tree else 1 + count(tree["left"]) + count(tree["right"])


def rules(tree: dict, features: list[str], classes: list[str], indent: str = "") -> list[str]:
    if "feature" not in tree:
        return [f"{indent}{classes[tree['class']]}  ({tree['samples']} training samples)"]
    condition = f"{features[tree['feature']]} {tree['op']} {number(tree['raw_threshold'])}"
    return ([f"{indent}if {condition}:"] + rules(tree["left"], features, classes, indent + "    ")
            + [f"{indent}else:"] + rules(tree["right"], features, classes, indent + "    "))


def layout(tree: dict, depth: int = 0, positions: list | None = None) -> list:
    """(node, x, depth) for every node: leaves at 0, 1, 2, ... left to right, parents centred above."""
    positions = [] if positions is None else positions
    if "feature" not in tree:
        x = sum(1 for _, _, _, leaf in positions if leaf)
        positions.append((tree, x, depth, True))
        return positions
    layout(tree["left"], depth + 1, positions)
    layout(tree["right"], depth + 1, positions)
    left_x = next(x for node, x, _, _ in positions if node is tree["left"])
    right_x = next(x for node, x, _, _ in positions if node is tree["right"])
    positions.append((tree, (left_x + right_x) / 2, depth, False))
    return positions


def tint(color: str, amount: float = 0.1) -> tuple:
    """An opaque wash of color on the chart surface (so the edges behind a box stay hidden)."""
    from matplotlib.colors import to_rgb

    return tuple((1 - amount) * s + amount * c for s, c in zip(to_rgb(SURFACE), to_rgb(color)))


def draw(ax, tree: dict, features: list[str], classes: list[str], color: str) -> None:
    positions = layout(tree)
    where = {id(node): (x, -depth) for node, x, depth, _ in positions}
    for node, x, depth, leaf in positions:
        if leaf:
            continue
        for child, answer in [(node["left"], "yes"), (node["right"], "no")]:
            cx, cy = where[id(child)]
            ax.plot([x, cx], [-depth, cy], color=AXIS, linewidth=1, zorder=1)
            ax.text((x + cx) / 2, (-depth + cy) / 2, answer, ha="center", va="center", fontsize=7, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"), zorder=2)
    for node, x, depth, leaf in positions:
        if leaf:
            plural = "" if node["samples"] == 1 else "s"
            text, weight = f"{classes[node['class']]}\n{node['samples']} sample{plural}", "semibold"
            box = dict(boxstyle="round,pad=0.35", facecolor=tint(color), edgecolor=color, linewidth=1)
        else:
            text = f"{features[node['feature']]}\n{node['op']} {number(node['raw_threshold'])}"
            weight = "normal"
            box = dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=AXIS, linewidth=1)
        ax.text(x, -depth, text, ha="center", va="center", fontsize=7.5, color=INK, fontweight=weight, bbox=box, zorder=3)
    xs = [x for _, x, _, _ in positions]
    depths = [d for _, _, d, _ in positions]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(-max(depths) - 0.6, 0.6)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="echocardiogram", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=MODEL_SEED, help="which trial's split to train on")
    parser.add_argument("--depth", default="3", help="maximum depth of both trees, or 'tuned'")
    parser.add_argument("--preprocess-dir", type=Path, default=PREPROCESS_DIR)
    args = parser.parse_args()

    name = args.dataset
    split = load_split(name, args.seed, args.preprocess_dir)
    preprocessor = joblib.load(split["folder"] / f"{name}_preprocessor.joblib")
    features = split["feature_names"]
    encoder = preprocessor.get("label_encoder")
    target = preprocessor.get("target", "class")
    if encoder is not None and isinstance(encoder.classes_[0], str):
        classes = [str(c) for c in encoder.classes_]
    else:
        values = encoder.classes_ if encoder is not None else range(split["n_classes"])
        classes = [f"{target} = {v}" for v in values]
    categorical = set(preprocessor.get("categorical_features") or [])
    features = [f"{f} (encoded)" if f in categorical else f for f in features]

    gradtree_params = gradtree_settings(name, "tuned")
    cart_params = dict(CART_TUNED[name])
    if args.depth != "tuned":
        gradtree_params["depth"] = cart_params["max_depth"] = int(args.depth)

    gradtree = GradTreeClassifier(**gradtree_params, n_classes=split["n_classes"], seed=MODEL_SEED)
    gradtree.fit(split["X_train"], split["y_train"], split["X_valid"], split["y_valid"])
    cart = DecisionTreeClassifier(random_state=MODEL_SEED, **cart_params)
    cart.fit(np.vstack([split["X_train"], split["X_valid"]]), np.concatenate([split["y_train"], split["y_valid"]]))

    trees = {"cart": cart_nodes(cart), "gradtree": gradtree_nodes(gradtree.tree_.root)}
    scores = {}
    for model, estimator in [("cart", cart), ("gradtree", gradtree)]:
        to_original_units(trees[model], preprocessor["quantile_transformer"], len(features))
        pred = estimator.predict(split["X_test"])
        scores[model] = (f1_score(split["y_test"], pred, average="macro"), accuracy_score(split["y_test"], pred))

    import matplotlib.pyplot as plt

    use_style()
    leaves = {m: sum(1 for *_, leaf in layout(t) if leaf) for m, t in trees.items()}
    max_depth = max(max(d for _, _, d, _ in layout(t)) for t in trees.values())
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.15 * sum(leaves.values()) + 1.0), 1.0 * max_depth + 1.5),
        gridspec_kw={"width_ratios": [max(leaves["cart"], 2), max(leaves["gradtree"], 2)]})
    titles = {"cart": "CART (greedy)", "gradtree": "GradTree (gradient-based)"}
    for ax, model in zip(axes, ["cart", "gradtree"]):
        draw(ax, trees[model], features, classes, COLORS[model])
        f1, accuracy = scores[model]
        ax.set_title(f"{titles[model]}\nTest macro F1 {f1:.3f} · accuracy {accuracy:.3f} · {count(trees[model])} nodes",
                     fontsize=9.5, color=INK, pad=10)
    depth_label = "tuned depth" if args.depth == "tuned" else f"maximum depth {args.depth}"
    fig.suptitle(f"{LABELS[name]}, trial seed {args.seed}, {depth_label}", x=0.0, y=1.02, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.text(0.0, -0.02, "Left branch = condition true. Thresholds in original units; samples = training rows "
             "reaching the node (CART: train + valid; GradTree: train).", color=MUTED, fontsize=8, ha="left")
    stem = f"trees_{name}_seed{args.seed}_depth{args.depth}"
    figure = save_figure(fig, stem)

    lines = []
    for model in ["cart", "gradtree"]:
        f1, accuracy = scores[model]
        lines += [f"{titles[model]}: test macro F1 {f1:.3f}, accuracy {accuracy:.3f}, {count(trees[model])} nodes", ""]
        lines += rules(trees[model], features, classes) + [""]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {figure} and {RESULTS_DIR / f'{stem}.txt'}")


if __name__ == "__main__":
    main()
