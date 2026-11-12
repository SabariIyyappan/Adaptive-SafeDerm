"""V4 result package — Grad-CAM explainability + IoU (PLAN.md §8, paper §3.5,
Table 10, Figures 8-11).

    python -m src.eval.run_v4

Joins C7 (`results/v4/inputs/gradcam_test.{csv,npz}`) with the V3 referral
decisions (C8) and entropies, and writes `results/v4/`: Table 10 (IoU overall,
accepted vs referred, per class, correct vs incorrect) and Figures 8-11.
Figures need the C9 images and C10 masks under `data/` (tables don't).

Example selection is rule-based, never hand-picked: Fig. 8/10 take the
highest-confidence correct prediction(s) per class; Fig. 9 the
highest-confidence correct and the highest-confidence wrong prediction per
class; Fig. 11 the images at evenly spaced entropy quantiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import CLASS_ORDER
from src.eval import tables as T
from src.eval.classes import CHART_INK, CLASS_FULL_NAME, MALIGNANCY_COLOR, SERIES_2, SINGLE_SERIES_BLUE, malignancy_of
from src.eval.common import write_json
from src.eval.style import finish_figure, grid_y, new_figure
from src.model.gradcam import upsample_normalize

DEFAULT_R = 0.30
PAPER_IOU = {"accepted": 0.61, "referred": 0.41}


def load_joined(inputs: Path, decisions: Path, r: float) -> tuple[pd.DataFrame, np.ndarray]:
    cam = pd.read_csv(inputs / "gradcam_test.csv")
    with np.load(inputs / "gradcam_test.npz", allow_pickle=False) as z:
        raw = z["cams"]
    c8 = pd.read_csv(decisions)
    c8 = c8.loc[np.isclose(c8["referral_rate"], r)].drop(columns=["y_true"])
    df = cam.merge(c8, on="image_id", how="left", validate="one_to_one")
    probs = pd.read_csv(decisions.parent / "predictions_test_T50.csv").set_index("image_id")
    p = probs.loc[df["image_id"], [f"prob_{c}" for c in range(7)]].to_numpy()
    df["conf_T50"] = p.max(1)
    df["pred_T50"] = p.argmax(1)
    df["correct"] = df["pred_class"] == df["y_true"]
    df["status"] = np.where(df["accepted"], "accepted", "referred")
    return df, raw


def mean_sd(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std():.3f}"


def table10(df: pd.DataFrame, c8_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [{"Group": "All test images", "n": len(df), "IoU mean ± SD": mean_sd(df["iou"]), "Paper": "—"}]
    for status in ("accepted", "referred"):
        g = df.loc[df["status"] == status]
        rows.append({"Group": f"{status.capitalize()} (r={DEFAULT_R:.0%})", "n": len(g), "IoU mean ± SD": mean_sd(g["iou"]),
                     "Paper": f"{PAPER_IOU[status]:.2f}"})
    for ok, name in ((True, "Correct predictions"), (False, "Incorrect predictions")):
        g = df.loc[df["correct"] == ok]
        rows.append({"Group": name, "n": len(g), "IoU mean ± SD": mean_sd(g["iou"]), "Paper": "—"})
    overall = pd.DataFrame(rows)

    per_class = []
    for idx, cls in enumerate(CLASS_ORDER):
        g = df.loc[df["y_true"] == idx]
        per_class.append({"Class": CLASS_FULL_NAME[cls], "n": len(g), "IoU mean ± SD": mean_sd(g["iou"]),
                          "Accepted IoU": mean_sd(g.loc[g["accepted"], "iou"]) if g["accepted"].any() else "—",
                          "Referred IoU": mean_sd(g.loc[~g["accepted"], "iou"]) if (~g["accepted"]).any() else "—",
                          "Mean CAM area": f"{g['cam_area'].mean():.3f}", "Mean mask area": f"{g['mask_area'].mean():.3f}"})

    by_r = []
    for r, g in c8_all.groupby("referral_rate"):
        j = df[["image_id", "iou"]].merge(g[["image_id", "accepted"]], on="image_id")
        by_r.append({"referral_rate": r, "n_accepted": int(j["accepted"].sum()),
                     "accepted_iou": j.loc[j["accepted"], "iou"].mean(),
                     "referred_iou": j.loc[~j["accepted"], "iou"].mean() if (~j["accepted"]).any() else np.nan})
    return overall, pd.DataFrame(per_class), pd.DataFrame(by_r)


# ---------------------------------------------------------------- figures
def _overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * image / 255.0 + alpha * heat, 0, 1)


class _Assets:
    def __init__(self, df, raw, images_dir: Path, masks_dir: Path):
        self.df = df.set_index("image_id")
        self.raw = raw
        self.images_dir, self.masks_dir = images_dir, masks_dir

    def image(self, image_id):
        from PIL import Image
        return np.asarray(Image.open(self.images_dir / f"{image_id}.png").convert("RGB"))

    def cam(self, image_id):
        return upsample_normalize(self.raw[[int(self.df.loc[image_id, "cam_index"])]])[0]

    def mask(self, image_id):
        from PIL import Image
        return np.asarray(Image.open(self.masks_dir / f"{image_id}.png")) > 127


def _panel(ax_img, ax_cam, a: _Assets, image_id: str, show_mask: bool = True):
    row = a.df.loc[image_id]
    img, cam = a.image(image_id), a.cam(image_id)
    ax_img.imshow(img)
    ax_cam.imshow(_overlay(img, cam))
    if show_mask:
        ax_cam.contour(a.mask(image_id), levels=[0.5], colors="white", linewidths=0.8)
    cls = CLASS_ORDER[int(row["y_true"])]
    pred = CLASS_ORDER[int(row["pred_class"])]
    color = MALIGNANCY_COLOR[malignancy_of(cls)]
    ax_img.set_title(f"true {cls} · pred {pred}", fontsize=7.5)
    ax_cam.set_title(f"H={row['entropy_total']:.3f} · {'ACCEPT' if row['accepted'] else 'REFER'} · IoU {row['iou']:.2f}",
                     fontsize=7.5)
    for ax in (ax_img, ax_cam):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(2.5)


def fig8_one_per_class(df, a, out_path):
    correct = df.loc[df["correct"]].sort_values(["conf_T50", "image_id"], ascending=[False, True])
    picks = [correct.loc[correct["y_true"] == i, "image_id"].iloc[0] for i in range(7)
             if (correct["y_true"] == i).any()]
    fig, axes = new_figure(2, len(picks), figsize=(2.3 * len(picks), 5.2))
    for k, image_id in enumerate(picks):
        _panel(axes[0, k], axes[1, k], a, image_id)
    return finish_figure(fig, out_path, "Paper Fig. 8: highest-confidence correct T=50 prediction per class. Border: red "
                         "malignant (mel, bcc), yellow pre-cancerous (akiec), green benign. White contour = ground-truth "
                         f"lesion mask. ACCEPT/REFER at r={DEFAULT_R:.0%}.")


def fig9_correct_vs_incorrect(df, a, out_path):
    classes = ["mel", "bcc", "akiec", "nv"]
    fig, axes = new_figure(len(classes), 4, figsize=(9.5, 2.5 * len(classes)))
    for r, cls in enumerate(classes):
        idx = CLASS_ORDER.index(cls)
        g = df.loc[df["y_true"] == idx].sort_values(["conf_T50", "image_id"], ascending=[False, True])
        for c, ok in enumerate((True, False)):
            sub = g.loc[g["correct"] == ok]
            if len(sub):
                _panel(axes[r, 2 * c], axes[r, 2 * c + 1], a, sub["image_id"].iloc[0])
            else:
                for ax in axes[r, 2 * c:2 * c + 2]:
                    ax.axis("off")
    return finish_figure(fig, out_path, "Paper Fig. 9: per class, left pair = highest-confidence CORRECT prediction, "
                         "right pair = highest-confidence WRONG prediction (true class shown). White contour = GT mask.")


def fig10_grid(df, a, out_path, per_class=2):
    correct = df.loc[df["correct"]].sort_values(["conf_T50", "image_id"], ascending=[False, True])
    fig, axes = new_figure(7, 2 * per_class, figsize=(2.3 * 2 * per_class, 2.45 * 7))
    for r in range(7):
        ids = correct.loc[correct["y_true"] == r, "image_id"].head(per_class).tolist()
        for k in range(per_class):
            if k < len(ids):
                _panel(axes[r, 2 * k], axes[r, 2 * k + 1], a, ids[k])
            else:
                axes[r, 2 * k].axis("off")
                axes[r, 2 * k + 1].axis("off")
    return finish_figure(fig, out_path, f"Paper Fig. 10: per-class grid — top-{per_class} most confident correct "
                         "predictions per class (original | Grad-CAM overlay), with total entropy H in nats.")


def fig11_by_entropy(df, a, out_path, n=8):
    order = df.sort_values(["entropy_total", "image_id"]).reset_index(drop=True)
    picks = [order["image_id"].iloc[int(round(q * (len(order) - 1)))] for q in np.linspace(0, 1, n)]
    fig, axes = new_figure(2, n, figsize=(2.15 * n, 5))
    for k, image_id in enumerate(picks):
        _panel(axes[0, k], axes[1, k], a, image_id)
    return finish_figure(fig, out_path, f"Paper Fig. 11: {n} test images at evenly spaced entropy quantiles (0 -> 1), "
                         "lowest entropy on the left. Attention tends to spread off the lesion as entropy rises.")


def plot_iou_distribution(df, out_path):
    fig, ax = new_figure(figsize=(8.5, 4.4))
    bins = np.linspace(0, 1, 26)
    for status, color in (("accepted", SINGLE_SERIES_BLUE), ("referred", SERIES_2)):
        g = df.loc[df["status"] == status, "iou"]
        ax.hist(g, bins=bins, color=color, alpha=0.75, label=f"{status} (n={len(g)}, mean {g.mean():.3f})")
    ax.set_xlabel("Grad-CAM IoU with ground-truth lesion mask")
    ax.set_ylabel("Test images")
    ax.set_title(f"IoU: accepted vs referred predictions (r={DEFAULT_R:.0%})", fontsize=10, color=CHART_INK["primary"])
    ax.legend(frameon=False, fontsize=8)
    grid_y(ax)
    return finish_figure(fig, out_path, f"Paper: accepted {PAPER_IOU['accepted']}, referred {PAPER_IOU['referred']}.")


def run(inputs: Path, decisions: Path, out_dir: Path, images_dir: Path, masks_dir: Path, figures: bool) -> dict:
    df, raw = load_joined(inputs, decisions, DEFAULT_R)
    c8_all = pd.read_csv(decisions)
    overall, per_class, by_r = table10(df, c8_all)
    T.write_tables({"table10_iou": overall, "table10_iou_per_class": per_class,
                    "iou_by_referral_rate": by_r.round(4)}, out_dir / "tables")

    acc = df.loc[df["accepted"], "iou"].mean()
    ref = df.loc[~df["accepted"], "iou"].mean()
    checks = [
        {"check": "Accepted-prediction IoU exceeds referred-prediction IoU",
         "value": f"{acc:.3f} vs {ref:.3f}", "passed": bool(acc > ref)},
        {"check": "Every test image has a ground-truth mask (coverage)",
         "value": f"{int(df['iou'].notna().sum())}/{len(df)}", "passed": bool(df["iou"].notna().all())},
        {"check": "Grad-CAM target class equals predicted class for every image",
         "value": f"{int((df['target_class'] == df['pred_class']).sum())}/{len(df)}",
         "passed": bool((df["target_class"] == df["pred_class"]).all())},
    ]

    if figures:
        a = _Assets(df, raw, images_dir, masks_dir)
        fdir = out_dir / "figures"
        fig8_one_per_class(df, a, fdir / "fig8_one_per_class.png")
        fig9_correct_vs_incorrect(df, a, fdir / "fig9_correct_vs_incorrect.png")
        fig10_grid(df, a, fdir / "fig10_per_class_grid.png")
        fig11_by_entropy(df, a, fdir / "fig11_by_entropy.png")
        plot_iou_distribution(df, fdir / "iou_accepted_vs_referred.png")

    payload = {"referral_rate_for_split": DEFAULT_R, "n": len(df), "iou_mean": float(df["iou"].mean()),
               "iou_accepted": float(acc), "iou_referred": float(ref),
               "iou_correct": float(df.loc[df["correct"], "iou"].mean()),
               "iou_incorrect": float(df.loc[~df["correct"], "iou"].mean()),
               "by_referral_rate": by_r.to_dict(orient="records"), "paper": PAPER_IOU, "sanity_checks": checks}
    write_json(payload, out_dir / "metrics.json")

    lines = [
        "# V4 — Grad-CAM Explainability + Quantitative IoU",
        "",
        "Generated by `python -m src.eval.run_v4` from C7 (`results/v4/inputs/`) and the V3 C8 decisions.",
        "Grad-CAM on `model.features[-1]` (1280×7×7) of the locked `v2-main-b0`, eval mode (dropout off), target = "
        "predicted class, bilinear upsample to 224, binarized at 50% of each map's max, IoU against the HAM10000 "
        "lesion mask (Dataverse `HAM10000_segmentations_lesion_tschandl.zip`, nearest-neighbour resize to 224). "
        f"Accepted/referred uses the V3 entropy referral at r={DEFAULT_R:.0%} (Table 10's r is unreadable in our copy "
        "of the paper; PLAN.md default).",
        "",
        "## Table 10 — IoU",
        "",
        overall.to_markdown(index=False),
        "",
        "### Per class",
        "",
        per_class.to_markdown(index=False),
        "",
        "### Accepted vs referred IoU at every referral rate",
        "",
        by_r.round(4).to_markdown(index=False),
        "",
        "## Sanity checks (PLAN.md §8.5)",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
        "## Figures",
        "",
        "- `figures/fig8_one_per_class.png` — paper Fig. 8",
        "- `figures/fig9_correct_vs_incorrect.png` — paper Fig. 9",
        "- `figures/fig10_per_class_grid.png` — paper Fig. 10",
        "- `figures/fig11_by_entropy.png` — paper Fig. 11",
        "- `figures/iou_accepted_vs_referred.png`",
        "",
        "Example selection is rule-based (module docstring), so no example was hand-picked. A high IoU means the "
        "map overlaps the lesion; it does not show the model's reasoning is clinically correct.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}; IoU accepted {acc:.3f} vs referred {ref:.3f}; checks "
          f"{sum(c['passed'] for c in checks)}/{len(checks)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 result package (PLAN.md §8).")
    parser.add_argument("--inputs", type=Path, default=Path("results/v4/inputs"))
    parser.add_argument("--decisions", type=Path, default=Path("results/v3/decisions/referral_decisions_c8.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/v4"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/cache_prep224"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/masks224"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run(args.inputs, args.decisions, args.out, args.images_dir, args.masks_dir, not args.no_figures)


if __name__ == "__main__":
    main()
