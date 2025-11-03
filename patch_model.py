

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches

from tiatoolbox.models.engine.patch_predictor import PatchPredictor, IOPatchPredictorConfig
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader


# ------------------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Patch Prediction for Tile and WSI")

parser.add_argument(
    "--model",
    type=str,
    default="googlenet-kather100k",
    help="Pretrained model to use (e.g., googlenet-kather100k, resnet50-kather100k).",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to run the model on: 'cpu' or 'cuda'.",
)
parser.add_argument(
    "--tile_path",
    type=str,
    required=False,
    default="D:/patch_model/TCGA-FF-8042-01Z-00-DX1.bf80d981-9209-4933-8f80-e9e689971999_thumbnail.png",
    help="Path to the tile image (PNG or JPG).",
)
parser.add_argument(
    "--wsi_path",
    type=str,
    required=False,
    default="D:/patch_model/TCGA-FF-8042-01Z-00-DX1.bf80d981-9209-4933-8f80-e9e689971999.svs",
    help="Path to the WSI file (.svs, .tif, etc.).",
)
parser.add_argument(
    "--weights",
    type=str,
    default=r"D:/patch_model/models/googlenet-kather100k.pth",
    help="Path to the pretrained weights file (.pth).",
)
args = parser.parse_args()


# ------------------------------------------------------------------------------
# Basic Checks and Logging
# ------------------------------------------------------------------------------
tile_file = Path(args.tile_path)
wsi_file = Path(args.wsi_path)
weights_path = Path(args.weights)
device = args.device
model_name = args.model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.info(f"Model: {model_name}")
logging.info(f"Device: {device}")
logging.info(f"Tile Path: {tile_file}")
logging.info(f"WSI Path: {wsi_file}")
logging.info(f"Weights Path: {weights_path}")

# Validate file existence
for f in [tile_file, wsi_file, weights_path]:
    if not f.exists():
        logging.error(f"File not found: {f}")
        sys.exit(1)

# ------------------------------------------------------------------------------
# Label and Color Mapping
# ------------------------------------------------------------------------------
label_dict = {
    "BACK": 0, "NORM": 1, "DEB": 2, "TUM": 3,
    "ADI": 4, "MUC": 5, "MUS": 6, "STR": 7, "LYM": 8
}

colors = cm.get_cmap("Set1").colors
label_color_dict = {0: ("empty", (0, 0, 0))}
for class_name, label in label_dict.items():
    label_color_dict[label + 1] = (class_name, 255 * np.array(colors[label]))


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def calculate_patch_percentages(pred_map, label_color_dict):
    """Calculate the percentage of each class in the prediction map."""
    unique_labels, counts = np.unique(pred_map, return_counts=True)
    total_patches = np.sum(counts)
    patch_percentages = {}
    for label, count in zip(unique_labels, counts):
        if label == 0:
            continue
        class_name = label_color_dict[label][0]
        patch_percentages[class_name] = (count, count / total_patches * 100)
    return total_patches, patch_percentages


def add_color_legend(ax, label_color_dict):
    """Add color legend to Matplotlib plot."""
    legend_patches = [
        patches.Patch(color=np.array(color) / 255, label=class_name)
        for label, (class_name, color) in label_color_dict.items() if class_name != "empty"
    ]
    ax.legend(
        handles=legend_patches,
        title="Tissue Classes",
        loc="upper right",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )


# ------------------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------------------
logging.info("Loading model and weights...")

predictor = PatchPredictor(
    pretrained_model=model_name,
    pretrained_weights=str(weights_path),
    batch_size=32,
)

# ------------------------------------------------------------------------------
# TILE PREDICTION
# ------------------------------------------------------------------------------
logging.info("Running tile-level prediction...")

tile_output = predictor.predict(
    imgs=[tile_file],
    mode="tile",
    merge_predictions=True,
    patch_input_shape=[224, 224],
    stride_shape=[224, 224],
    resolution=1,
    units="baseline",
    return_probabilities=True,
    device=device,
)

pred_map_tile = predictor.merge_predictions(
    tile_file, tile_output[0], resolution=1, units="baseline"
)

tile_image = imread(tile_file)

overlay_tile = overlay_prediction_mask(
    tile_image, pred_map_tile, alpha=0.5, label_info=label_color_dict, return_ax=False
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(overlay_tile)
ax.set_title("Tile-Level Prediction (All 9 Classes)", fontsize=13)
ax.axis("off")
add_color_legend(ax, label_color_dict)
plt.show()

tile_total, tile_percentages = calculate_patch_percentages(pred_map_tile, label_color_dict)
logging.info(f"Tile-level Patch Summary (Total patches: {tile_total})")
for cls, (cnt, pct) in tile_percentages.items():
    logging.info(f"{cls}: {cnt} patches ({pct:.2f}%)")


# ------------------------------------------------------------------------------
# WSI PREDICTION
# ------------------------------------------------------------------------------
logging.info("Running WSI-level prediction...")

wsi_ioconfig = IOPatchPredictorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.5}],
    patch_input_shape=[224, 224],
    stride_shape=[224, 224],
)

wsi_output = predictor.predict(
    imgs=[wsi_file],
    masks=None,
    mode="wsi",
    merge_predictions=False,
    ioconfig=wsi_ioconfig,
    return_probabilities=True,
    device=device,
)

wsi = WSIReader.open(str(wsi_file))
overview_resolution = 4
overview_unit = "mpp"
wsi_overview = wsi.slide_thumbnail(resolution=overview_resolution, units=overview_unit)

pred_map_wsi = predictor.merge_predictions(
    wsi_file, wsi_output[0], resolution=overview_resolution, units=overview_unit
)

overlay_wsi = overlay_prediction_mask(
    wsi_overview, pred_map_wsi, alpha=0.5, label_info=label_color_dict, return_ax=False
)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(overlay_wsi)
ax.set_title("WSI-Level Prediction (All 9 Classes)", fontsize=13)
ax.axis("off")
add_color_legend(ax, label_color_dict)
plt.show()

wsi_total, wsi_percentages = calculate_patch_percentages(pred_map_wsi, label_color_dict)
logging.info(f"WSI-level Patch Summary (Total patches: {wsi_total})")
for cls, (cnt, pct) in wsi_percentages.items():
    logging.info(f"{cls}: {cnt} patches ({pct:.2f}%)")

logging.info("✅ Prediction completed successfully!")
