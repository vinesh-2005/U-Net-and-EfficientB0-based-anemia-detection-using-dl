#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_yaml
from src.segmentation.predict import predict_mask
from src.roi.extract import extract_roi


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--split",
        default="data/splits/split_v1.csv"
    )

    p.add_argument(
        "--weights",
        required=True
    )

    p.add_argument(
        "--seg-config",
        default="configs/segmentation_unet.yaml"
    )

    p.add_argument(
        "--cls-config",
        default="configs/classifier_efficientnetb0.yaml"
    )

    p.add_argument(
        "--output",
        default="data/processed/roi_predicted"
    )

    a = p.parse_args()

    # Load configs
    sc = load_yaml(a.seg_config)
    cc = load_yaml(a.cls_config)

    # Build U-Net
    from src.segmentation.model import build_unet

    model = build_unet(
        tuple(sc["input_size"]),
        sc["base_filters"]
    )

    # Load trained U-Net
    model.load_weights(a.weights)

    # Output directory
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # Read split
    frame = pd.read_csv(a.split)

    rows = []

    print(f"Total rows: {len(frame)}")
    print("Generating predicted ROIs...")

    for i, row in enumerate(frame.itertuples(), start=1):

        # Accept both quality status names
        if row.quality_status not in ("accepted", "ok"):
            continue

        # Skip missing image
        if pd.isna(row.image_path):
            continue

        # Skip missing label
        if pd.isna(row.label_binary):
            continue

        try:
            # -------------------------
            # U-NET PREDICTION
            # -------------------------
            image, mask, binary_mask, quality = predict_mask(
                model,
                row.image_path,
                tuple(sc["input_size"]),
                sc["mask_threshold"],
                sc.get("postprocess", {})
            )

            # Skip poor predicted masks
            if quality["quality_status"] != "accepted":
                continue

            # -------------------------
            # ROI EXTRACTION
            # -------------------------
            roi, bbox = extract_roi(
                image,
                binary_mask,
                tuple(cc["input_size"]),
                cc["roi_margin_fraction"],
                cc.get("mask_outside_roi", False)
            )

            # -------------------------
            # SAVE ROI WITH UNIQUE FILENAME
            # -------------------------
            country = "India" if "India" in str(row.image_path) else "Italy"
            unique_name = f"{country}_{row.patient_id}_{row.image_id}.png"
            path = out / unique_name

            cv2.imwrite(
                str(path),
                cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            )

            # -------------------------
            # SAVE METADATA
            # -------------------------
            rows.append({
                "patient_id": row.patient_id,
                "image_id": row.image_id,
                "country": country,
                "roi_path": str(path),
                "label_binary": int(row.label_binary),
                "split": row.split,
                **quality,
                **bbox
            })

            if i % 10 == 0:
                print(f"Processed {i}/{len(frame)}")

        except Exception as e:
            print(
                f"ERROR: image_id={row.image_id} -> {e}"
            )

    # -------------------------
    # SAVE ROI MANIFEST
    # -------------------------
    roi_manifest = out / "roi_manifest.csv"

    pd.DataFrame(rows).to_csv(
        roi_manifest,
        index=False
    )

    print("\n================================")
    print("ROI generation completed")
    print("================================")
    print(f"ROIs generated : {len(rows)}")
    print(f"Output folder  : {out}")
    print(f"ROI manifest   : {roi_manifest}")
