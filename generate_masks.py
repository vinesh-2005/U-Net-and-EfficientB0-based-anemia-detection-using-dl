from pathlib import Path
import cv2
import numpy as np

RAW_DIR = Path("data/raw/dataset anemia")
MASK_DIR = Path("data/processed/masks")

MASK_DIR.mkdir(parents=True, exist_ok=True)

total = 0
rgba_count = 0
rgb_count = 0
palpebral_only = 0
failed = []

for country in ["India", "Italy"]:

    country_dir = RAW_DIR / country
    output_dir = MASK_DIR / country
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        [
            p for p in country_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda x: int(x.name)
    )

    for patient_dir in patient_dirs:

        # Find all PNG files
        png_files = list(patient_dir.glob("*.png"))

        # Combined conjunctiva segmentation
        combined_files = [
            f for f in png_files
            if "forniceal_palpebral" in f.name.lower()
        ]

        # Palpebral-only segmentation
        palpebral_files = [
            f for f in png_files
            if "palpebral" in f.name.lower()
            and "forniceal_palpebral" not in f.name.lower()
        ]

        # Prefer combined segmentation
        if combined_files:

            source = combined_files[0]
            source_type = "forniceal_palpebral"

        # Otherwise use palpebral segmentation
        elif palpebral_files:

            source = palpebral_files[0]
            source_type = "palpebral_only"
            palpebral_only += 1

        else:

            failed.append(
                f"{country}/{patient_dir.name}: no segmentation"
            )
            continue

        # Read image
        img = cv2.imread(
            str(source),
            cv2.IMREAD_UNCHANGED
        )

        if img is None:

            failed.append(
                f"{country}/{patient_dir.name}: unreadable"
            )
            continue

        # --------------------------------
        # RGBA IMAGE
        # --------------------------------
        if img.ndim == 3 and img.shape[2] == 4:

            alpha = img[:, :, 3]

            mask = np.where(
                alpha > 0,
                255,
                0
            ).astype(np.uint8)

            rgba_count += 1

        # --------------------------------
        # RGB IMAGE
        # --------------------------------
        elif img.ndim == 3 and img.shape[2] == 3:

            # White background → background
            # Colored/different pixels → foreground

            distance_from_white = np.sqrt(
                np.sum(
                    (255.0 - img.astype(np.float32)) ** 2,
                    axis=2
                )
            )

            mask = np.where(
                distance_from_white > 15,
                255,
                0
            ).astype(np.uint8)

            rgb_count += 1

        else:

            failed.append(
                f"{country}/{patient_dir.name}: "
                f"unsupported shape {img.shape}"
            )
            continue

        # Save mask
        output_path = (
            output_dir / f"{patient_dir.name}.png"
        )

        cv2.imwrite(
            str(output_path),
            mask
        )

        total += 1

        print(
            f"[OK] {country}/{patient_dir.name} "
            f"-> {source_type}"
        )


# --------------------------------
# FINAL SUMMARY
# --------------------------------

print("\n==============================")
print("MASK GENERATION COMPLETE")
print("==============================")

print("Masks generated :", total)
print("RGBA masks      :", rgba_count)
print("RGB masks       :", rgb_count)
print("Palpebral only  :", palpebral_only)
print("Failed          :", len(failed))

if failed:

    print("\nFAILED ITEMS:")

    for item in failed:
        print(" -", item)
