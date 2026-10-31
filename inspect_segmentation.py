import cv2
from pathlib import Path

base = Path("data/raw/dataset anemia/Italy")

for number in ["9", "95"]:
    print(f"\n--- Italy/{number} ---")

    for f in sorted((base / number).glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)

        channels = img.shape[2] if img.ndim == 3 else 1

        print(
            f"{f.name}: "
            f"shape={img.shape}, "
            f"channels={channels}, "
            f"min={img.min()}, "
            f"max={img.max()}"
        )
