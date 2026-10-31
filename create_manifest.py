from pathlib import Path
import pandas as pd

BASE = Path("data/raw/dataset anemia")
MASK_BASE = Path("data/processed/masks")
OUTPUT = Path("data/processed/manifest.csv")

rows = []

for country in ["India", "Italy"]:

    country_dir = BASE / country
    excel_path = country_dir / f"{country}.xlsx"

    df = pd.read_excel(excel_path)

    # Clean Excel Number
    df["Number"] = pd.to_numeric(
        df["Number"],
        errors="coerce"
    )

    df = df.dropna(subset=["Number"])
    df["Number"] = df["Number"].astype(int)

    excel_lookup = df.set_index("Number").to_dict("index")

    patient_dirs = sorted(
        [
            p for p in country_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda x: int(x.name)
    )

    for patient_dir in patient_dirs:

        patient_id = int(patient_dir.name)

        if patient_id not in excel_lookup:
            print(f"[SKIP] {country}/{patient_id}: not in Excel")
            continue

        record = excel_lookup[patient_id]

        # Find original JPG
        jpg_files = list(patient_dir.glob("*.jpg"))

        if not jpg_files:
            print(f"[SKIP] {country}/{patient_id}: no JPG")
            continue

        image_path = jpg_files[0]

        # Generated U-Net ground-truth mask
        mask_path = (
            MASK_BASE
            / country
            / f"{patient_id}.png"
        )

        if not mask_path.exists():
            print(
                f"[SKIP] {country}/{patient_id}: "
                f"mask missing"
            )
            continue

        rows.append({
            "patient_id": patient_id,
            "country": country,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "hgb": record.get("Hgb"),
            "gender": record.get("Gender"),
            "age": record.get("Age"),
            "note": record.get("Note"),
        })

manifest = pd.DataFrame(rows)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

manifest.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("MANIFEST CREATED")
print("==============================")
print("Total records :", len(manifest))
print("India         :", (manifest["country"] == "India").sum())
print("Italy         :", (manifest["country"] == "Italy").sum())
print("Missing Hgb   :", manifest["hgb"].isna().sum())
print("Missing image :", manifest["image_path"].isna().sum())
print("Missing mask  :", manifest["mask_path"].isna().sum())
print("\nSaved:", OUTPUT)
print("\nFirst 5 records:")
print(manifest.head().to_string(index=False))
