import pandas as pd
from pathlib import Path

base = Path("data/raw/dataset anemia")

for country in ["India", "Italy"]:
    print("\n====================")
    print(country)
    print("====================")

    excel_path = base / country / f"{country}.xlsx"
    country_dir = base / country

    df = pd.read_excel(excel_path)

    folders = sorted(
        int(p.name)
        for p in country_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    excel_numbers = set(
        pd.to_numeric(df["Number"], errors="coerce")
        .dropna()
        .astype(int)
    )

    folder_numbers = set(folders)

    print("Folders       :", len(folders))
    print("Excel records :", len(excel_numbers))
    print("Missing in Excel:", sorted(folder_numbers - excel_numbers))
    print("Missing folders:", sorted(excel_numbers - folder_numbers))
    print("Missing Hb:", int(df["Hgb"].isna().sum()))
    print("Duplicate Excel Numbers:", int(df["Number"].duplicated().sum()))

