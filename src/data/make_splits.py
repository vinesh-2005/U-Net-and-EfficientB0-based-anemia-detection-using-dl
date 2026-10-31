"""Create deterministic patient-disjoint train/val/test assignments."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def make_patient_splits(manifest_path: str, output_path: str, seed: int, ratios=(.70, .15, .15)) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    usable = df[df["quality_status"].eq("accepted")].copy()
    patients = usable.groupby("patient_id", as_index=False).agg(label_binary=("label_binary", "first"))
    strat = patients["label_binary"] if patients["label_binary"].nunique() > 1 and patients["label_binary"].value_counts().min() >= 2 else None
    train, hold = train_test_split(patients, test_size=ratios[1] + ratios[2], random_state=seed, stratify=strat)
    strat_hold = hold["label_binary"] if hold["label_binary"].nunique() > 1 and hold["label_binary"].value_counts().min() >= 2 else None
    val, test = train_test_split(hold, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=seed, stratify=strat_hold)
    assignment = {p: "train" for p in train.patient_id} | {p: "val" for p in val.patient_id} | {p: "test" for p in test.patient_id}
    df["split"] = df.patient_id.map(assignment).fillna("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
