"""Build the canonical manifest only from explicit IDs and verified masks.

Dataset-specific filename and clinical-column mappings are intentionally configurable;
this module refuses to infer a label from row order or an unconfirmed threshold.
"""
from pathlib import Path
import hashlib, re
import cv2
import pandas as pd
from .validate_dataset import inspect_annotation, to_binary_mask
from ..common.config import load_yaml, require_confirmed_label_rule

FIELDS = ["patient_id", "image_id", "image_path", "mask_path", "hb_g_dl", "sex", "age_years",
          "label_binary", "label_definition", "split", "has_valid_mask", "quality_status"]

def normalize_id(value, strip_leading_zeros=False):
    value = str(value).strip().casefold()
    return value.lstrip("0") if strip_leading_zeros else value

def build_manifest(data_config_path: str, labels_config_path: str, output_manifest: str, output_excluded: str) -> pd.DataFrame:
    config, labels = load_yaml(data_config_path), load_yaml(labels_config_path)
    threshold = require_confirmed_label_rule(labels)
    root = Path(config["dataset_root"])
    clinical_path = config.get("clinical_table")
    if not clinical_path: raise ValueError("Set configs/data.yaml: clinical_table to the verified EYES-DEFY-ANEMIA metadata table.")
    clinical_path = Path(clinical_path)
    if not clinical_path.is_absolute(): clinical_path = root / clinical_path
    clinical = pd.read_csv(clinical_path) if str(clinical_path).lower().endswith(".csv") else pd.read_excel(clinical_path)
    columns = config.get("clinical_columns", {})
    required = {columns.get("patient_id", "patient_id"), columns.get("hb_g_dl", "hb_g_dl")}
    missing = required - set(clinical.columns)
    if missing: raise ValueError(f"Clinical table must explicitly contain columns: {sorted(required)}; missing {sorted(missing)}")
    clinical = clinical.rename(columns={columns.get("patient_id", "patient_id"): "patient_id", columns.get("hb_g_dl", "hb_g_dl"): "hb_g_dl",
                                        columns.get("sex", "sex"): "sex", columns.get("age_years", "age_years"): "age_years"})
    clinical["_pid"] = clinical.patient_id.map(lambda x: normalize_id(x, config["id"].get("strip_leading_zeros", False)))
    if clinical._pid.duplicated().any(): raise ValueError("Duplicate clinical patient IDs are ambiguous; resolve them before joining.")
    clinical["hb_g_dl"] = pd.to_numeric(clinical.hb_g_dl, errors="coerce")
    rows, excluded, seen_hashes = [], [], set()
    image_root, ann_root = root / config["images_dir"], root / config["annotations_dir"]
    for image in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in config["image_extensions"]):
        image_id = image.stem
        patient_regex = config["id"].get("image_patient_regex")
        if patient_regex:
            match_id = re.search(patient_regex, image.stem)
            patient_raw = match_id.group("patient_id") if match_id and "patient_id" in match_id.groupdict() else image.parent.name
        else:
            patient_raw = image.parent.name
        patient_id = normalize_id(patient_raw, config["id"].get("strip_leading_zeros", False))
        reasons = []
        data = image.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes: reasons.append("duplicate_image")
        seen_hashes.add(digest)
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        match = clinical[clinical._pid.eq(patient_id)]
        if decoded is None: reasons.append("corrupted_image")
        if len(match) == 0: reasons.append("missing_or_ambiguous_patient_mapping")
        if len(match) > 1: reasons.append("ambiguous_patient_mapping")
        candidates = [p for p in ann_root.rglob(f"{image.stem}*" ) if p.suffix.lower() in config["image_extensions"]]
        ann = next((p for p in candidates if any(p.stem.endswith(suffix) for suffix in config["annotation_suffixes"])), None)
        ann = ann or (candidates[0] if candidates else None)
        info = inspect_annotation(image, ann) if ann else {"valid": False, "reason": "missing_or_invalid_mask"}
        if not info.get("valid"): reasons.append(str(info.get("reason", "missing_or_invalid_mask")))
        hb = float(match.iloc[0].hb_g_dl) if len(match) == 1 and pd.notna(match.iloc[0].hb_g_dl) else None
        if hb is None: reasons.append("missing_hb")
        valid = not reasons
        row = {"patient_id": patient_id, "image_id": image_id, "image_path": str(image),
               "mask_path": str(ann) if valid else "", "hb_g_dl": hb,
               "sex": match.iloc[0].get("sex", "") if len(match) == 1 else "",
               "age_years": match.iloc[0].get("age_years", "") if len(match) == 1 else "",
               "label_binary": int(hb < threshold) if hb is not None else "",
               "label_definition": labels["target"]["label_rule_id"], "split": "",
               "has_valid_mask": bool(valid), "quality_status": "accepted" if valid else ";".join(sorted(set(reasons)))}
        (rows if valid else excluded).append(row)
    frame = pd.DataFrame(rows, columns=FIELDS); excluded_frame = pd.DataFrame(excluded)
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True); frame.to_csv(output_manifest, index=False)
    excluded_frame.to_csv(output_excluded, index=False)
    return frame
