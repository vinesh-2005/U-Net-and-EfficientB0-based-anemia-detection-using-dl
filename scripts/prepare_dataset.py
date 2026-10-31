#!/usr/bin/env python3
"""Inventory, build, and split a verified dataset."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml
from src.data.inventory import inventory, write_inventory_report
from src.data.build_manifest import build_manifest
from src.data.make_splits import make_patient_splits

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-config", default="configs/data.yaml"); parser.add_argument("--labels-config", default="configs/labels.yaml")
    args = parser.parse_args(); config = load_yaml(args.data_config)
    write_inventory_report(inventory(config["dataset_root"], config["image_extensions"]), "data/interim/inventory.json")
    build_manifest(args.data_config, args.labels_config, config["output_manifest"], config["output_excluded"])
    make_patient_splits(config["output_manifest"], "data/splits/split_v1.csv", config["split"]["seed"], (config["split"]["train"], config["split"]["val"], config["split"]["test"]))

if __name__ == "__main__": main()
