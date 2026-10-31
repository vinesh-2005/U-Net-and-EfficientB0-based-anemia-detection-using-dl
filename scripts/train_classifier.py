#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml
from src.common.reproducibility import seed_everything
from src.classification.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/classifier_efficientnetb0.yaml"); p.add_argument("--rois", required=True); p.add_argument("--output", default="artifacts/current/classifier"); a = p.parse_args()
    seed_everything(2025); train(load_yaml(a.config), a.rois, a.output)
