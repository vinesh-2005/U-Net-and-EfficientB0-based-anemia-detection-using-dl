#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml
from src.common.reproducibility import seed_everything
from src.segmentation.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/segmentation_unet.yaml"); p.add_argument("--split", default="data/splits/split_v1.csv"); p.add_argument("--output", default="artifacts/current/segmentation"); a = p.parse_args()
    config = load_yaml(a.config); seed_everything(2025); train(config, a.split, a.output)
