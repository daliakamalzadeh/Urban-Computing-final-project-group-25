"""
Loading the NAB NYC-Taxi dataset and its labelled anomaly windows.
The files are downloaded once from GitHub and cached in ``data/``.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import Range, normalize_ranges

NAB_RAW = "https://raw.githubusercontent.com/numenta/NAB/master"
SERIES_KEY = "realKnownCause/nyc_taxi.csv"
FILES = {"nyc_taxi.csv": f"{NAB_RAW}/data/{SERIES_KEY}",
         "combined_windows.json": f"{NAB_RAW}/labels/combined_windows.json"}


@dataclass
class Dataset:
    name: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    anomalies: list[Range]          # ground-truth ranges 

    @property
    def n(self) -> int:
        return len(self.values)


def load_nyc_taxi(data_dir: str | Path = "data") -> Dataset:
    data_dir = Path(data_dir)
    for name, url in FILES.items():            # download once, then reuse the cached copy
        if not (data_dir / name).exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, data_dir / name)

    df = pd.read_csv(data_dir / "nyc_taxi.csv", parse_dates=["timestamp"])
    ts = pd.DatetimeIndex(df["timestamp"])
    windows = json.loads((data_dir / "combined_windows.json").read_text())[SERIES_KEY]
    
    ranges = [(ts.searchsorted(pd.Timestamp(s)), ts.searchsorted(pd.Timestamp(e), side="right") - 1)
              for s, e in windows]
    return Dataset("NYC-Taxi", ts, df["value"].to_numpy(float), normalize_ranges(ranges, len(df)))
