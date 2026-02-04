"""Unified external data loaders for wind/environmental factors.

All loaders expose get_wind_factor(t, x=None, y=None) for use in DisasterScenario.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta


class VizagLoader:
    """Loader for Visakhapatnam tidal data (prs/m pressure -> wind factor)."""

    def __init__(self, csv_path: str = "Visakhapatnam_UTide_full2024_hourly_IST.csv"):
        self.csv_path = Path(csv_path)
        project_root = Path(__file__).resolve().parents[1]
        if not self.csv_path.is_absolute():
            self.csv_path = project_root.parent / self.csv_path
        self.data = None
        self.start_time = None
        self.end_time = None
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            self.data = None
            return
        try:
            df = pd.read_csv(self.csv_path)
            if "Time(IST)" in df.columns:
                df["Time(IST)"] = pd.to_datetime(df["Time(IST)"])
                df = df.set_index("Time(IST)")
            col = "prs(m)" if "prs(m)" in df.columns else "pressure"
            self.data = df[[col]].rename(columns={col: "value"})
            self.start_time = self.data.index[0]
            self.end_time = self.data.index[-1]
        except Exception:
            self.data = None

    def get_wind_factor(self, t: float, x: Optional[float] = None, y: Optional[float] = None) -> float:
        """Return wind modification factor (0.5–1.5) from tidal pressure."""
        if self.data is None or len(self.data) == 0:
            return 1.0
        target = self.start_time + timedelta(seconds=t)
        if target > self.end_time:
            t = t % (self.end_time - self.start_time).total_seconds()
            target = self.start_time + timedelta(seconds=t)
        target = max(self.start_time, min(self.end_time, target))
        idx = self.data.index.get_indexer([target], method="nearest")[0]
        val = float(self.data.iloc[idx]["value"])
        vmin, vmax = self.data["value"].min(), self.data["value"].max()
        if vmax == vmin:
            return 1.0
        norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
        return 0.5 + norm * 1.0


class NOAALoader:
    """Loader for NOAA water level data (e.g. San Diego) -> wind factor."""

    def __init__(self, data_dir: Optional[str] = None):
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else project_root.parent / "data" / "noaa" / "sandiego"
        if not self.data_dir.is_absolute():
            self.data_dir = project_root.parent / self.data_dir
        self.data = None
        self.start_time = None
        self.end_time = None
        self._load()

    def _load(self):
        if not self.data_dir.exists():
            self.data = None
            return
        dfs = []
        for f in sorted(self.data_dir.glob("2024_*.csv")):
            try:
                df = pd.read_csv(f)
                time_col = "Date Time" if "Date Time" in df.columns else df.columns[0]
                level_col = " Water Level" if " Water Level" in df.columns else "Water Level"
                if level_col not in df.columns:
                    level_col = [c for c in df.columns if "level" in c.lower() or "water" in c.lower()]
                    level_col = level_col[0] if level_col else None
                if level_col is None:
                    continue
                df["time"] = pd.to_datetime(df[time_col])
                df["value"] = pd.to_numeric(df[level_col].astype(str).str.replace(",", ""), errors="coerce")
                dfs.append(df[["time", "value"]].dropna())
            except Exception:
                continue
        if not dfs:
            self.data = None
            return
        self.data = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
        self.data = self.data.set_index("time")
        self.start_time = self.data.index[0]
        self.end_time = self.data.index[-1]

    def get_wind_factor(self, t: float, x: Optional[float] = None, y: Optional[float] = None) -> float:
        """Return wind factor from normalized water level."""
        if self.data is None or len(self.data) == 0:
            return 1.0
        duration = (self.end_time - self.start_time).total_seconds()
        t_wrap = t % duration if duration > 0 else 0
        target = self.start_time + timedelta(seconds=t_wrap)
        target = max(self.start_time, min(self.end_time, target))
        idx = self.data.index.get_indexer([target], method="nearest")[0]
        val = float(self.data.iloc[idx]["value"])
        vmin, vmax = self.data["value"].min(), self.data["value"].max()
        if vmax == vmin:
            return 1.0
        norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
        return 0.5 + norm * 1.0


class AMOVFLYLoader:
    """Loader for AMOVFLY wind data (w_s, w_a) -> wind factor."""

    def __init__(self, data_dir: Optional[str] = None):
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else project_root.parent / "data" / "amovfly" / "wind"
        if not self.data_dir.is_absolute():
            self.data_dir = project_root.parent / self.data_dir
        self.data = None
        self._load()

    def _load(self):
        if not self.data_dir.exists():
            self.data = None
            return
        dfs = []
        for f in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                if "time" in df.columns and "w_s" in df.columns:
                    df["value"] = df["w_s"].astype(float)
                elif "time" in df.columns and "w_a" in df.columns:
                    df["value"] = np.clip(df["w_a"].astype(float) / 360.0, 0, 1)
                else:
                    continue
                dfs.append(df[["time", "value"]].dropna())
            except Exception:
                continue
        if not dfs:
            self.data = None
            return
        self.data = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
        self.t_max = float(self.data["time"].max()) if len(self.data) > 0 else 1.0

    def get_wind_factor(self, t: float, x: Optional[float] = None, y: Optional[float] = None) -> float:
        """Return wind factor from AMOVFLY wind speed/direction."""
        if self.data is None or len(self.data) == 0:
            return 1.0
        t_wrap = t % self.t_max if self.t_max > 0 else 0
        idx = np.searchsorted(self.data["time"].values, t_wrap, side="right") - 1
        idx = max(0, min(idx, len(self.data) - 1))
        val = float(self.data.iloc[idx]["value"])
        vmin, vmax = self.data["value"].min(), self.data["value"].max()
        if vmax == vmin:
            return 1.0
        norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
        return 0.5 + norm * 1.0


def get_data_loader(source: Optional[str] = None) -> Optional[object]:
    """Factory: return loader for 'vizag', 'noaa', 'amovfly', or None."""
    if source == "vizag":
        return VizagLoader()
    if source == "noaa":
        return NOAALoader()
    if source == "amovfly":
        return AMOVFLYLoader()
    return None
