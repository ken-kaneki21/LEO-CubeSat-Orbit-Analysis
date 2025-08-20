"""
LEO Satellite Orbit Propagation and Maneuver Planning — Historical Window (No-Poliastro Version)
Period: September 1, 2024 → February 28, 2025 (inclusive)

This version removes the poliastro & astropy dependency to simplify installs on Windows.
It uses only: skyfield, sgp4, numpy, pandas, matplotlib, requests.

What this script does
---------------------
1) Pulls **historical TLEs** between 2024‑09‑01 and 2025‑02‑28 (Space-Track API or local file).
2) Uses **SGP4** (Skyfield) to propagate **24‑hour ground tracks** (weekly samples by default).
3) Summarizes weekly osculating elements (a, e, i, RAAN) and RAAN drift across the window.
4) Picks a **mid‑window reference epoch** and computes a **Hohmann altitude change** plan (ΔV, timing, transfer time).
5) Estimates **propellant** via Tsiolkovsky.

Outputs
-------
- figures/ground_track_YYYYMMDD.png   — weekly ground tracks over the window
- figures/altitude_time_YYYYMMDD.png  — altitude profiles for each 24h window
- summary/orbit_weekly_summary.csv    — elements + RAAN drift per sample
- summary/maneuver_plan.txt           — ΔV breakdown & fuel estimate at reference date

Usage (quick)
-------------
1) Install deps in your venv:
   pip install numpy pandas matplotlib skyfield sgp4 requests
2) Set NORAD_IDS and (optionally) Space-Track creds below.
3) Run: python leo_historical_tle_nopoliastro.py
"""

from __future__ import annotations
import os
import math
import time
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skyfield.api import Loader, EarthSatellite, wgs84

# -----------------------------
# CONSTANTS (no poliastro/astropy required)
# -----------------------------
MU_EARTH_KM3_S2 = 398600.4418           # Earth GM [km^3/s^2]
R_EARTH_KM      = 6378.137              # Earth equatorial radius [km]
G0              = 9.80665               # m/s^2

# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------
START_UTC = datetime(2024, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
END_UTC   = datetime(2025, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

# Provide at least one NORAD ID (placeholder below — replace with your target)
NORAD_IDS = [42915]  # e.g., [25544] for ISS (for a quick test)

# Weekly sampling across the window (set to 1 for daily, 7 for weekly, etc.)
SAMPLE_EVERY_DAYS = 7

# Propagation settings for each selected epoch
PROP_DURATION_HOURS = 24
PROP_STEP_SECONDS   = 60

# Ground station (optional): Bangalore approx; used for a map marker
GROUND_STATION_LAT = 12.9716
GROUND_STATION_LON = 77.5946

# Maneuver plan assumptions (for mid‑window reference epoch)
TARGET_ALTITUDE_DELTA_KM = +50.0  # raise circular altitude by +50 km
M0_KG  = 5.0
ISP_S  = 220.0

# Space-Track credentials (optional but recommended for reliable history)
SPACETRACK_USERNAME = os.getenv("SPACETRACK_USERNAME", "")
SPACETRACK_PASSWORD = os.getenv("SPACETRACK_PASSWORD", "")

# Fallback local TLE file (if not using Space-Track). Should contain many TLEs over time.
LOCAL_TLE_FILE = "tle_history.txt"

# Output folders
FIG_DIR = "figures"
SUM_DIR = "summary"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SUM_DIR, exist_ok=True)

# Skyfield loader cache
load = Loader(".skyfield")
ts = load.timescale()

@dataclass
class TLERow:
    norad: int
    line1: str
    line2: str
    epoch_dt: datetime

# -----------------------------
# Utilities
# -----------------------------

def world_outline():
    # simple coarse outline (avoids Cartopy)
    coast = np.array([
        [-180, -60], [-170, -55], [-100, -50], [-50, -45], [0, -40], [30, -35],
        [60, -30], [100, -20], [140, -10], [180, 0], [160, 20], [120, 30],
        [60, 40], [10, 50], [-30, 55], [-80, 60], [-120, 55], [-160, 45],
        [-180, 30], [-180, -60]
    ])
    return coast[:,0], coast[:,1]


def parse_tles_from_text(text: str, norad_filter: List[int]) -> List[TLERow]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tles: List[TLERow] = []
    i = 0
    while i < len(lines)-1:
        l1 = lines[i]
        l2 = lines[i+1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                norad = int(l1[2:7])
                if norad_filter and norad not in norad_filter:
                    i += 2
                    continue
                sat = EarthSatellite(l1, l2)
                epoch = sat.epoch.utc_datetime().replace(tzinfo=timezone.utc)
                tles.append(TLERow(norad=norad, line1=l1, line2=l2, epoch_dt=epoch))
            except Exception:
                pass
            i += 2
        else:
            i += 1
    return tles


def fetch_spacetrack_tles(norad_ids: List[int], start: datetime, end: datetime) -> List[TLERow]:
    """Fetch historical TLEs from Space-Track (requires credentials). If not set, return []."""
    import requests
    if not (SPACETRACK_USERNAME and SPACETRACK_PASSWORD):
        return []

    base = "https://www.space-track.org"
    sess = requests.Session()

    # Login
    login_data = {"identity": SPACETRACK_USERNAME, "password": SPACETRACK_PASSWORD}
    r = sess.post(base + "/ajaxauth/login", data=login_data, timeout=30)
    if r.status_code != 200:
        print("[WARN] Space-Track login failed; falling back to LOCAL_TLE_FILE if present.")
        return []

    tles: List[TLERow] = []
    for norad in norad_ids:
        query = (
            "/basicspacedata/query/class/tle/NORAD_CAT_ID/{norad}/"
            "EPOCH/{start}--{end}/orderby/EPOCH asc/format/tle"
        ).format(
            norad=norad,
            start=start.strftime("%Y-%m-%d%%20%H:%M:%S"),
            end=end.strftime("%Y-%m-%d%%20%H:%M:%S"),
        )
        resp = sess.get(base + query, timeout=60)
        if resp.status_code != 200:
            print(f"[WARN] Space-Track query failed for {norad} ({resp.status_code}).")
            continue
        tles.extend(parse_tles_from_text(resp.text, [norad]))
        time.sleep(0.3)

    return tles


def load_tles(norad_ids: List[int], start: datetime, end: datetime) -> List[TLERow]:
    tles = fetch_spacetrack_tles(norad_ids, start, end)
    if tles:
        return tles

    # Fallback: local file
    if not os.path.exists(LOCAL_TLE_FILE):
        print("[ERROR] No Space-Track credentials and LOCAL_TLE_FILE not found. Provide credentials or a local TLE file with historical data.")
        return []
    with open(LOCAL_TLE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    all_tles = parse_tles_from_text(text, norad_ids)
    # Filter by time window
    return [t for t in all_tles if start <= t.epoch_dt <= end]


def select_epochs(tles: List[TLERow], sample_every_days: int) -> List[TLERow]:
    if not tles:
        return []
    # Ensure sorted by epoch
    tles_sorted = sorted(tles, key=lambda t: t.epoch_dt)
    picked: List[TLERow] = []
    next_pick = tles_sorted[0].epoch_dt
    for tle in tles_sorted:
        if tle.epoch_dt >= next_pick or not picked:
            picked.append(tle)
            next_pick = tle.epoch_dt + timedelta(days=sample_every_days)
    return picked


def elements_from_satellite(sat: EarthSatellite):
    # Semi-major axis from mean motion (no_kozai in rad/min → rad/s)
    mm_rad_s = sat.model.no_kozai * 60.0
    a_km = (MU_EARTH_KM3_S2 / (mm_rad_s**2)) ** (1/3)
    e = sat.model.ecco
    i_deg = math.degrees(sat.model.inclo)
    raan_deg = math.degrees(sat.model.nodeo)
    return a_km, e, i_deg, raan_deg


def hohmann_delta_v(r1_km: float, r2_km: float):
    a_t = 0.5 * (r1_km + r2_km)
    v1 = math.sqrt(MU_EARTH_KM3_S2 / r1_km)
    v2 = math.sqrt(MU_EARTH_KM3_S2 / r2_km)
    v_p = math.sqrt(MU_EARTH_KM3_S2 * (2.0 / r1_km - 1.0 / a_t))
    v_a = math.sqrt(MU_EARTH_KM3_S2 * (2.0 / r2_km - 1.0 / a_t))
    dv1 = abs(v_p - v1) * 1000.0  # m/s
    dv2 = abs(v2 - v_a) * 1000.0  # m/s
    tof = math.pi * math.sqrt(a_t**3 / MU_EARTH_KM3_S2)  # s
    return dv1, dv2, tof


def tsiolkovsky(m0: float, dv_total: float, Isp: float, g0: float = G0):
    mf = m0 / math.exp(dv_total / (Isp * g0))
    return (m0 - mf), mf


def plot_groundtrack(subsat_lats, subsat_lons, title: str, outfile: str,
                     gs_lat: Optional[float] = None, gs_lon: Optional[float] = None):
    lons = ((np.array(subsat_lons) + 180) % 360) - 180
    lats = np.array(subsat_lats)
    fig, ax = plt.subplots(figsize=(11, 5))
    wx, wy = world_outline()
    ax.plot(wx, wy, lw=1, alpha=0.6)
    jumps = np.where(np.abs(np.diff(lons)) > 180)[0]
    start = 0
    for j in np.append(jumps, len(lons)-1):
        ax.plot(lons[start:j+1], lats[start:j+1], lw=1.2)
        start = j+1
    if gs_lat is not None and gs_lon is not None:
        ax.scatter([gs_lon], [gs_lat], marker='^', s=60)
        ax.text(gs_lon+2, gs_lat, "GS", fontsize=9)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def plot_altitude(time_hours, alt_km, title: str, outfile: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_hours, alt_km)
    ax.set_xlabel("Time since start (hours)")
    ax.set_ylabel("Geodetic altitude (km)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def propagate_24h(sat: EarthSatellite, start_dt: datetime, step_s: int):
    t0 = ts.from_datetime(start_dt)
    tN = ts.from_datetime(start_dt + timedelta(hours=PROP_DURATION_HOURS))
    t = t0
    times, lats, lons, alts = [], [], [], []
    while t.tt <= tN.tt:
        geoc = sat.at(t)
        sub = wgs84.subpoint(geoc)
        lats.append(sub.latitude.degrees)
        lons.append(sub.longitude.degrees)
        alts.append(sub.elevation.km)
        times.append(t)
        t = ts.from_datetime(t.utc_datetime().replace(tzinfo=timezone.utc) + timedelta(seconds=step_s))
    return times, lats, lons, alts


def main():
    if NORAD_IDS == [99999]:
        print("[NOTE] Please set NORAD_IDS to your target satellite (e.g., [25544] for ISS to test).")

    # 1) Load historical TLEs within the window
    tles = load_tles(NORAD_IDS, START_UTC, END_UTC)
    if not tles:
        return

    # 2) Select sampling epochs (weekly by default)
    epochs = select_epochs(tles, SAMPLE_EVERY_DAYS)
    print(f"Selected {len(epochs)} epoch(s) between {START_UTC.date()} and {END_UTC.date()}.")

    weekly_rows = []

    for row in epochs:
        sat = EarthSatellite(row.line1, row.line2)
        a_km, e, i_deg, raan_deg = elements_from_satellite(sat)

        # Propagate 24h from this epoch
        times, lats, lons, alts = propagate_24h(sat, row.epoch_dt, PROP_STEP_SECONDS)
        hh = [(t.tt - times[0].tt) * 24.0 for t in times]

        # Save plots
        tag = row.epoch_dt.strftime("%Y%m%d")
        gt_file = os.path.join(FIG_DIR, f"ground_track_{row.norad}_{tag}.png")
        at_file = os.path.join(FIG_DIR, f"altitude_time_{row.norad}_{tag}.png")
        plot_groundtrack(
            lats, lons,
            title=f"Ground Track (24h) from TLE epoch {row.epoch_dt.isoformat()} UTC,NORAD {row.norad}",
            outfile=gt_file,
            gs_lat=GROUND_STATION_LAT, gs_lon=GROUND_STATION_LON)
        plot_altitude(
            hh, alts,
            title=f"Altitude vs Time (24h) from TLE epoch {row.epoch_dt.isoformat()} UTC,NORAD {row.norad}",
            outfile=at_file)

        weekly_rows.append({
            "norad": row.norad,
            "epoch_utc": row.epoch_dt.isoformat(),
            "a_km": a_km,
            "e": e,
            "i_deg": i_deg,
            "raan_deg": raan_deg,
        })

    # 3) Save weekly summary and RAAN drift
    df = pd.DataFrame(weekly_rows).sort_values(["norad", "epoch_utc"]).reset_index(drop=True)
    df["epoch_dt"] = pd.to_datetime(df["epoch_utc"], utc=True)
    df["raan_drift_deg_per_day"] = np.nan
    for norad in df["norad"].unique():
        idx = df.index[df["norad"] == norad].tolist()
        for j in range(1, len(idx)):
            i0, i1 = idx[j-1], idx[j]
            draan = df.loc[i1, "raan_deg"] - df.loc[i0, "raan_deg"]
            dt_days = (df.loc[i1, "epoch_dt"] - df.loc[i0, "epoch_dt"]).total_seconds()/86400.0
            # wrap
            if draan > 180: draan -= 360
            if draan < -180: draan += 360
            df.loc[i1, "raan_drift_deg_per_day"] = draan / dt_days if dt_days > 0 else np.nan

    df.drop(columns=["epoch_dt"], inplace=True)
    out_csv = os.path.join(SUM_DIR, "orbit_weekly_summary.csv")
    df.to_csv(out_csv, index=False)

    # 4) Maneuver plan at mid-window epoch (Hohmann altitude raise)
    if len(epochs) == 0:
        print("No epochs selected; exiting.")
        return
    ref = epochs[len(epochs)//2]
    sat_ref = EarthSatellite(ref.line1, ref.line2)
    a_km, e, i_deg, raan_deg = elements_from_satellite(sat_ref)

    # Treat reference orbit as circular at altitude h ≈ a - R_E (ok for small e)
    h_ref_km = max(0.0, a_km - R_EARTH_KM)
    r1_km = R_EARTH_KM + h_ref_km
    r2_km = R_EARTH_KM + (h_ref_km + TARGET_ALTITUDE_DELTA_KM)

    dv1, dv2, tof_s = hohmann_delta_v(r1_km, r2_km)
    dv_total = dv1 + dv2

    m_prop, m_final = tsiolkovsky(M0_KG, dv_total, ISP_S, g0=G0)

    plan_txt = textwrap.dedent(f"""
    === Maneuver Plan (Reference Epoch) ===
    NORAD: {ref.norad}
    Reference TLE epoch (UTC): {ref.epoch_dt.isoformat()}

    Assumed circular altitude at reference: ~{h_ref_km:.1f} km
    Target circular altitude: ~{h_ref_km + TARGET_ALTITUDE_DELTA_KM:.1f} km

    Burn 1 (inject to transfer): dv1 = {dv1:7.2f} m/s at {ref.epoch_dt.isoformat()} UTC
    Burn 2 (circularize):       dv2 = {dv2:7.2f} m/s at {(ref.epoch_dt + timedelta(seconds=tof_s)).isoformat()} UTC
    Transfer time:               {tof_s/60.0:7.2f} minutes

    ΔV total: {dv_total:.2f} m/s

    Fuel estimate (Tsiolkovsky):
      m0 = {M0_KG:.3f} kg, Isp = {ISP_S:.1f} s, g0 = {G0:.5f} m/s²
      propellant ≈ {m_prop:.4f} kg, final mass ≈ {m_final:.4f} kg
    """)

    with open(os.path.join(SUM_DIR, "maneuver_plan.txt"), "w", encoding="utf-8") as f:
        f.write(plan_txt)

    print("Done. Outputs:")
    print(f" - {out_csv}")
    print(f" - {os.path.join(SUM_DIR, 'maneuver_plan.txt')}")
    print(f" - {FIG_DIR}/*.png")


if __name__ == "__main__":
    main()
