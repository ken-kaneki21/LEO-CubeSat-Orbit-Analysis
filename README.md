# LEO CubeSat Orbit Analysis (Sept 2024 – Feb 2025)

This project propagates real LEO satellite orbits from historical TLEs and produces:
- 24-hour **ground tracks** (PNG)
- **Altitude vs time** plots (PNG)
- A **weekly orbital elements summary** (CSV, with RAAN drift)
- A **Hohmann altitude-raise** plan (+ΔV and fuel estimate)
- **Ground-station pass prediction (Bengaluru)** ≥ 10° elevation (CSV)
- **24-hour ground-track animation GIF** at mid-window epoch

## How to run (Windows, cmd)

```bash
# 1) Create & activate venv (optional if already done)
python -m venv .venv
.\.venv\Scripts\activate.bat

# 2) Install dependencies
pip install numpy pandas matplotlib skyfield sgp4 requests pillow

# 3) (Option A) Use Space-Track for historical TLEs (recommended)
set SPACETRACK_USERNAME=your_spacetrack_email
set SPACETRACK_PASSWORD=your_spacetrack_password

#    (Option B) Or provide a local TLE file
#    Put many two-line TLEs covering 2024-09-01 → 2025-02-28 in: tle_history.txt

# 4) Edit the script config:
#    - Set NORAD_IDS = [ ... ]   (one or many)
#    - (Optional) SAMPLE_EVERY_DAYS = 1 for daily samples

# 5) Run
python leo_historical_tle_nopoliastro.py

Outputs

figures/ground_track_<NORAD>_<DATE>.png

figures/altitude_time_<NORAD>_<DATE>.png

figures/animation_<NORAD>_<DATE>.gif

summary/orbit_weekly_summary.csv

summary/passes_bengaluru.csv

summary/maneuver_plan_<NORAD>.txt

Configure satellites

In leo_historical_tle_nopoliastro.py, set:
NORAD_IDS = [25544]  # Example: ISS; you can add more IDs here

Good multi-sat examples:

ISS — 25544

Hubble — 20580

Terra — 25994

Aqua — 27424

Landsat-8 — 39084

Landsat-9 — 49260

Sentinel-2A — 40697

NOAA-19 — 33591

(Start with 2–3, confirm outputs, then add more.)

Roadmap

 Publish a short technical report with figures