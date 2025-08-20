# LEO CubeSat Orbit Analysis (Sept 2024 – Feb 2025)

This project propagates real LEO satellite orbits from historical TLEs and produces:
- 24-hour **ground tracks** (PNG)
- **Altitude vs time** plots (PNG)
- A **weekly elements summary** (CSV, with RAAN drift)
- A **Hohmann altitude-raise** plan (+ΔV and fuel estimate)

## How to run

```bash
# 1) Create & activate venv (optional if you already have one)
python -m venv .venv
.\.venv\Scripts\activate.bat

# 2) Install deps
pip install numpy pandas matplotlib skyfield sgp4 requests

# 3) Set a NORAD ID inside the script (e.g., ISS 25544), then run:
python leo_historical_tle_nopoliastro.py

Roadmap
 - Ground-station pass prediction (Bengaluru)
 - Multi-satellite comparison
 - Animated ground track
