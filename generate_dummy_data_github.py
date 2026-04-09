"""
Brand Proxy POC — Dummy Data Generator
Generates 15 realistic .xlsx files with 60 months of correlated synthetic data.
Uses Cholesky decomposition for inter-signal correlations, with layered
trend, seasonality, and event shocks.

Random seed: 42 for full reproducibility.
Time window: March 2021 – February 2026 (60 months)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dummy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = pd.date_range("2021-03-01", "2026-02-01", freq="MS")
N_MONTHS = len(MONTHS)  # 60

# ─── Wave Schedule (13 waves) ────────────────────────────────────────────────

WAVES = [
    {"wave_id": "Q2_2021",  "field_start": "2021-06-01", "field_end": "2021-07-15", "midpoint": "2021-06-01"},
    {"wave_id": "Q4_2021",  "field_start": "2021-12-01", "field_end": "2022-01-15", "midpoint": "2021-12-01"},
    {"wave_id": "Q2_2022",  "field_start": "2022-06-01", "field_end": "2022-07-15", "midpoint": "2022-06-01"},
    {"wave_id": "Q4_2022",  "field_start": "2022-11-01", "field_end": "2022-12-15", "midpoint": "2022-11-01"},
    {"wave_id": "Q1_2023",  "field_start": "2023-02-01", "field_end": "2023-04-15", "midpoint": "2023-03-01"},
    {"wave_id": "Q2_2023",  "field_start": "2023-06-01", "field_end": "2023-06-30", "midpoint": "2023-06-01"},
    {"wave_id": "Q3_2023",  "field_start": "2023-08-01", "field_end": "2023-09-15", "midpoint": "2023-09-01"},
    {"wave_id": "Q4_2023",  "field_start": "2023-11-01", "field_end": "2023-12-15", "midpoint": "2023-12-01"},
    {"wave_id": "Q1_2024",  "field_start": "2024-02-01", "field_end": "2024-03-15", "midpoint": "2024-03-01"},
    {"wave_id": "Q2_2024",  "field_start": "2024-05-01", "field_end": "2024-06-15", "midpoint": "2024-05-01"},
    {"wave_id": "Q4_2024",  "field_start": "2024-11-01", "field_end": "2024-12-15", "midpoint": "2024-11-01"},
    {"wave_id": "Q2_2025",  "field_start": "2025-05-01", "field_end": "2025-06-15", "midpoint": "2025-06-01"},
    {"wave_id": "Q4_2025",  "field_start": "2025-11-01", "field_end": "2025-12-15", "midpoint": "2025-11-01"},
]

# ─── Event Registry ──────────────────────────────────────────────────────────

EVENTS = [
    {"date": "2021-03-15", "end": "2021-03-15", "type": "Product Launch", "name": "Gen 11 Processor Launch", "constructs": "Awareness,Consideration", "desc": "Desktop processor launch competing with rival brand"},
    {"date": "2021-06-01", "end": "2021-06-03", "type": "Industry Event", "name": "Tech Expo 2021 — Next Gen Preview", "constructs": "Awareness", "desc": "First preview of next gen hybrid architecture"},
    {"date": "2021-09-09", "end": "2021-09-10", "type": "Industry Event", "name": "Brand Innovation 2021 — CEO Keynote", "constructs": "Awareness,Consideration", "desc": "CEO keynote, new strategic vision announced"},
    {"date": "2021-11-04", "end": "2021-11-04", "type": "Product Launch", "name": "Gen 12 Desktop Launch", "constructs": "Awareness,Consideration,Purchase Intent", "desc": "Hybrid architecture launch, strong reviews"},
    {"date": "2022-01-04", "end": "2022-01-08", "type": "Industry Event", "name": "CES 2022 — Brand Keynote", "constructs": "Awareness", "desc": "CES keynote showcasing new mobile and GPU products"},
    {"date": "2022-02-23", "end": "2022-02-23", "type": "Leadership Change", "name": "Foundry Services Expansion Announced", "constructs": "Awareness", "desc": "Major foundry investment announcements"},
    {"date": "2022-06-18", "end": "2022-06-18", "type": "Competitive Event", "name": "Competitor Chip Launch", "constructs": "Consideration,Purchase Intent", "desc": "Competitor chip launch raises competitive pressure"},
    {"date": "2022-09-28", "end": "2022-09-28", "type": "Product Launch", "name": "Gen 13 Processor Launch", "constructs": "Awareness,Consideration,Purchase Intent", "desc": "Strong performance uplift, positive reviews"},
    {"date": "2023-01-03", "end": "2023-01-06", "type": "Industry Event", "name": "CES 2023 — Next Gen Announcement", "constructs": "Awareness", "desc": "Next gen preview and AI acceleration messaging"},
    {"date": "2023-05-30", "end": "2023-06-02", "type": "Industry Event", "name": "Tech Expo 2023 — Architecture Deep Dive", "constructs": "Awareness,Consideration", "desc": "Detailed next gen architecture reveal"},
    {"date": "2023-09-19", "end": "2023-09-20", "type": "Industry Event", "name": "Brand Innovation 2023 — AI Chip Reveal", "constructs": "Awareness,Consideration", "desc": "New brand and AI PC category creation"},
    {"date": "2023-10-30", "end": "2023-10-30", "type": "Competitive Event", "name": "Competitor Next Gen Launch", "constructs": "Consideration,Purchase Intent", "desc": "Competitor raises the bar again"},
    {"date": "2023-12-14", "end": "2023-12-14", "type": "Product Launch", "name": "AI Chip Launch", "constructs": "Awareness,Consideration,Purchase Intent", "desc": "AI PC chip launch, major brand moment"},
    {"date": "2024-01-09", "end": "2024-01-12", "type": "Industry Event", "name": "CES 2024 — AI PC Momentum", "constructs": "Awareness", "desc": "Major CES presence, AI PC partnerships announced"},
    {"date": "2024-04-09", "end": "2024-04-09", "type": "Competitive Event", "name": "Rival Next Gen Announcement", "constructs": "Consideration,Purchase Intent", "desc": "Rival next-gen announcement increases competitive pressure"},
    {"date": "2024-06-04", "end": "2024-06-04", "type": "Industry Event", "name": "Tech Expo 2024 — Next Gen Reveal", "constructs": "Awareness,Consideration", "desc": "Next gen architecture detailed at expo"},
    {"date": "2024-09-03", "end": "2024-09-03", "type": "Product Launch", "name": "Next Gen AI Chip Launch", "constructs": "Awareness,Consideration,Purchase Intent", "desc": "Next-gen AI PC chip, efficiency focus"},
    {"date": "2024-10-24", "end": "2024-10-24", "type": "Competitive Event", "name": "ARM-based PC Launch", "constructs": "Consideration,Purchase Intent", "desc": "ARM-based Windows PCs create new competitive category"},
    {"date": "2024-12-03", "end": "2024-12-03", "type": "Leadership Change", "name": "CEO Transition", "constructs": "Awareness,Consideration", "desc": "CEO transition creates uncertainty"},
    {"date": "2025-01-06", "end": "2025-01-10", "type": "Industry Event", "name": "CES 2025 — Roadmap Preview", "constructs": "Awareness", "desc": "CES keynote with next-gen roadmap"},
    {"date": "2025-03-12", "end": "2025-03-12", "type": "Leadership Change", "name": "New CEO Appointment", "constructs": "Awareness,Consideration", "desc": "New CEO named, stability signal"},
    {"date": "2025-04-15", "end": "2025-04-15", "type": "Restructuring", "name": "Business Restructuring", "constructs": "Awareness", "desc": "Business structural changes announced"},
    {"date": "2025-06-03", "end": "2025-06-03", "type": "Industry Event", "name": "Tech Expo 2025 — Next Gen Details", "constructs": "Awareness,Consideration", "desc": "Next gen architecture and partner showcase"},
    {"date": "2025-07-15", "end": "2025-07-15", "type": "Competitive Event", "name": "Rival Processor Launch", "constructs": "Consideration,Purchase Intent", "desc": "Rival launch increases competitive pressure"},
    {"date": "2025-09-16", "end": "2025-09-17", "type": "Industry Event", "name": "Brand Innovation 2025", "constructs": "Awareness,Consideration", "desc": "Annual developer event"},
    {"date": "2025-10-15", "end": "2025-10-15", "type": "Product Launch", "name": "Next Gen Consumer Processor Launch", "constructs": "Awareness,Consideration,Purchase Intent", "desc": "Next-gen consumer processor launch"},
    {"date": "2025-11-01", "end": "2025-11-01", "type": "Competitive Event", "name": "Competitor Chip Announcement", "constructs": "Consideration,Purchase Intent", "desc": "Competitor next-gen silicon raises competitive bar"},
    {"date": "2026-01-06", "end": "2026-01-09", "type": "Industry Event", "name": "CES 2026 — Future Roadmap Preview", "constructs": "Awareness", "desc": "CES keynote with next-gen roadmap and AI strategy"},
    {"date": "2021-04-22", "end": "2021-04-22", "type": "Earnings", "name": "Q1 2021 Earnings", "constructs": "Awareness", "desc": "First earnings under new CEO"},
    {"date": "2022-04-28", "end": "2022-04-28", "type": "Earnings", "name": "Q1 2022 Earnings — Revenue Miss", "constructs": "Awareness,Consideration", "desc": "Revenue miss, negative sentiment spike"},
    {"date": "2022-10-27", "end": "2022-10-27", "type": "Earnings", "name": "Q3 2022 Earnings — Significant Miss", "constructs": "Awareness,Consideration", "desc": "Major revenue miss, share price decline"},
    {"date": "2023-01-26", "end": "2023-01-26", "type": "Earnings", "name": "Q4 2022 Earnings — Recovery Signs", "constructs": "Awareness", "desc": "Slight recovery, cost-cutting measures positive"},
    {"date": "2024-01-25", "end": "2024-01-25", "type": "Earnings", "name": "Q4 2023 Earnings — Beat Expectations", "constructs": "Awareness,Consideration", "desc": "Better than expected results, sentiment positive"},
]

# Campaign Calendar
CAMPAIGNS = [
    {"id": "CU_2021_Q4", "name": "Gen 12 Consumer Launch", "start": "2021-11-01", "end": "2022-01-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search", "tier": "High", "spend": 14000000},
    {"id": "EVO_2022_Q1", "name": "Platform Spring Campaign", "start": "2022-02-01", "end": "2022-04-30", "objective": "Consideration", "audience": "Consumer", "geo": "US", "channels": "Digital,Social", "tier": "Medium", "spend": 5500000},
    {"id": "BTS_2022", "name": "Back-to-School 2022", "start": "2022-07-15", "end": "2022-09-15", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 6000000},
    {"id": "RL_2022_Q4", "name": "Gen 13 Launch Campaign", "start": "2022-09-15", "end": "2022-12-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search,Video", "tier": "High", "spend": 16000000},
    {"id": "HOL_2022", "name": "Holiday Season 2022", "start": "2022-11-01", "end": "2022-12-31", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search", "tier": "High", "spend": 12000000},
    {"id": "CES_2023", "name": "CES 2023 Presence", "start": "2023-01-01", "end": "2023-01-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Video", "tier": "Medium", "spend": 4000000},
    {"id": "EVO_2023_Q1", "name": "Platform Campaign 2023", "start": "2023-02-01", "end": "2023-04-30", "objective": "Consideration", "audience": "Consumer", "geo": "US", "channels": "Digital,Social", "tier": "Medium", "spend": 5000000},
    {"id": "BTS_2023", "name": "Back-to-School 2023", "start": "2023-07-15", "end": "2023-09-15", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 6500000},
    {"id": "AIPC_2023_Q4", "name": "AI PC Category Creation", "start": "2023-10-01", "end": "2024-01-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search,Video", "tier": "High", "spend": 18000000},
    {"id": "CU_2023_Q4", "name": "AI Chip Launch Campaign", "start": "2023-12-01", "end": "2024-02-28", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search,Video", "tier": "High", "spend": 15000000},
    {"id": "CES_2024", "name": "CES 2024 AI PC Push", "start": "2024-01-01", "end": "2024-01-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Video", "tier": "Medium", "spend": 5000000},
    {"id": "SPRING_2024", "name": "AI PC Spring Campaign", "start": "2024-03-01", "end": "2024-05-31", "objective": "Consideration", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 7000000},
    {"id": "BTS_2024", "name": "Back-to-School AI PC", "start": "2024-07-15", "end": "2024-09-15", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 7500000},
    {"id": "LL_2024_Q3", "name": "Next Gen Launch Campaign", "start": "2024-09-01", "end": "2024-11-30", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search,Video", "tier": "High", "spend": 14000000},
    {"id": "HOL_2024", "name": "Holiday Season 2024", "start": "2024-11-01", "end": "2024-12-31", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search", "tier": "High", "spend": 13000000},
    {"id": "CES_2025", "name": "CES 2025 Presence", "start": "2025-01-01", "end": "2025-01-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Video", "tier": "Medium", "spend": 4500000},
    {"id": "SPRING_2025", "name": "Spring 2025 Brand", "start": "2025-03-01", "end": "2025-05-31", "objective": "Consideration", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 6000000},
    {"id": "BTS_2025", "name": "Back-to-School 2025", "start": "2025-07-15", "end": "2025-09-15", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Search", "tier": "Medium", "spend": 7000000},
    {"id": "PL_2025_Q4", "name": "Next Gen Launch Campaign Q4", "start": "2025-10-01", "end": "2025-12-31", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search,Video", "tier": "High", "spend": 16000000},
    {"id": "HOL_2025", "name": "Holiday Season 2025", "start": "2025-11-01", "end": "2025-12-31", "objective": "Purchase Intent", "audience": "Consumer", "geo": "US", "channels": "TV,Digital,Social,Search", "tier": "High", "spend": 14000000},
    {"id": "CES_2026", "name": "CES 2026 Future Preview", "start": "2026-01-01", "end": "2026-02-28", "objective": "Awareness", "audience": "Consumer", "geo": "US", "channels": "Digital,Social,Video", "tier": "Medium", "spend": 5000000},
]


# ─── Helper Functions ─────────────────────────────────────────────────────────

def month_index(date_str):
    """Convert date string to index in MONTHS array (0-based)."""
    dt = pd.Timestamp(date_str)
    target = pd.Timestamp(dt.year, dt.month, 1)
    idx = MONTHS.get_loc(target) if target in MONTHS else -1
    return idx


def trend_component(n_months=60):
    """Generate the narrative-arc trend for brand signals.
    Returns values centered around 0, range roughly -0.3 to +0.3.
    """
    t = np.linspace(0, 1, n_months)
    trend = np.zeros(n_months)
    for i in range(n_months):
        m = i / (n_months - 1)
        if m < 0.15:
            trend[i] = -0.05 - 0.10 * (m / 0.15)
        elif m < 0.30:
            trend[i] = -0.15 + 0.10 * ((m - 0.15) / 0.15)
        elif m < 0.40:
            trend[i] = -0.05 - 0.08 * ((m - 0.30) / 0.10)
        elif m < 0.55:
            trend[i] = -0.13 + 0.25 * ((m - 0.40) / 0.15)
        elif m < 0.70:
            trend[i] = 0.12 + 0.15 * ((m - 0.55) / 0.15)
        elif m < 0.82:
            trend[i] = 0.27 - 0.05 * ((m - 0.70) / 0.12)
        elif m < 0.92:
            trend[i] = 0.22 - 0.02 * ((m - 0.82) / 0.10)
        else:
            trend[i] = 0.20 + 0.08 * ((m - 0.92) / 0.08)
    return trend


def seasonality_component(n_months=60, ces_boost=0.12, bts_boost=0.06, holiday_boost=0.08):
    """Monthly seasonality: CES January spike, Back-to-School Aug-Sep, Holiday Nov-Dec."""
    seasonal = np.zeros(n_months)
    for i in range(n_months):
        month_num = MONTHS[i].month
        if month_num == 1:
            seasonal[i] = ces_boost
        elif month_num == 2:
            seasonal[i] = ces_boost * 0.3
        elif month_num == 8:
            seasonal[i] = bts_boost * 0.7
        elif month_num == 9:
            seasonal[i] = bts_boost
        elif month_num == 10:
            seasonal[i] = 0.02
        elif month_num == 11:
            seasonal[i] = holiday_boost * 0.8
        elif month_num == 12:
            seasonal[i] = holiday_boost
        elif month_num == 6:
            seasonal[i] = 0.03
        else:
            seasonal[i] = 0.0
    return seasonal


def event_shocks(n_months=60, events=EVENTS, base_magnitude=0.08, decay=0.5):
    """Generate event-driven shocks. Returns dict of {construct: array}."""
    shocks = {
        "Awareness": np.zeros(n_months),
        "Consideration": np.zeros(n_months),
        "Purchase Intent": np.zeros(n_months),
    }
    type_magnitudes = {
        "Product Launch": 1.2,
        "Industry Event": 0.8,
        "Competitive Event": -0.6,
        "Leadership Change": 0.4,
        "Earnings": 0.3,
        "Restructuring": 0.2,
        "Campaign Launch": 0.5,
    }
    for evt in events:
        idx = month_index(evt["date"])
        if idx < 0 or idx >= n_months:
            continue
        constructs = [c.strip() for c in evt["constructs"].split(",")]
        mag = base_magnitude * type_magnitudes.get(evt["type"], 0.5)
        if evt["type"] == "Competitive Event":
            mag = -abs(mag)
        for c in constructs:
            if c in shocks:
                shocks[c][idx] += mag
                if idx + 1 < n_months:
                    shocks[c][idx + 1] += mag * decay
                if idx + 2 < n_months:
                    shocks[c][idx + 2] += mag * decay * 0.3
    return shocks


def generate_correlated_noise(n_months=60):
    """Generate 8 correlated noise series using Cholesky decomposition.
    Order: Search, Social, GTrends, Web, Sales, Awareness, Consideration, PI
    """
    corr_matrix = np.array([
        [1.00, 0.65, 0.85, 0.72, 0.45, 0.75, 0.55, 0.50],
        [0.65, 1.00, 0.60, 0.55, 0.30, 0.60, 0.50, 0.35],
        [0.85, 0.60, 1.00, 0.68, 0.40, 0.72, 0.48, 0.45],
        [0.72, 0.55, 0.68, 1.00, 0.50, 0.70, 0.52, 0.48],
        [0.45, 0.30, 0.40, 0.50, 1.00, 0.40, 0.45, 0.65],
        [0.75, 0.60, 0.72, 0.70, 0.40, 1.00, 0.70, 0.55],
        [0.55, 0.50, 0.48, 0.52, 0.45, 0.70, 1.00, 0.72],
        [0.50, 0.35, 0.45, 0.48, 0.65, 0.55, 0.72, 1.00],
    ])

    L = np.linalg.cholesky(corr_matrix)
    independent = np.random.randn(8, n_months)
    correlated = L @ independent
    return correlated


def scale_to_range(series, lo, hi):
    """Scale a series to target range [lo, hi]."""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-10:
        return np.full_like(series, (lo + hi) / 2)
    return lo + (series - s_min) / (s_max - s_min) * (hi - lo)


def add_multiplicative_effects(base, trend, seasonal, shock, noise_scale=0.03):
    """Combine base level with multiplicative trend, seasonality, and shocks."""
    return base * (1 + trend + seasonal + shock + np.random.randn(len(base)) * noise_scale)


# ─── Main Generation Logic ───────────────────────────────────────────────────

def generate_all():
    print("=" * 70)
    print("Brand Proxy POC — Dummy Data Generator")
    print("=" * 70)

    corr_noise = generate_correlated_noise(N_MONTHS)
    noise_search = corr_noise[0]
    noise_social = corr_noise[1]
    noise_gtrends = corr_noise[2]
    noise_web = corr_noise[3]
    noise_sales = corr_noise[4]
    noise_aware = corr_noise[5]
    noise_consid = corr_noise[6]
    noise_pi = corr_noise[7]

    trend = trend_component(N_MONTHS)
    seasonal = seasonality_component(N_MONTHS)
    shocks = event_shocks(N_MONTHS)

    # ─── PRE-COMPUTE ─────────────────────────────────────────────────────────

    _branded_kw = {"BrandX": 750000, "BrandX processor": 180000,
                   "BrandX CPU": 120000, "BrandX Core": 95000}
    _branded_search_monthly = np.zeros(N_MONTHS)
    for i in range(N_MONTHS):
        for kw, base_vol in _branded_kw.items():
            t = trend[i]
            s = seasonal[i]
            shock_val = shocks["Awareness"][i] * 1.5
            n = noise_search[i]
            vol = base_vol * (1 + t * 0.35 + s * 2 + shock_val + n * 0.06)
            vol = max(int(base_vol * 0.5), int(vol))
            _branded_search_monthly[i] += vol

    _sentiment_monthly = np.zeros(N_MONTHS)
    for i in range(N_MONTHS):
        net_sent = 22 + trend[i] * 20 + shocks["Consideration"][i] * 15 + noise_consid[i] * 2
        _sentiment_monthly[i] = np.clip(net_sent, 15, 35)

    _sellthrough_monthly = np.zeros(N_MONTHS)
    base_units = 2300000
    for i in range(N_MONTHS):
        month_num = MONTHS[i].month
        holiday_mult = 1.0
        if month_num == 11: holiday_mult = 1.30
        elif month_num == 12: holiday_mult = 1.40
        elif month_num in [1, 2]: holiday_mult = 0.80
        units = base_units * holiday_mult * (1 + trend[i] * 0.25 +
                                              shocks["Purchase Intent"][i] * 1.0 +
                                              noise_sales[i] * 0.06)
        _sellthrough_monthly[i] = max(1500000, units)

    # ─── FILE 1: Brand Tracker Scores ─────────────────────────────────────

    print("\n[1/15] Generating brand_tracker_scores.xlsx ...")

    wave_indices = np.array([month_index(w["midpoint"]) for w in WAVES])

    search_at_waves = _branded_search_monthly[wave_indices]
    search_z = (search_at_waves - search_at_waves.mean()) / search_at_waves.std()
    aware_signal = search_z * 0.70 + noise_aware[wave_indices] * 0.30
    brandx_aware_base = scale_to_range(aware_signal, 84.9, 87.8)

    sent_at_waves = _sentiment_monthly[wave_indices]
    sent_z = (sent_at_waves - sent_at_waves.mean()) / (sent_at_waves.std() + 1e-8)
    consid_signal = sent_z * 0.60 + noise_consid[wave_indices] * 0.25 + trend[wave_indices] * 0.5
    brandx_consid_base = scale_to_range(consid_signal, 47.3, 51.2)

    sell_at_waves = _sellthrough_monthly[wave_indices]
    sell_z = (sell_at_waves - sell_at_waves.mean()) / (sell_at_waves.std() + 1e-8)
    pi_signal = sell_z * 0.55 + noise_pi[wave_indices] * 0.25 + trend[wave_indices] * 0.4
    brandx_pi_base = scale_to_range(pi_signal, 35.0, 38.1)

    competitor_ranges = {
        "BrandA":  {"Awareness": (71, 76), "Consideration": (42, 48)},
        "BrandB":  {"Awareness": (66, 74), "Consideration": (38, 45)},
        "BrandC":  {"Awareness": (45, 55), "Consideration": (28, 36)},
        "BrandD":  {"Awareness": (78, 85), "Consideration": (55, 62)},
        "BrandE":  {"Awareness": (80, 86), "Consideration": (48, 55)},
        "BrandF":  {"Awareness": (75, 82), "Consideration": (40, 48)},
        "BrandG":  {"Awareness": (25, 35), "Consideration": (12, 20)},
        "BrandH":  {"Awareness": (30, 40), "Consideration": (15, 25)},
    }

    pi_ranges = {
        "BrandA (Processor)":  (30, 36),
        "BrandD (Chip)":       (25, 33),
        "BrandC (Mobile)":     (8, 18),
    }

    tracker_rows = []
    for i, wave in enumerate(WAVES):
        sample_size = np.random.randint(1100, 1301)
        margin_of_error = round(np.random.uniform(2.5, 3.0), 1)
        base_row = {
            "wave_id": wave["wave_id"],
            "field_start_date": wave["field_start"],
            "field_end_date": wave["field_end"],
            "audience": "Overall Consumer",
            "sample_size": sample_size,
            "margin_of_error": margin_of_error,
        }

        tracker_rows.append({**base_row, "construct": "Awareness", "brand": "BrandX",
                             "score": round(brandx_aware_base[i], 1)})
        tracker_rows.append({**base_row, "construct": "Consideration", "brand": "BrandX",
                             "score": round(brandx_consid_base[i], 1)})
        tracker_rows.append({**base_row, "construct": "Purchase Intent", "brand": "BrandX",
                             "score": round(brandx_pi_base[i], 1)})

        for brand, ranges in competitor_ranges.items():
            progress = i / (len(WAVES) - 1)
            for construct in ["Awareness", "Consideration"]:
                lo, hi = ranges[construct]
                if brand in ["BrandA", "BrandB"]:
                    score = lo + (hi - lo) * (0.3 + 0.6 * progress + np.random.randn() * 0.08)
                elif brand in ["BrandC"]:
                    score = lo + (hi - lo) * (0.2 + 0.5 * progress + np.random.randn() * 0.08)
                else:
                    score = lo + (hi - lo) * (0.4 + np.random.randn() * 0.15)
                score = np.clip(score, lo, hi)
                tracker_rows.append({**base_row, "construct": construct, "brand": brand,
                                     "score": round(score, 1)})

        for brand, (lo, hi) in pi_ranges.items():
            progress = i / (len(WAVES) - 1)
            if "BrandD" in brand or "BrandC" in brand:
                score = lo + (hi - lo) * (0.3 + 0.5 * progress + np.random.randn() * 0.08)
            else:
                score = lo + (hi - lo) * (0.4 + 0.3 * progress + np.random.randn() * 0.1)
            score = np.clip(score, lo, hi)
            tracker_rows.append({**base_row, "construct": "Purchase Intent", "brand": brand,
                                 "score": round(score, 1)})

    df_tracker = pd.DataFrame(tracker_rows)
    df_tracker.to_excel(OUTPUT_DIR / "brand_tracker_scores.xlsx", index=False)
    print(f"  -> {len(df_tracker)} rows written")

    # ─── FILE 2: Social Mention Volume ────────────────────────────────────

    print("[2/15] Generating social_mention_volume.xlsx ...")

    base_mentions = 140000
    total_mentions = np.zeros(N_MONTHS)
    for i in range(N_MONTHS):
        m = base_mentions * (1
            + trend[i] * 0.4
            + seasonal[i] * 2.5
            + shocks["Awareness"][i] * 2
            + noise_social[i] * 0.08)
        total_mentions[i] = max(80000, m)

    platform_ratios = {
        "twitter_x": 0.36, "reddit": 0.19, "facebook": 0.12,
        "youtube": 0.09, "linkedin": 0.06, "forums": 0.07, "news_blogs": 0.11
    }
    social_rows = []
    for i in range(N_MONTHS):
        total = int(total_mentions[i])
        original = int(total * np.random.uniform(0.65, 0.72))
        row = {"month": MONTHS[i].strftime("%Y-%m-%d"), "brand": "BrandX",
               "total_mentions": total, "original_mentions": original}
        remaining = total
        for j, (plat, ratio) in enumerate(platform_ratios.items()):
            if j == len(platform_ratios) - 1:
                row[f"{plat}_mentions"] = remaining
            else:
                val = int(total * ratio * np.random.uniform(0.90, 1.10))
                row[f"{plat}_mentions"] = val
                remaining -= val
        social_rows.append(row)

    df_social = pd.DataFrame(social_rows)
    df_social.loc[23, "forums_mentions"] = None
    df_social.to_excel(OUTPUT_DIR / "social_mention_volume.xlsx", index=False)
    print(f"  -> {len(df_social)} rows written")

    # ─── FILE 3: Social Sentiment ─────────────────────────────────────────

    print("[3/15] Generating social_sentiment.xlsx ...")

    sentiment_rows = []
    for i in range(N_MONTHS):
        total = int(total_mentions[i])
        net_sent = 22 + trend[i] * 20 + shocks["Consideration"][i] * 15 + noise_consid[i] * 2
        net_sent = np.clip(net_sent, 15, 35)

        positive_ratio = (50 + net_sent) / 200
        negative_ratio = positive_ratio - net_sent / 100
        neutral_ratio = 1 - positive_ratio - negative_ratio
        mixed_pct = np.random.uniform(0.03, 0.06)

        positive = int(total * positive_ratio)
        negative = int(total * negative_ratio)
        neutral = int(total * (neutral_ratio - mixed_pct))
        mixed = total - positive - negative - neutral

        total_engagement = int(total * np.random.uniform(2.5, 4.0))
        likes = int(total_engagement * np.random.uniform(0.55, 0.65))
        shares = int(total_engagement * np.random.uniform(0.10, 0.18))
        comments = total_engagement - likes - shares

        sentiment_rows.append({
            "month": MONTHS[i].strftime("%Y-%m-%d"),
            "brand": "BrandX",
            "positive_mentions": positive,
            "negative_mentions": negative,
            "neutral_mentions": neutral,
            "mixed_mentions": mixed,
            "net_sentiment_score": round(net_sent, 2),
            "total_engagement": total_engagement,
            "likes": likes,
            "comments": comments,
            "shares": shares,
        })

    df_sentiment = pd.DataFrame(sentiment_rows)
    df_sentiment.to_excel(OUTPUT_DIR / "social_sentiment.xlsx", index=False)
    print(f"  -> {len(df_sentiment)} rows written")

    # ─── FILE 4: Social Purchase Language ─────────────────────────────────

    print("[4/15] Generating social_purchase_language.xlsx ...")

    purchase_rows = []
    for i in range(N_MONTHS):
        base_purch = 3500
        pm = base_purch * (1 + trend[i] * 0.5 + seasonal[i] * 1.5 +
                           shocks["Purchase Intent"][i] * 2 + noise_pi[i] * 0.10)
        pm = max(2000, int(pm))
        rm = int(pm * np.random.uniform(0.60, 0.80))
        purchase_rows.append({
            "month": MONTHS[i].strftime("%Y-%m-%d"),
            "purchase_mentions": pm,
            "recommendation_mentions": rm,
            "advocacy_mentions": pm + rm,
        })

    df_purchase = pd.DataFrame(purchase_rows)
    df_purchase.to_excel(OUTPUT_DIR / "social_purchase_language.xlsx", index=False)
    print(f"  -> {len(df_purchase)} rows written")

    # ─── FILE 5: Search Volume by Keyword ─────────────────────────────────

    print("[5/15] Generating search_volume_by_keyword.xlsx ...")

    keywords = {
        "branded_awareness": [
            ("BrandX", 750000), ("BrandX processor", 180000), ("BrandX CPU", 120000),
            ("BrandX Core", 95000), ("BrandX laptop", 65000),
            ("BrandA", 580000), ("BrandA processor", 220000), ("BrandB", 450000),
            ("BrandC", 85000), ("BrandD chip", 110000),
        ],
        "category_generic": [
            ("CPU", 320000), ("processor", 180000), ("laptop processor", 55000),
            ("best CPU", 42000), ("desktop CPU", 38000),
        ],
        "comparative_consideration": [
            ("BrandX vs BrandA", 68000), ("BrandX or BrandA", 32000), ("BrandX review", 28000),
            ("BrandX benchmark", 22000), ("BrandX vs BrandD", 15000),
        ],
        "purchase_intent": [
            ("buy BrandX", 12000), ("BrandX price", 18000), ("BrandX deals", 8000),
            ("BrandX laptop deals", 9500), ("where to buy BrandX", 5000),
        ],
        "configuration_spec": [
            ("BrandX Core i7 specs", 14000), ("BrandX Core i9 specs", 11000),
            ("BrandX configurator", 8000),
        ],
    }

    search_rows = []
    for i in range(N_MONTHS):
        for group, kw_list in keywords.items():
            for kw, base_vol in kw_list:
                if "BrandX" in kw:
                    n = noise_search[i]
                    t = trend[i]
                    s = seasonal[i]
                elif "BrandA" in kw:
                    n = np.random.randn() * 0.08
                    t = trend[i] * 0.5 + 0.1 * (i / N_MONTHS)
                    s = seasonal[i] * 0.8
                elif "BrandB" in kw:
                    n = np.random.randn() * 0.08
                    t = 0.15 * (i / N_MONTHS)
                    s = seasonal[i] * 0.7
                else:
                    n = np.random.randn() * 0.06
                    t = trend[i] * 0.3
                    s = seasonal[i] * 0.5

                shock_val = 0
                if group == "branded_awareness":
                    shock_val = shocks["Awareness"][i] * 1.5
                elif group == "comparative_consideration":
                    shock_val = shocks["Consideration"][i] * 1.2
                elif group == "purchase_intent":
                    shock_val = shocks["Purchase Intent"][i] * 1.5

                vol = base_vol * (1 + t * 0.35 + s * 2 + shock_val + n * 0.06)
                vol = max(int(vol * 0.5), int(vol))

                search_rows.append({
                    "month": MONTHS[i].strftime("%Y-%m-%d"),
                    "keyword": kw,
                    "keyword_group": group,
                    "search_volume": int(vol),
                })

    df_search = pd.DataFrame(search_rows)
    df_search.to_excel(OUTPUT_DIR / "search_volume_by_keyword.xlsx", index=False)
    print(f"  -> {len(df_search)} rows written")

    # ─── FILE 6: Google Trends ────────────────────────────────────────────

    print("[6/15] Generating google_trends.xlsx ...")

    gtrends_queries = [
        ("BrandX", "brand_comparison"),
        ("BrandA", "brand_comparison"),
        ("BrandB", "brand_comparison"),
        ("BrandC", "brand_comparison"),
        ("BrandX processor", "product"),
        ("buy BrandX", "purchase"),
        ("BrandX vs BrandA", "comparison"),
        ("CPU", "category"),
    ]

    gtrends_rows = []
    for i in range(N_MONTHS):
        brandx_interest = 55 + trend[i] * 30 + seasonal[i] * 40 + shocks["Awareness"][i] * 20 + noise_gtrends[i] * 3
        brandx_interest = np.clip(brandx_interest, 25, 100)

        branda_interest = 42 + 8 * (i / N_MONTHS) + np.random.randn() * 3 + seasonal[i] * 25
        brandb_interest = 48 + 12 * (i / N_MONTHS) + np.random.randn() * 4 + seasonal[i] * 20
        brandc_interest = 15 + 5 * (i / N_MONTHS) + np.random.randn() * 2

        brand_total = brandx_interest + branda_interest + brandb_interest + brandc_interest
        sos = brandx_interest / brand_total if brand_total > 0 else 0

        for query, qset in gtrends_queries:
            if query == "BrandX":
                val = brandx_interest
            elif query == "BrandA":
                val = branda_interest
            elif query == "BrandB":
                val = brandb_interest
            elif query == "BrandC":
                val = brandc_interest
            elif query == "BrandX processor":
                val = brandx_interest * np.random.uniform(0.30, 0.40)
            elif query == "buy BrandX":
                val = 10 + trend[i] * 10 + shocks["Purchase Intent"][i] * 10 + noise_pi[i] * 2
                val = np.clip(val, 3, 50)
            elif query == "BrandX vs BrandA":
                val = 18 + trend[i] * 8 + shocks["Consideration"][i] * 8 + noise_consid[i] * 2
                val = np.clip(val, 5, 55)
            elif query == "CPU":
                val = 40 + seasonal[i] * 30 + np.random.randn() * 3
                val = np.clip(val, 20, 80)
            else:
                val = 30

            row = {
                "month": MONTHS[i].strftime("%Y-%m-%d"),
                "query": query,
                "query_set": qset,
                "relative_interest": round(float(np.clip(val, 0, 100)), 1),
            }
            if qset == "brand_comparison":
                if query == "BrandX":
                    row["share_of_search"] = round(float(sos), 4)
                else:
                    row["share_of_search"] = None
            gtrends_rows.append(row)

    df_gtrends = pd.DataFrame(gtrends_rows)
    df_gtrends.to_excel(OUTPUT_DIR / "google_trends.xlsx", index=False)
    print(f"  -> {len(df_gtrends)} rows written")

    # ─── FILE 7: Web Traffic Overview ─────────────────────────────────────

    print("[7/15] Generating web_traffic_overview.xlsx ...")

    web_rows = []
    base_traffic = 12000000
    for i in range(N_MONTHS):
        total = base_traffic * (1 + trend[i] * 0.3 + seasonal[i] * 1.8 +
                                shocks["Awareness"][i] * 1.5 + noise_web[i] * 0.05)
        total = max(8000000, int(total))
        unique = int(total * np.random.uniform(0.62, 0.70))

        direct_pct = np.random.uniform(0.22, 0.28)
        organic_pct = np.random.uniform(0.32, 0.38)
        paid_pct = np.random.uniform(0.12, 0.18)
        social_pct = np.random.uniform(0.04, 0.06)
        email_pct = np.random.uniform(0.02, 0.04)
        referral_pct = 1 - direct_pct - organic_pct - paid_pct - social_pct - email_pct

        web_rows.append({
            "month": MONTHS[i].strftime("%Y-%m-%d"),
            "total_visits": total,
            "unique_visitors": unique,
            "direct_traffic_visits": int(total * direct_pct),
            "organic_search_visits": int(total * organic_pct),
            "paid_search_visits": int(total * paid_pct),
            "social_referral_visits": int(total * social_pct),
            "email_visits": int(total * email_pct),
            "referral_visits": int(total * referral_pct),
        })

    df_web = pd.DataFrame(web_rows)
    df_web.loc[41, "social_referral_visits"] = None
    df_web.to_excel(OUTPUT_DIR / "web_traffic_overview.xlsx", index=False)
    print(f"  -> {len(df_web)} rows written")

    # ─── FILE 8: Web Traffic by Section ───────────────────────────────────

    print("[8/15] Generating web_traffic_by_section.xlsx ...")

    sections = {
        "core_processors":   {"visit_pct": 0.14, "ppv": 2.5, "time": 175, "bounce": 0.38},
        "products_consumer": {"visit_pct": 0.12, "ppv": 2.8, "time": 195, "bounce": 0.35},
        "comparison_tools":  {"visit_pct": 0.05, "ppv": 3.2, "time": 210, "bounce": 0.28},
        "support":           {"visit_pct": 0.10, "ppv": 2.0, "time": 120, "bounce": 0.45},
        "ark_configurator":  {"visit_pct": 0.035, "ppv": 4.5, "time": 245, "bounce": 0.22},
    }

    section_rows = []
    for i in range(N_MONTHS):
        total_visits = web_rows[i]["total_visits"]
        for sec, params in sections.items():
            visits = int(total_visits * params["visit_pct"] * np.random.uniform(0.90, 1.10))
            unique = int(visits * np.random.uniform(0.65, 0.78))

            if sec in ["products_consumer", "comparison_tools"]:
                visits = int(visits * (1 + shocks["Consideration"][i] * 0.5))
            elif sec == "ark_configurator":
                visits = int(visits * (1 + shocks["Purchase Intent"][i] * 0.8))

            pps = params["ppv"] * np.random.uniform(0.90, 1.10)
            section_rows.append({
                "month": MONTHS[i].strftime("%Y-%m-%d"),
                "site_section": sec,
                "visits": visits,
                "unique_visitors": unique,
                "page_views": int(visits * pps),
                "avg_time_on_site_seconds": round(params["time"] * np.random.uniform(0.85, 1.15), 1),
                "pages_per_session": round(pps, 1),
                "bounce_rate": round(params["bounce"] * np.random.uniform(0.90, 1.10), 2),
            })

    df_sections = pd.DataFrame(section_rows)
    df_sections.to_excel(OUTPUT_DIR / "web_traffic_by_section.xlsx", index=False)
    print(f"  -> {len(df_sections)} rows written")

    # ─── FILE 9: Web Conversion Events ────────────────────────────────────

    print("[9/15] Generating web_conversion_events.xlsx ...")

    event_types = {
        "comparison_tool_usage":    {"base": 35000, "construct": "Consideration"},
        "newsletter_signup":        {"base": 8000,  "construct": "Awareness"},
        "configurator_completion":  {"base": 15000, "construct": "Purchase Intent"},
        "resource_download":        {"base": 22000, "construct": "Consideration"},
        "contact_form_submission":  {"base": 3500,  "construct": "Purchase Intent"},
        "where_to_buy_click":       {"base": 28000, "construct": "Purchase Intent"},
    }

    conv_rows = []
    for i in range(N_MONTHS):
        for etype, params in event_types.items():
            shock_key = params["construct"]
            count = params["base"] * (1 + trend[i] * 0.3 + seasonal[i] * 1.5 +
                                       shocks[shock_key][i] * 1.5 + np.random.randn() * 0.06)
            count = max(int(params["base"] * 0.5), int(count))
            unique = int(count * np.random.uniform(0.70, 0.85))
            conv_rows.append({
                "month": MONTHS[i].strftime("%Y-%m-%d"),
                "event_type": etype,
                "event_count": count,
                "unique_users": unique,
            })

    df_conv = pd.DataFrame(conv_rows)
    df_conv.loc[15, "unique_users"] = None
    df_conv.to_excel(OUTPUT_DIR / "web_conversion_events.xlsx", index=False)
    print(f"  -> {len(df_conv)} rows written")

    # ─── FILE 10: Sell-Through ────────────────────────────────────────────

    print("[10/15] Generating sell_through.xlsx ...")

    sell_rows = []
    base_units = 2300000
    base_asp = 190
    for i in range(N_MONTHS):
        month_num = MONTHS[i].month
        holiday_mult = 1.0
        if month_num == 11:
            holiday_mult = 1.30
        elif month_num == 12:
            holiday_mult = 1.40
        elif month_num in [1, 2]:
            holiday_mult = 0.80

        units = base_units * holiday_mult * (1 + trend[i] * 0.25 +
                                              shocks["Purchase Intent"][i] * 1.0 +
                                              noise_sales[i] * 0.06)
        units = max(1500000, int(units))
        asp = base_asp * (1 + 0.003 * i + np.random.randn() * 0.02)
        asp = round(np.clip(asp, 170, 220), 2)
        revenue = int(units * asp)

        sell_rows.append({
            "month": MONTHS[i].strftime("%Y-%m-%d"),
            "segment": "Consumer PC",
            "units_sold": units,
            "revenue_usd": revenue,
            "asp_usd": asp,
        })

    df_sell = pd.DataFrame(sell_rows)
    df_sell.to_excel(OUTPUT_DIR / "sell_through.xlsx", index=False)
    print(f"  -> {len(df_sell)} rows written")

    # ─── FILE 11: Market Share (Quarterly) ────────────────────────────────

    print("[11/15] Generating market_share.xlsx ...")

    quarters = pd.date_range("2021-01-01", "2025-12-31", freq="QS")[:20]

    share_rows = []
    vendors = {
        "BrandX": {"start": 65, "end": 60, "trend": "declining"},
        "BrandA": {"start": 26, "end": 31, "trend": "rising"},
        "BrandD": {"start": 7,  "end": 8,  "trend": "rising"},
        "BrandC": {"start": 0.5, "end": 2.5, "trend": "rising"},
    }

    for qi, q in enumerate(quarters):
        progress = qi / (len(quarters) - 1) if len(quarters) > 1 else 0
        period_label = f"Q{(q.month - 1) // 3 + 1}_{q.year}"
        base_units_q = 8000000

        for vendor, params in vendors.items():
            share = params["start"] + (params["end"] - params["start"]) * progress
            share += np.random.randn() * 0.8
            share = max(0.1, share)

            units = int(base_units_q * share / 100)
            rev_share = share * np.random.uniform(0.95, 1.05)

            share_rows.append({
                "period": period_label,
                "period_type": "quarterly",
                "segment": "Consumer PC",
                "vendor": vendor,
                "unit_share_pct": round(share, 1),
                "units_shipped": units,
                "revenue_share_pct": round(rev_share, 1),
            })

    df_share = pd.DataFrame(share_rows)
    df_share.to_excel(OUTPUT_DIR / "market_share.xlsx", index=False)
    print(f"  -> {len(df_share)} rows written")

    # ─── FILE 12: Campaign Calendar ───────────────────────────────────────

    print("[12/15] Generating campaign_calendar.xlsx ...")

    campaign_rows = []
    for c in CAMPAIGNS:
        campaign_rows.append({
            "campaign_id": c["id"],
            "campaign_name": c["name"],
            "flight_start_date": c["start"],
            "flight_end_date": c["end"],
            "objective": c["objective"],
            "target_audience": c["audience"],
            "geo_scope": c["geo"],
            "channels": c["channels"],
            "spend_tier": c["tier"],
            "spend_usd": c["spend"],
        })

    df_campaigns = pd.DataFrame(campaign_rows)
    df_campaigns.to_excel(OUTPUT_DIR / "campaign_calendar.xlsx", index=False)
    print(f"  -> {len(df_campaigns)} rows written")

    # ─── FILE 13: Ad Spend Monthly ────────────────────────────────────────

    print("[13/15] Generating ad_spend_monthly.xlsx ...")

    channels = ["TV", "CTV_OTT", "Digital Display", "Paid Social", "Paid Search", "Video", "Total"]
    channel_pcts = {"TV": 0.25, "CTV_OTT": 0.12, "Digital Display": 0.22,
                    "Paid Social": 0.15, "Paid Search": 0.14, "Video": 0.12}

    spend_rows = []
    for i in range(N_MONTHS):
        month_date = MONTHS[i]
        total_spend = 5000000
        for c in CAMPAIGNS:
            c_start = pd.Timestamp(c["start"])
            c_end = pd.Timestamp(c["end"])
            if c_start <= month_date <= c_end:
                c_months = max(1, (c_end.year - c_start.year) * 12 + c_end.month - c_start.month + 1)
                total_spend += c["spend"] / c_months

        total_spend = int(total_spend * np.random.uniform(0.92, 1.08))

        for ch in channels:
            if ch == "Total":
                spend_rows.append({
                    "month": MONTHS[i].strftime("%Y-%m-%d"),
                    "channel": ch,
                    "spend_usd": total_spend,
                    "impressions": int(total_spend * np.random.uniform(8, 15)),
                    "target_audience": "Consumer",
                })
            else:
                ch_spend = int(total_spend * channel_pcts[ch] * np.random.uniform(0.85, 1.15))
                spend_rows.append({
                    "month": MONTHS[i].strftime("%Y-%m-%d"),
                    "channel": ch,
                    "spend_usd": ch_spend,
                    "impressions": int(ch_spend * np.random.uniform(8, 15)),
                    "target_audience": "Consumer",
                })

    df_spend = pd.DataFrame(spend_rows)
    df_spend.to_excel(OUTPUT_DIR / "ad_spend_monthly.xlsx", index=False)
    print(f"  -> {len(df_spend)} rows written")

    # ─── FILE 14: Event Registry ──────────────────────────────────────────

    print("[14/15] Generating event_registry.xlsx ...")

    event_rows = []
    for evt in EVENTS:
        event_rows.append({
            "event_date": evt["date"],
            "event_end_date": evt["end"],
            "event_type": evt["type"],
            "event_name": evt["name"],
            "constructs_affected": evt["constructs"],
            "description": evt["desc"],
        })

    df_events = pd.DataFrame(event_rows)
    df_events.to_excel(OUTPUT_DIR / "event_registry.xlsx", index=False)
    print(f"  -> {len(df_events)} rows written")

    # ─── FILE 15: Macro-Economic ──────────────────────────────────────────

    print("[15/15] Generating macro_economic.xlsx ...")

    macro_rows = []
    for i in range(N_MONTHS):
        progress = i / (N_MONTHS - 1)
        cci = 102 + 20 * np.sin(progress * np.pi * 0.8) + np.random.randn() * 3
        if i < 6:
            cci -= 5
        elif 30 < i < 38:
            cci -= 8
        cci = np.clip(cci, 95, 135)

        dram = 100
        if i < 12:
            dram = 120 - 2 * i + np.random.randn() * 3
        elif i < 24:
            dram = 100 - 3 * (i - 12) + np.random.randn() * 4
        elif i < 30:
            dram = 65 + np.random.randn() * 3
        elif i < 42:
            dram = 65 + 3 * (i - 30) + np.random.randn() * 3
        else:
            dram = 100 + 1.5 * (i - 42) + np.random.randn() * 3
        dram = max(50, dram)

        macro_rows.append({
            "month": MONTHS[i].strftime("%Y-%m-%d"),
            "consumer_confidence_index": round(cci, 1),
            "dram_spot_price_index": round(dram, 1),
        })

    df_macro = pd.DataFrame(macro_rows)
    df_macro.to_excel(OUTPUT_DIR / "macro_economic.xlsx", index=False)
    print(f"  -> {len(df_macro)} rows written")

    # ─── VALIDATION ───────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    all_pass = True

    print("\n[CHECK 1] Date ranges ...")
    for fname in ["social_mention_volume.xlsx", "social_sentiment.xlsx", "web_traffic_overview.xlsx"]:
        df = pd.read_excel(OUTPUT_DIR / fname)
        months = pd.to_datetime(df["month"])
        assert months.min() == pd.Timestamp("2021-03-01"), f"{fname}: start date wrong"
        assert months.max() == pd.Timestamp("2026-02-01"), f"{fname}: end date wrong"
        assert len(df) == 60, f"{fname}: expected 60 rows, got {len(df)}"
    print("  PASS All monthly files span Mar 2021 – Feb 2026 (60 months)")

    assert len(df_tracker["wave_id"].unique()) == 13, "Tracker: expected 13 waves"
    print("  PASS Brand tracker has 13 waves")

    assert len(df_share["period"].unique()) == 20, f"Market share: expected 20 quarters"
    print("  PASS Market share has 20 quarters")

    print("\n[CHECK 2] No negative values ...")
    for fname in ["social_mention_volume.xlsx", "sell_through.xlsx", "ad_spend_monthly.xlsx"]:
        df = pd.read_excel(OUTPUT_DIR / fname)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dropna().min() < 0:
                print(f"  FAIL: {fname}.{col} has negative values!")
                all_pass = False
    print("  PASS No negative values found")

    print("\n[CHECK 3] Branded search vs Awareness correlation ...")
    df_s = pd.read_excel(OUTPUT_DIR / "search_volume_by_keyword.xlsx")
    brandx_branded = df_s[
        (df_s["keyword_group"] == "branded_awareness") &
        (df_s["keyword"].isin(["BrandX", "BrandX processor", "BrandX CPU", "BrandX Core"]))
    ].groupby("month")["search_volume"].sum().reset_index()
    brandx_branded["month"] = pd.to_datetime(brandx_branded["month"])

    wave_months = [pd.Timestamp(w["midpoint"]) for w in WAVES]
    aware_scores = df_tracker[(df_tracker["brand"] == "BrandX") & (df_tracker["construct"] == "Awareness")]["score"].values

    search_at_waves = []
    for wm in wave_months:
        match = brandx_branded[brandx_branded["month"] == wm]
        if len(match) > 0:
            search_at_waves.append(match["search_volume"].values[0])
        else:
            search_at_waves.append(np.nan)

    search_at_waves = np.array(search_at_waves)
    valid = ~np.isnan(search_at_waves)
    if valid.sum() >= 5:
        r, p = pearsonr(search_at_waves[valid], aware_scores[valid])
        status = "PASS" if r >= 0.60 else "FAIL"
        print(f"  {status}: r = {r:.3f} (target >= 0.60), p = {p:.4f}")
        if r < 0.60:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review above")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    generate_all()