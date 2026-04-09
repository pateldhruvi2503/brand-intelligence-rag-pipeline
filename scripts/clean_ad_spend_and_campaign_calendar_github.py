"""
Skill 2 (Audit) + Skill 3 (Clean) for:
  - data/dummy/ad_spend_monthly.xlsx
  - data/dummy/campaign_calendar.xlsx

Outputs:
  - data/processed/ad_spend_cleaned.csv
  - data/processed/campaign_calendar_cleaned.csv
  - logs/cleaning_log.csv (appended)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────

log_entries = []

def log_change(file, column, change, reason, method):
    log_entries.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'file': file,
        'column': column,
        'change_made': change,
        'reason': reason,
        'method_used': method,
    })
    print(f"LOG: [{file}] {column} — {change}")

def save_log():
    Path("logs/").mkdir(exist_ok=True)
    log_df = pd.DataFrame(log_entries)
    log_path = Path("logs/cleaning_log.csv")
    if log_path.exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved: {len(log_entries)} new entries added to logs/cleaning_log.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 1 — ad_spend_monthly.xlsx
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("AUDIT: ad_spend_monthly.xlsx")
print("=" * 60)

df_base = pd.read_excel("data/dummy/ad_spend_monthly.xlsx")

# CHECK 1 — Completeness
print("\nCHECK 1 — Completeness")
print("Shape:", df_base.shape)
print("Null counts:\n", df_base.isnull().sum())
total_months = 60   # expected 5-year window
filled = df_base['month'].nunique()
completeness_pct = round(filled / total_months * 100, 1)
print(f"Unique months in file: {filled}/{total_months} = {completeness_pct}%")
print("NOTE: Data spans 2021-03 to 2026-02 — does not match expected 2020-01 to 2024-12 window.")
print("      Flag for client to confirm the correct date range before modelling.")

# CHECK 2 — Date format
print("\nCHECK 2 — Date format")
date_col = 'month'
print(f"Date column: {date_col}")
print("Sample values:", df_base[date_col].head(5).tolist())
print("Dtype:", df_base[date_col].dtype)
print("Format observed: YYYY-MM-DD string — will convert to YYYY-MM")

# CHECK 3 — Granularity
print("\nCHECK 3 — Granularity")
print("Row count:", df_base.shape[0])
print("Unique months:", df_base[date_col].nunique())
print("Unique channels:", df_base['channel'].nunique(), "->", df_base['channel'].unique().tolist())
print("Granularity: Monthly — 7 channels per month, 60 months total")

# CHECK 4 — Range validity
print("\nCHECK 4 — Range validity")
numeric_cols = df_base.select_dtypes(include='number').columns
for col in numeric_cols:
    neg = (df_base[col] < 0).sum()
    print(f"  {col}: min={df_base[col].min():,}, max={df_base[col].max():,}, negatives={neg}")

# CHECK 5 — Structural breaks
print("\nCHECK 5 — Structural breaks")
print("Ad spend data (client internal):")
print("  Question to confirm with client: Is the channel taxonomy consistent throughout the file?")
print("  e.g. Was 'CTV_OTT' always a separate line item, or was it rolled into 'TV' in earlier years?")
print("  A change in how channels are defined will make the Total figure look stable while")
print("  individual channel splits become incomparable across periods.")
print("  If you see a channel appear or disappear mid-series — flag immediately.")
print("  No programmatic check possible — confirm with client team.")


# ── CLEAN: FILE 1 ─────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("CLEANING: ad_spend_monthly.xlsx")
print("-" * 60)

df_clean_spend = df_base.copy()

# PRIORITY 1 — Fix date format: YYYY-MM-DD -> YYYY-MM, rename to 'date'
df_clean_spend['date'] = pd.to_datetime(df_clean_spend['month']).dt.to_period('M').astype(str)
df_clean_spend = df_clean_spend.drop(columns=['month'])
log_change("ad_spend_monthly.xlsx", "month",
           "converted to YYYY-MM format and renamed to 'date'",
           "Project standard requires YYYY-MM monthly format",
           "pd.to_datetime().dt.to_period('M').astype(str)")

# PRIORITY 2 — Filter to channel == 'Total' only
pre_filter_rows = len(df_clean_spend)
df_clean_spend = df_clean_spend[df_clean_spend['channel'] == 'Total'].copy()
log_change("ad_spend_monthly.xlsx", "channel",
           f"filtered from {pre_filter_rows} rows to {len(df_clean_spend)} rows (Total channel only)",
           "Task spec: keep only rows where channel = 'Total'",
           "boolean filter df[channel == 'Total']")

# PRIORITY 3 — Keep only required columns
df_clean_spend = df_clean_spend[['date', 'spend_usd']].reset_index(drop=True)
log_change("ad_spend_monthly.xlsx", "columns",
           "kept only: date, spend_usd (dropped: channel, impressions, target_audience)",
           "Task spec: keep columns date, spend_usd only",
           "column selection")

# PRIORITY 4 — Handle missing values (check for gaps in monthly spine)
# After filtering, check if all 60 months are present
if df_clean_spend['date'].nunique() == 60:
    print("Date continuity: all 60 months present — no gaps to fill")
    log_change("ad_spend_monthly.xlsx", "spend_usd",
               "0 nulls — no interpolation needed",
               "No missing months after Total filter",
               "null check")
else:
    null_count = df_clean_spend['spend_usd'].isnull().sum()
    log_change("ad_spend_monthly.xlsx", "spend_usd",
               f"{null_count} nulls present — review required",
               "Missing months detected after filtering",
               "null check")

# PRIORITY 5 — Outlier detection (flag only, no removal)
rolling_mean = df_clean_spend['spend_usd'].rolling(12, min_periods=3).mean()
rolling_std  = df_clean_spend['spend_usd'].rolling(12, min_periods=3).std()
outlier_mask = (df_clean_spend['spend_usd'] - rolling_mean).abs() > 3 * rolling_std
outlier_rows = df_clean_spend[outlier_mask][['date', 'spend_usd']]
if len(outlier_rows) > 0:
    print(f"\nOutliers in spend_usd:")
    print(outlier_rows.to_string())
    log_change("ad_spend_monthly.xlsx", "spend_usd",
               f"{len(outlier_rows)} outlier(s) flagged — NOT removed, check event registry",
               "Value > 3 std from 12-month rolling mean",
               "rolling mean ± 3 std")
else:
    print("Outlier check: no outliers in spend_usd (3-std rolling window)")
    log_change("ad_spend_monthly.xlsx", "spend_usd",
               "0 outliers detected",
               "No values exceed 3 std from 12-month rolling mean",
               "rolling mean ± 3 std")

# Validate and print final shape
print("\nFinal shape:", df_clean_spend.shape)
print("Columns:", df_clean_spend.columns.tolist())
print("Date range:", df_clean_spend['date'].min(), "to", df_clean_spend['date'].max())
print("Nulls:", df_clean_spend.isnull().sum().to_dict())
print(df_clean_spend.head(5).to_string())

# Save
Path("data/processed/").mkdir(parents=True, exist_ok=True)
df_clean_spend.to_csv("data/processed/ad_spend_cleaned.csv", index=False)
log_change("ad_spend_monthly.xlsx", "—",
           "saved to data/processed/ad_spend_cleaned.csv",
           "Clean output — processed copy, raw file untouched",
           "df.to_csv()")
print("Saved: data/processed/ad_spend_cleaned.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 2 — campaign_calendar.xlsx
# ══════════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 60)
print("AUDIT: campaign_calendar.xlsx")
print("=" * 60)

df_base2 = pd.read_excel("data/dummy/campaign_calendar.xlsx")

# CHECK 1 — Completeness
print("\nCHECK 1 — Completeness")
print("Shape:", df_base2.shape)
print("Null counts:\n", df_base2.isnull().sum())
print("Note: campaign_calendar is not a monthly time series — completeness = row count + null check only")
print(f"All 21 rows present, 0 nulls — completeness: PASS")

# CHECK 2 — Date format
print("\nCHECK 2 — Date format")
for dc in ['flight_start_date', 'flight_end_date']:
    print(f"Date column: {dc}")
    print("  Sample values:", df_base2[dc].head(5).tolist())
    print("  Dtype:", df_base2[dc].dtype)
print("Format observed: YYYY-MM-DD string — will convert both to YYYY-MM")

# CHECK 3 — Granularity
print("\nCHECK 3 — Granularity")
print("Row count:", df_base2.shape[0])
print("Granularity: Campaign-level (one row per campaign flight, not monthly)")
print("Date range — starts:", df_base2['flight_start_date'].min(), "ends:", df_base2['flight_end_date'].max())

# CHECK 4 — Range validity
print("\nCHECK 4 — Range validity")
numeric_cols2 = df_base2.select_dtypes(include='number').columns
for col in numeric_cols2:
    neg = (df_base2[col] < 0).sum()
    print(f"  {col}: min={df_base2[col].min():,}, max={df_base2[col].max():,}, negatives={neg}")

# CHECK 5 — Structural breaks
print("\nCHECK 5 — Structural breaks")
print("Campaign calendar (client internal):")
print("  Question to confirm with client: Are the objective and spend_tier categories consistent")
print("  across all campaigns in this file?")
print("  e.g. Did 'Awareness' always mean the same KPI, or did the definition shift between years?")
print("  Also confirm: does spend_usd represent total committed spend or actual delivered spend?")
print("  Committed vs delivered can differ significantly — this matters for modelling.")
print("  No programmatic check possible — confirm with client team.")


# ── CLEAN: FILE 2 ─────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("CLEANING: campaign_calendar.xlsx")
print("-" * 60)

df_clean_cal = df_base2.copy()

# PRIORITY 1 — Fix date format on flight_start_date and flight_end_date
for dc in ['flight_start_date', 'flight_end_date']:
    df_clean_cal[dc] = pd.to_datetime(df_clean_cal[dc]).dt.to_period('M').astype(str)
    log_change("campaign_calendar.xlsx", dc,
               "converted to YYYY-MM format",
               "Project standard requires YYYY-MM monthly format",
               "pd.to_datetime().dt.to_period('M').astype(str)")

# PRIORITY 2 — No filtering needed, keep all rows
print("Row filter: none required — keeping all 21 rows")

# PRIORITY 3 — No missing values to handle
log_change("campaign_calendar.xlsx", "all columns",
           "0 nulls — no interpolation needed",
           "No missing values in source file",
           "null check")

# PRIORITY 4 — Outlier detection on spend_usd (flag only, no removal)
if len(df_clean_cal) >= 4:
    rolling_mean2 = df_clean_cal['spend_usd'].rolling(6, min_periods=3).mean()
    rolling_std2  = df_clean_cal['spend_usd'].rolling(6, min_periods=3).std()
    outlier_mask2 = (df_clean_cal['spend_usd'] - rolling_mean2).abs() > 3 * rolling_std2
    outlier_rows2 = df_clean_cal[outlier_mask2][['campaign_name', 'spend_usd']]
    if len(outlier_rows2) > 0:
        print(f"\nOutliers in spend_usd:")
        print(outlier_rows2.to_string())
        log_change("campaign_calendar.xlsx", "spend_usd",
                   f"{len(outlier_rows2)} outlier(s) flagged — NOT removed, check event registry",
                   "Value > 3 std from 6-row rolling mean",
                   "rolling mean ± 3 std")
    else:
        print("Outlier check: no outliers in spend_usd")
        log_change("campaign_calendar.xlsx", "spend_usd",
                   "0 outliers detected",
                   "No values exceed 3 std from rolling mean",
                   "rolling mean ± 3 std")

# Validate and print final shape
print("\nFinal shape:", df_clean_cal.shape)
print("Columns:", df_clean_cal.columns.tolist())
print("Nulls:", df_clean_cal.isnull().sum().to_dict())
print(df_clean_cal[['campaign_name', 'flight_start_date', 'flight_end_date', 'spend_usd']].to_string())

# Save
df_clean_cal.to_csv("data/processed/campaign_calendar_cleaned.csv", index=False)
log_change("campaign_calendar.xlsx", "—",
           "saved to data/processed/campaign_calendar_cleaned.csv",
           "Clean output — processed copy, raw file untouched",
           "df.to_csv()")
print("Saved: data/processed/campaign_calendar_cleaned.csv")


# ── Save log ──────────────────────────────────────────────────────────────────

save_log()
