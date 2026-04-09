"""
Task 1: Audit and clean macro_economic.xlsx
Task 2: Build COVID dummy reference file

Follows Skill 2 (5 audit checks), Skill 3 (cleaning priority order), Skill 4e (COVID dummy)
Project window: 2021-03 to 2026-02 (60 months)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Logging setup (from skills.md Automatic Cleaning Log) ──────────────────

log_entries = []

def log_change(file, column, change, reason, method):
    log_entries.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'file': file,
        'column': column,
        'change_made': change,
        'reason': reason,
        'method_used': method
    })
    print(f"LOG: [{file}] {column} — {change}")

def save_log():
    Path("logs/").mkdir(exist_ok=True)
    log_df = pd.DataFrame(log_entries)
    log_path = "logs/cleaning_log.csv"
    if Path(log_path).exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved: {len(log_entries)} new entries added to logs/cleaning_log.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Audit and clean macro_economic.xlsx
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_FILE = "data/dummy/macro_economic.xlsx"
PROCESSED_FILE = "data/processed/macro_economic_cleaned.csv"
PROJECT_START = '2021-03'
PROJECT_END   = '2026-02'
TOTAL_MONTHS  = 60

print("=" * 60)
print("AUDIT: macro_economic.xlsx")
print("=" * 60)

# Load into df_base — read-only from here on
df_base = pd.read_excel(SOURCE_FILE)

# Always print on load
print("\nShape:", df_base.shape)
print("Columns:", df_base.columns.tolist())
print("\nDtypes:\n", df_base.dtypes)
print("\nNull counts:\n", df_base.isnull().sum())

# ── CHECK 1 — Completeness ──────────────────────────────────────────────────
print("\n" + "─" * 50)
print("CHECK 1 — Completeness")
print("─" * 50)

filled = df_base.shape[0]
completeness_pct = round(filled / TOTAL_MONTHS * 100, 1)
print(f"Rows loaded: {filled}")
print(f"Expected rows (project window): {TOTAL_MONTHS}")
print(f"Completeness: {filled}/{TOTAL_MONTHS} months = {completeness_pct}%")
if completeness_pct < 90:
    print("FLAG: Below 90% — investigate before proceeding")
else:
    print("PASS: >= 90% completeness")

# ── CHECK 2 — Date format ───────────────────────────────────────────────────
print("\n" + "─" * 50)
print("CHECK 2 — Date format")
print("─" * 50)

date_col = df_base.columns[0]
print(f"Date column: '{date_col}'")
print("Sample values:", df_base[date_col].head(5).tolist())
print("Dtype:", df_base[date_col].dtype)

# ── CHECK 3 — Granularity ───────────────────────────────────────────────────
print("\n" + "─" * 50)
print("CHECK 3 — Granularity")
print("─" * 50)
print(f"Row count: {df_base.shape[0]}")
# Try to infer from date column if parseable
try:
    parsed_dates = pd.to_datetime(df_base[date_col], infer_datetime_format=True)
    diffs = parsed_dates.sort_values().diff().dropna()
    median_diff_days = diffs.median().days
    if median_diff_days < 10:
        print("Inferred granularity: DAILY")
    elif median_diff_days < 20:
        print("Inferred granularity: WEEKLY")
    elif 25 <= median_diff_days <= 35:
        print("Inferred granularity: MONTHLY")
    elif 80 <= median_diff_days <= 100:
        print("Inferred granularity: QUARTERLY")
    else:
        print(f"Inferred granularity: UNKNOWN (median gap = {median_diff_days} days)")
except Exception:
    print("Could not infer granularity from date column — check manually")

# ── CHECK 4 — Range validity ────────────────────────────────────────────────
print("\n" + "─" * 50)
print("CHECK 4 — Range validity")
print("─" * 50)

numeric_cols = df_base.select_dtypes(include='number').columns.tolist()
print("Numeric columns:", numeric_cols)
for col in numeric_cols:
    print(f"\n  {col}:")
    print(f"    min={df_base[col].min()}, max={df_base[col].max()}")
    print(f"    negatives={(df_base[col] < 0).sum()}")
    print(f"    nulls={df_base[col].isnull().sum()}")

# CCI-specific range check
# Conference Board Consumer Confidence Index: valid range 0–200
# Readings < 0 are impossible; readings > 300 would be extreme outliers
CCI_COL = None
for col in df_base.columns:
    if 'consumer_confidence' in col.lower() or 'cci' in col.lower() or 'confidence_index' in col.lower():
        CCI_COL = col
        break

if CCI_COL:
    print(f"\nCCI column identified: '{CCI_COL}'")
    cci_min = df_base[CCI_COL].min()
    cci_max = df_base[CCI_COL].max()
    # Conference Board CCI: theoretical range 0–200, never negative
    cci_invalid = df_base[(df_base[CCI_COL] < 0) | (df_base[CCI_COL] > 200)]
    print(f"  Range found: {cci_min:.2f} to {cci_max:.2f}")
    print(f"  Expected range: 0 to 200 (Conference Board CCI)")
    if len(cci_invalid) > 0:
        print(f"  FLAG: {len(cci_invalid)} rows outside valid range [0, 200]:")
        print(cci_invalid[[date_col, CCI_COL]].to_string())
    else:
        print(f"  PASS: All {CCI_COL} values within [0, 200]")
else:
    print("\nWARNING: No column identified as consumer_confidence_index — check column names above")

# ── CHECK 5 — Structural breaks ─────────────────────────────────────────────
print("\n" + "─" * 50)
print("CHECK 5 — Structural breaks (macro / external data)")
print("─" * 50)
print("Questions to confirm with data provider:")
print("  Conference Board CCI: Is the methodology consistent across the full window?")
print("    e.g. Any panel or survey methodology changes 2021–2026?")
print("    If CCI shows a sudden level shift unrelated to macro events — flag it.")
print("  DRAM pricing: Is the source (TrendForce) consistent throughout?")
print("    e.g. Spot price vs contract price — must be the same definition throughout.")
print("    Any product category scope changes (DDR4 only vs DDR4+DDR5 mixed)?")
print("    A sudden level shift in DRAM pricing unrelated to market conditions — flag it.")
# Visual scan for large jumps (>50% month-on-month) that may indicate structural break
for col in numeric_cols:
    series = df_base[col].dropna()
    if len(series) > 1:
        pct_change = series.pct_change().abs()
        large_jumps = pct_change[pct_change > 0.5]
        if len(large_jumps) > 0:
            print(f"\n  Potential structural break — {col} has {len(large_jumps)} month(s) with >50% change:")
            # Get corresponding dates
            jump_indices = large_jumps.index
            jump_dates = df_base.loc[jump_indices, date_col].tolist()
            jump_vals  = large_jumps.values
            for d, v in zip(jump_dates, jump_vals):
                print(f"    {d}: {v*100:.1f}% change — verify against known macro events")


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING — Priority order from Skill 3
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLEANING: macro_economic.xlsx")
print("=" * 60)

df_clean = df_base.copy()

# PRIORITY 1 — Fix date format -> standard YYYY-MM
print("\n[PRIORITY 1] Fix date format")

original_dates = df_clean[date_col].head(3).tolist()

df_clean[date_col] = pd.to_datetime(df_clean[date_col], infer_datetime_format=True) \
                       .dt.to_period('M').astype(str)

new_dates = df_clean[date_col].head(3).tolist()
print(f"  Before: {original_dates}")
print(f"  After:  {new_dates}")

log_change(
    file="macro_economic.xlsx",
    column=date_col,
    change=f"Converted to YYYY-MM format (was: {df_base[date_col].dtype})",
    reason="Project standard — all dates must be YYYY-MM",
    method="pd.to_datetime().dt.to_period('M').astype(str)"
)

# Create date spine — project window 2021-03 to 2026-02
date_spine = pd.DataFrame({
    date_col: pd.period_range(start=PROJECT_START, end=PROJECT_END, freq='M').astype(str)
})

rows_before = df_clean.shape[0]
df_clean = date_spine.merge(df_clean, on=date_col, how='left')
rows_after = df_clean.shape[0]

print(f"\n  Date spine created: {PROJECT_START} to {PROJECT_END} ({rows_after} rows)")
if rows_before < TOTAL_MONTHS:
    added = rows_after - rows_before
    print(f"  {added} missing month rows added by spine merge")
    log_change(
        file="macro_economic.xlsx",
        column=date_col,
        change=f"Date spine applied — {added} missing rows added",
        reason=f"Source had {rows_before} rows, project window requires {TOTAL_MONTHS}",
        method="date spine merge left join"
    )
else:
    print("  All project months present before spine merge")
    log_change(
        file="macro_economic.xlsx",
        column=date_col,
        change="Date spine applied — no rows added, source coverage complete",
        reason=f"Source had {rows_before} rows covering project window",
        method="date spine merge left join"
    )

# PRIORITY 2 — Handle missing values
print("\n[PRIORITY 2] Handle missing values")

numeric_cols_clean = df_clean.select_dtypes(include='number').columns.tolist()

for col in numeric_cols_clean:
    null_count = df_clean[col].isnull().sum()
    if null_count == 0:
        print(f"  {col}: no nulls — skip")
        continue

    is_null = df_clean[col].isnull()
    # Find max consecutive null run
    consecutive = (
        is_null.groupby((is_null != is_null.shift()).cumsum())
               .sum()
               .max()
    )

    if consecutive <= 2:
        df_clean[col] = df_clean[col].interpolate(method='linear')
        print(f"  {col}: {null_count} nulls filled — max {consecutive} consecutive")
        log_change(
            file="macro_economic.xlsx",
            column=col,
            change=f"{null_count} nulls filled",
            reason=f"Missing months in time series (max {consecutive} consecutive)",
            method="linear interpolation"
        )
    else:
        print(f"  FLAG: {col} has {null_count} nulls, max {consecutive} consecutive — NOT filled, needs discussion")
        log_change(
            file="macro_economic.xlsx",
            column=col,
            change=f"FLAG — {null_count} nulls NOT filled, max {consecutive} consecutive",
            reason="Consecutive gap exceeds 2-month interpolation limit",
            method="no action — flagged for discussion"
        )

# PRIORITY 3 — Granularity (already monthly — no action needed if confirmed above)
# Skipping aggregation step — data is monthly

# PRIORITY 4 — Detect outliers (flag only, do not remove)
print("\n[PRIORITY 4] Outlier detection (flag only)")

for col in numeric_cols_clean:
    rolling_mean = df_clean[col].rolling(12, min_periods=3).mean()
    rolling_std  = df_clean[col].rolling(12, min_periods=3).std()
    outlier_mask = (df_clean[col] - rolling_mean).abs() > 3 * rolling_std
    outlier_rows = df_clean[outlier_mask][[date_col, col]]
    if len(outlier_rows) > 0:
        print(f"\n  Outliers in {col} ({len(outlier_rows)} rows):")
        print(outlier_rows.to_string(index=False))
        log_change(
            file="macro_economic.xlsx",
            column=col,
            change=f"FLAG — {len(outlier_rows)} outlier(s) detected",
            reason="Value > 3 SD from 12-month rolling mean",
            method="kept — check event registry before any action"
        )
    else:
        print(f"  {col}: no outliers detected")

# PRIORITY 5 — CCI final range validation on cleaned data
print("\n[PRIORITY 5] CCI range validation on cleaned data")
if CCI_COL:
    cci_invalid_post = df_clean[(df_clean[CCI_COL] < 0) | (df_clean[CCI_COL] > 200)]
    if len(cci_invalid_post) > 0:
        print(f"  FLAG: {len(cci_invalid_post)} CCI values still outside [0, 200] after cleaning:")
        print(cci_invalid_post[[date_col, CCI_COL]].to_string())
        log_change(
            file="macro_economic.xlsx",
            column=CCI_COL,
            change=f"FLAG — {len(cci_invalid_post)} values outside valid range [0, 200]",
            reason="Conference Board CCI cannot be negative or exceed 200",
            method="flagged — needs investigation"
        )
    else:
        print(f"  PASS: All CCI values within valid range [0, 200]")
        log_change(
            file="macro_economic.xlsx",
            column=CCI_COL,
            change="Range validation passed — all values in [0, 200]",
            reason="Conference Board CCI valid range check",
            method="min/max bounds check"
        )

# Save cleaned file
Path("data/processed/").mkdir(exist_ok=True)
df_clean.to_csv(PROCESSED_FILE, index=False)
print(f"\nSaved: {PROCESSED_FILE}")
print("Shape:", df_clean.shape)
print("Columns:", df_clean.columns.tolist())
print("\nFirst 5 rows:")
print(df_clean.head().to_string(index=False))
print("\nNull counts after cleaning:")
print(df_clean.isnull().sum())

log_change(
    file="macro_economic.xlsx",
    column="ALL",
    change=f"Cleaned file saved to {PROCESSED_FILE}",
    reason="Task 1 complete",
    method="to_csv"
)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Build COVID dummy reference file (Skill 4e)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TASK 2 — Build COVID dummy (Skill 4e)")
print("=" * 60)

# Date range: project window 2021-03 to 2026-02
dates = pd.period_range(start='2021-03', end='2026-02', freq='M').astype(str)
covid_df = pd.DataFrame({'date': dates})

# March 2020 to June 2021 = 1, all others = 0
# Within the 2021-03 to 2026-02 window, the overlap is 2021-03 to 2021-06
covid_df['covid_dummy'] = 0
covid_df.loc[
    (covid_df['date'] >= '2020-03') & (covid_df['date'] <= '2021-06'),
    'covid_dummy'
] = 1

print("\nCOVID dummy summary:")
print(f"  Total rows: {len(covid_df)}")
print(f"  Months flagged as 1: {covid_df['covid_dummy'].sum()}")

flagged = covid_df[covid_df['covid_dummy'] == 1]
if len(flagged) > 0:
    print(f"  First flagged month: {flagged['date'].iloc[0]}")
    print(f"  Last flagged month:  {flagged['date'].iloc[-1]}")
    print(f"  Flagged months: {flagged['date'].tolist()}")
else:
    print("  No months flagged (COVID period 2020-03 to 2021-06 is fully outside window 2021-03 to 2026-02)")

print(f"\nDate range: {covid_df['date'].iloc[0]} to {covid_df['date'].iloc[-1]}")
print("\nDistribution:")
print(covid_df['covid_dummy'].value_counts().sort_index().to_string())

covid_df.to_csv("data/processed/covid_dummy.csv", index=False)
print("\nSaved: data/processed/covid_dummy.csv")

log_change(
    file="covid_dummy.csv",
    column="covid_dummy",
    change="COVID dummy file created — 2021-03 to 2026-02 window; 2021-03 to 2021-06 flagged as 1",
    reason="Project window is 2021-03 to 2026-02; COVID flag period is March 2020–June 2021",
    method="Skill 4e — period_range + boolean mask"
)


# ── Save all log entries ─────────────────────────────────────────────────────
save_log()

print("\n" + "=" * 60)
print("DONE — Both tasks complete")
print(f"  Task 1: data/processed/macro_economic_cleaned.csv")
print(f"  Task 2: data/processed/covid_dummy.csv")
print(f"  Log:    logs/cleaning_log.csv")
print("=" * 60)
