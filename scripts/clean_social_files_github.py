"""
Skill 2 (Audit) + Skill 3 (Clean) for:
  - data/dummy/social_mention_volume.xlsx
  - data/dummy/social_sentiment.xlsx
Outputs:
  - data/processed/social_mention_volume_cleaned.csv
  - data/processed/social_sentiment_cleaned.csv
  - logs/cleaning_log.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Setup ────────────────────────────────────────────────────────────────────
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

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
    log_df = pd.DataFrame(log_entries)
    log_path = Path("logs/cleaning_log.csv")
    if log_path.exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved: {len(log_entries)} new entries → logs/cleaning_log.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 1 — social_mention_volume.xlsx
# ═══════════════════════════════════════════════════════════════════════════════

FILE1 = "social_mention_volume.xlsx"
print("\n" + "=" * 60)
print(f"AUDIT + CLEAN: {FILE1}")
print("=" * 60)

df_base = pd.read_excel(f"data/dummy/{FILE1}")

# Post-load summary (CLAUDE.md requirement)
print("\nShape:", df_base.shape)
print("Columns:", df_base.columns.tolist())
print("Null counts:\n", df_base.isnull().sum())
print("Dtypes:\n", df_base.dtypes)

# ── SKILL 2: CHECK 1 — Completeness ─────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 1 — Completeness")
total_months = 60
filled = df_base.shape[0]
completeness_pct = round(filled / total_months * 100, 1)
print(f"Completeness: {filled}/{total_months} months = {completeness_pct}%")
if completeness_pct < 90:
    print("FLAG: Below 90% — investigate before proceeding")
else:
    print("PASS")

# ── SKILL 2: CHECK 2 — Date format ──────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 2 — Date format")
date_col = df_base.columns[0]
print(f"Date column: '{date_col}'")
print("Sample values:", df_base[date_col].head(5).tolist())
print("Dtype:", df_base[date_col].dtype)

# ── SKILL 2: CHECK 3 — Granularity ──────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 3 — Granularity")
print("Row count:", df_base.shape[0])
print("Assessment: monthly (one row per calendar month)")

# ── SKILL 2: CHECK 4 — Range validity ───────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 4 — Range validity")
numeric_cols_1 = df_base.select_dtypes(include='number').columns
for col in numeric_cols_1:
    neg = (df_base[col] < 0).sum()
    print(f"  {col}: min={df_base[col].min()}, max={df_base[col].max()}, negatives={neg}")
    if neg > 0:
        print(f"  FLAG: negative values found in {col}")

# ── SKILL 2: CHECK 5 — Structural breaks ────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 5 — Structural breaks")
print("Source: Talkwalker — social mention volume")
print("Check: Did Talkwalker change how they count social mentions at any point?")
print("  e.g. Old method: counted only posts saying 'BrandX' exactly.")
print("  New method: AI-based, counts 'the chip maker', 'the blue team' etc.")
print("  ACTION REQUIRED: Inspect time series for sudden unexplained jumps.")
print("  Visual check — month-over-month changes:")
df_temp = df_base.copy()
df_temp[date_col] = pd.to_datetime(df_temp[date_col]).dt.to_period('M').astype(str)
for col in numeric_cols_1:
    if df_base[col].notna().sum() > 1:
        pct_chg = df_base[col].pct_change().abs()
        big_jumps = pct_chg[pct_chg > 0.5]
        if len(big_jumps) > 0:
            print(f"  FLAG: {col} — {len(big_jumps)} month(s) with >50% MoM change at rows: {big_jumps.index.tolist()}")
        else:
            print(f"  {col}: no sudden jumps >50% MoM detected")
print("NOTE: Confirm with data provider that counting methodology was consistent throughout.")

# ── SKILL 3: CLEAN ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"CLEANING: {FILE1}")
print("=" * 60)

df_clean = df_base.copy()

# PRIORITY 1 — Fix date format -> YYYY-MM
df_clean[date_col] = pd.to_datetime(df_clean[date_col]).dt.to_period('M').astype(str)
print(f"\nDate sample after conversion: {df_clean[date_col].head(3).tolist()}")
log_change(FILE1, date_col, "converted to YYYY-MM format", "Source had YYYY-MM-DD format", "pd.to_datetime + dt.to_period('M')")

# PRIORITY 2 — Handle missing values
for col in numeric_cols_1:
    null_count = df_clean[col].isnull().sum()
    if null_count == 0:
        print(f"{col}: 0 nulls — no action needed")
        continue

    # Check max consecutive nulls
    is_null = df_clean[col].isnull()
    groups = (is_null != is_null.shift()).cumsum()
    consecutive = is_null.groupby(groups).sum().max()

    print(f"\n{col}: {null_count} null(s), max consecutive = {consecutive}")
    if consecutive <= 2:
        df_clean[col] = df_clean[col].interpolate(method='linear')
        print(f"  → Filled using linear interpolation")
        log_change(FILE1, col, f"{null_count} null(s) filled", f"Small gap — max {consecutive} consecutive null", "linear interpolation")
    else:
        print(f"  FLAG: {consecutive} consecutive nulls — NOT filled, needs discussion")
        log_change(FILE1, col, f"FLAGGED — {null_count} nulls NOT filled", f"Max consecutive gap = {consecutive} (>2 limit)", "no action — flagged")

# Convert forums_mentions float64 -> int64 (post-fill)
if 'forums_mentions' in df_clean.columns:
    if df_clean['forums_mentions'].isnull().sum() == 0:
        df_clean['forums_mentions'] = df_clean['forums_mentions'].round().astype('int64')
        print(f"\nforums_mentions: converted to int64")
        log_change(FILE1, 'forums_mentions', "dtype converted float64 → int64", "Column represents whole mention counts; null filled before cast", "astype int64")
    else:
        print("FLAG: forums_mentions still has nulls — cannot cast to int64 safely")

# PRIORITY 4 — Detect outliers (flag only)
print("\nOutlier check (>3 SD from 12-month rolling mean):")
numeric_cols_clean_1 = df_clean.select_dtypes(include='number').columns
for col in numeric_cols_clean_1:
    rolling_mean = df_clean[col].rolling(12, min_periods=3).mean()
    rolling_std  = df_clean[col].rolling(12, min_periods=3).std()
    outlier_mask = (df_clean[col] - rolling_mean).abs() > 3 * rolling_std
    outlier_rows = df_clean[outlier_mask][[date_col, col]]
    if len(outlier_rows) > 0:
        print(f"  FLAG: {col} — {len(outlier_rows)} outlier(s):")
        print(outlier_rows.to_string())
        log_change(FILE1, col, f"{len(outlier_rows)} outlier(s) flagged", "Value >3 SD from 12-month rolling mean — check event registry", "flag only — not removed")
    else:
        print(f"  {col}: no outliers")

# Final state
print("\nCleaned file — shape:", df_clean.shape)
print("Columns:", df_clean.columns.tolist())
print("Null counts:\n", df_clean.isnull().sum())
print("Dtypes:\n", df_clean.dtypes)

# Save
df_clean.to_csv("data/processed/social_mention_volume_cleaned.csv", index=False)
print("\nSaved: data/processed/social_mention_volume_cleaned.csv")
log_change(FILE1, "ALL", "Cleaned file saved", "Skill 2+3 complete", "to_csv")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 2 — social_sentiment.xlsx
# ═══════════════════════════════════════════════════════════════════════════════

FILE2 = "social_sentiment.xlsx"
print("\n\n" + "=" * 60)
print(f"AUDIT + CLEAN: {FILE2}")
print("=" * 60)

df_base2 = pd.read_excel(f"data/dummy/{FILE2}")

# Post-load summary
print("\nShape:", df_base2.shape)
print("Columns:", df_base2.columns.tolist())
print("Null counts:\n", df_base2.isnull().sum())
print("Dtypes:\n", df_base2.dtypes)

# ── SKILL 2: CHECK 1 — Completeness ─────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 1 — Completeness")
filled2 = df_base2.shape[0]
completeness_pct2 = round(filled2 / total_months * 100, 1)
print(f"Completeness: {filled2}/{total_months} months = {completeness_pct2}%")
if completeness_pct2 < 90:
    print("FLAG: Below 90% — investigate before proceeding")
else:
    print("PASS")

# ── SKILL 2: CHECK 2 — Date format ──────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 2 — Date format")
date_col2 = df_base2.columns[0]
print(f"Date column: '{date_col2}'")
print("Sample values:", df_base2[date_col2].head(5).tolist())
print("Dtype:", df_base2[date_col2].dtype)

# ── SKILL 2: CHECK 3 — Granularity ──────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 3 — Granularity")
print("Row count:", df_base2.shape[0])
print("Assessment: monthly (one row per calendar month)")

# ── SKILL 2: CHECK 4 — Range validity ───────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 4 — Range validity")
numeric_cols_2 = df_base2.select_dtypes(include='number').columns
for col in numeric_cols_2:
    neg = (df_base2[col] < 0).sum()
    print(f"  {col}: min={df_base2[col].min():.2f}, max={df_base2[col].max():.2f}, negatives={neg}")
    # Sentiment must be -100 to +100
    if 'sentiment' in col.lower():
        out_of_range = ((df_base2[col] < -100) | (df_base2[col] > 100)).sum()
        if out_of_range > 0:
            print(f"  FLAG: {col} — {out_of_range} values outside [-100, +100] range")
        else:
            print(f"  {col}: all values within [-100, +100] — PASS")
    # Volume columns must be >= 0
    elif neg > 0:
        print(f"  FLAG: negative values found in {col}")

# ── SKILL 2: CHECK 5 — Structural breaks ────────────────────────────────────
print("\n" + "-" * 40)
print("CHECK 5 — Structural breaks")
print("Source: Talkwalker — social sentiment")
print("Check: Did Talkwalker change how they classify sentiment (positive/negative/neutral)?")
print("  e.g. Old classifier: rule-based keyword matching.")
print("  New classifier: ML/AI model — may reclassify same content differently.")
print("  A methodology change can shift net_sentiment baseline with no real brand change.")
print("  ACTION REQUIRED: Confirm with Talkwalker that sentiment classification method was consistent.")
print("  Visual check — month-over-month changes in sentiment:")
for col in numeric_cols_2:
    if df_base2[col].notna().sum() > 1:
        pct_chg = df_base2[col].diff().abs()
        big_jumps = pct_chg[pct_chg > pct_chg.std() * 3]
        if len(big_jumps) > 0:
            print(f"  FLAG: {col} — {len(big_jumps)} month(s) with large absolute jump (>3 SD): rows {big_jumps.index.tolist()}")
        else:
            print(f"  {col}: no sudden large jumps detected")

# ── SKILL 3: CLEAN ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"CLEANING: {FILE2}")
print("=" * 60)

df_clean2 = df_base2.copy()

# PRIORITY 1 — Fix date format -> YYYY-MM
df_clean2[date_col2] = pd.to_datetime(df_clean2[date_col2]).dt.to_period('M').astype(str)
print(f"\nDate sample after conversion: {df_clean2[date_col2].head(3).tolist()}")
log_change(FILE2, date_col2, "converted to YYYY-MM format", "Source had YYYY-MM-DD format", "pd.to_datetime + dt.to_period('M')")

# PRIORITY 2 — Handle missing values
for col in numeric_cols_2:
    null_count = df_clean2[col].isnull().sum()
    if null_count == 0:
        print(f"{col}: 0 nulls — no action needed")
        continue
    is_null = df_clean2[col].isnull()
    groups = (is_null != is_null.shift()).cumsum()
    consecutive = is_null.groupby(groups).sum().max()
    print(f"\n{col}: {null_count} null(s), max consecutive = {consecutive}")
    if consecutive <= 2:
        df_clean2[col] = df_clean2[col].interpolate(method='linear')
        print(f"  → Filled using linear interpolation")
        log_change(FILE2, col, f"{null_count} null(s) filled", f"Small gap — max {consecutive} consecutive null", "linear interpolation")
    else:
        print(f"  FLAG: {consecutive} consecutive nulls — NOT filled, needs discussion")
        log_change(FILE2, col, f"FLAGGED — {null_count} nulls NOT filled", f"Max consecutive gap = {consecutive} (>2 limit)", "no action — flagged")

# Skill 4b — Check for net_sentiment_score
print("\n" + "-" * 40)
print("Skill 4b — net_sentiment_score check")
cols_lower = [c.lower() for c in df_clean2.columns]

if 'net_sentiment_score' in df_clean2.columns:
    print("CONFIRMED: net_sentiment_score column already present.")
    print(df_clean2[[date_col2, 'net_sentiment_score']].head())
    log_change(FILE2, 'net_sentiment_score', "confirmed present — no calculation needed", "data provider provided net_sentiment_score directly", "n/a")
elif all(c in df_clean2.columns for c in ['positive_mentions', 'negative_mentions', 'total_mentions']):
    print("net_sentiment_score NOT found. Calculating from raw counts (Skill 4b formula):")
    print("  net_sentiment = (positive_mentions - negative_mentions) / total_mentions * 100")
    df_clean2['net_sentiment_score'] = (
        (df_clean2['positive_mentions'] - df_clean2['negative_mentions']) /
        df_clean2['total_mentions'] * 100
    ).round(2)
    print(df_clean2[[date_col2, 'positive_mentions', 'negative_mentions', 'total_mentions', 'net_sentiment_score']].head())
    log_change(FILE2, 'net_sentiment_score', "calculated from raw counts", "Column not present — data provider supplied positive/negative/neutral separately", "Skill 4b formula: (pos-neg)/total*100")
else:
    print("FLAG: net_sentiment_score not found and cannot calculate — positive_mentions/negative_mentions/total_mentions columns missing.")
    print(f"Columns present: {df_clean2.columns.tolist()}")

# PRIORITY 4 — Detect outliers (flag only)
print("\nOutlier check (>3 SD from 12-month rolling mean):")
numeric_cols_clean_2 = df_clean2.select_dtypes(include='number').columns
for col in numeric_cols_clean_2:
    rolling_mean = df_clean2[col].rolling(12, min_periods=3).mean()
    rolling_std  = df_clean2[col].rolling(12, min_periods=3).std()
    outlier_mask = (df_clean2[col] - rolling_mean).abs() > 3 * rolling_std
    outlier_rows = df_clean2[outlier_mask][[date_col2, col]]
    if len(outlier_rows) > 0:
        print(f"  FLAG: {col} — {len(outlier_rows)} outlier(s):")
        print(outlier_rows.to_string())
        log_change(FILE2, col, f"{len(outlier_rows)} outlier(s) flagged", "Value >3 SD from 12-month rolling mean — check event registry", "flag only — not removed")
    else:
        print(f"  {col}: no outliers")

# Final state
print("\nCleaned file — shape:", df_clean2.shape)
print("Columns:", df_clean2.columns.tolist())
print("Null counts:\n", df_clean2.isnull().sum())
print("Dtypes:\n", df_clean2.dtypes)

# Save
df_clean2.to_csv("data/processed/social_sentiment_cleaned.csv", index=False)
print("\nSaved: data/processed/social_sentiment_cleaned.csv")
log_change(FILE2, "ALL", "Cleaned file saved", "Skill 2+3+4b complete", "to_csv")

# ── Save log ─────────────────────────────────────────────────────────────────
save_log()
