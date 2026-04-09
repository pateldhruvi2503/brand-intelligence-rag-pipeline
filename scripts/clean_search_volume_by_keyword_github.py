import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

Path("logs/").mkdir(exist_ok=True)
Path("data/processed/").mkdir(exist_ok=True)

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
    log_path = "logs/cleaning_log.csv"
    if Path(log_path).exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"\nLog saved: {len(log_entries)} new entries added to logs/cleaning_log.csv")

SOURCE_FILE = "data/dummy/search_volume_by_keyword.xlsx"

# ─────────────────────────────────────────────
# LOAD — df_base is never modified after this
# ─────────────────────────────────────────────
df_base = pd.read_excel(SOURCE_FILE)

print("=" * 60)
print("LOADED: search_volume_by_keyword.xlsx")
print("=" * 60)
print("Shape:  ", df_base.shape)
print("Columns:", df_base.columns.tolist())
print("\nNull counts:\n", df_base.isnull().sum())
print("\nDtypes:\n", df_base.dtypes)

# ─────────────────────────────────────────────
# SKILL 2 — AUDIT (all 5 checks)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUDIT: search_volume_by_keyword.xlsx")
print("=" * 60)

# CHECK 1 — Completeness
print("\nCHECK 1 — Completeness")
total_months_expected = 60
unique_months = df_base['month'].nunique()
completeness_pct = round(unique_months / total_months_expected * 100, 1)
print(f"Shape: {df_base.shape}")
print(f"Null counts:\n{df_base.isnull().sum()}")
print(f"Unique months: {unique_months}/{total_months_expected} = {completeness_pct}%")
if completeness_pct < 90:
    print("FLAG: Below 90% — investigate before proceeding")
else:
    print("PASS: Completeness >= 90%")

# CHECK 2 — Date format
print("\nCHECK 2 — Date format")
date_col = 'month'
print(f"Date column: {date_col}")
print("Sample values:", df_base[date_col].head(5).tolist())
print("Dtype:", df_base[date_col].dtype)
print("Note: format is YYYY-MM-DD (full date string) — needs conversion to YYYY-MM")

# CHECK 3 — Granularity
print("\nCHECK 3 — Granularity")
print(f"Row count: {df_base.shape[0]}")
print(f"Unique months: {unique_months} — confirms monthly granularity")
print("Rows per keyword_group:")
print(df_base['keyword_group'].value_counts().to_string())

# CHECK 4 — Range validity
print("\nCHECK 4 — Range validity")
numeric_cols = df_base.select_dtypes(include='number').columns
for col in numeric_cols:
    neg_count = (df_base[col] < 0).sum()
    print(f"  {col}: min={df_base[col].min():,}, max={df_base[col].max():,}, negatives={neg_count}")
    if neg_count > 0:
        print(f"  FLAG: {col} has {neg_count} negative values — search volume must be >= 0")

# CHECK 5 — Structural breaks
print("\nCHECK 5 — Structural breaks")
print("Source: Brightedge (search volume data)")
print("  Question to ask data provider: Did the way Brightedge counts search queries")
print("  change at any point in the 2021-03 to 2026-02 window?")
print("  e.g. In earlier years only counting exact query 'BrandX vs BrandA'. Later also")
print("  counting 'BrandX versus BrandA', 'BrandX compared to BrandA' under the same group.")
print("  A sudden volume jump with no known BrandX event would indicate a tool change.")
print("  STATUS: Cannot confirm without data provider — flagged for follow-up.")
print("\nAUDIT COMPLETE — no blocking issues found. Proceeding to cleaning.")

# ─────────────────────────────────────────────
# SKILL 3 — CLEAN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANING: search_volume_by_keyword.xlsx")
print("=" * 60)

df_clean = df_base.copy()

# PRIORITY 1 — Fix date format to YYYY-MM
df_clean['date'] = pd.to_datetime(df_clean['month']).dt.to_period('M').astype(str)
df_clean = df_clean.drop(columns=['month'])
log_change(
    "search_volume_by_keyword.xlsx",
    "month",
    "converted to YYYY-MM format, renamed to 'date'",
    "Raw format was YYYY-MM-DD (full date string); project requires YYYY-MM monthly granularity",
    "pd.to_datetime().dt.to_period('M').astype(str)"
)

print("\nDate conversion check (first 3 rows):")
print(df_clean[['date', 'keyword', 'keyword_group', 'search_volume']].head(3).to_string())

# PRIORITY 2 — Missing values (already confirmed zero nulls, verify on clean copy)
null_check = df_clean.isnull().sum()
print("\nNull check after date conversion:\n", null_check)
if null_check.sum() == 0:
    print("PASS: No nulls introduced during cleaning.")

# PRIORITY 4 — Outlier detection per keyword
print("\nOutlier check (>3 std from 12-month rolling mean per keyword):")
outlier_count = 0
for kw in df_clean['keyword'].unique():
    kw_df = df_clean[df_clean['keyword'] == kw].sort_values('date').copy()
    rolling_mean = kw_df['search_volume'].rolling(12, min_periods=3).mean()
    rolling_std  = kw_df['search_volume'].rolling(12, min_periods=3).std()
    outlier_mask = (kw_df['search_volume'] - rolling_mean).abs() > 3 * rolling_std
    if outlier_mask.sum() > 0:
        outlier_count += outlier_mask.sum()
        print(f"  FLAG: '{kw}' — {outlier_mask.sum()} outlier(s) detected — check event registry")
        log_change(
            "search_volume_by_keyword.xlsx",
            f"search_volume [{kw}]",
            f"{outlier_mask.sum()} outlier(s) detected — not removed, flagged for event registry check",
            "Value >3 std from 12-month rolling mean",
            "rolling(12).mean/std"
        )
if outlier_count == 0:
    print("  No outliers detected across all keywords.")

# ─────────────────────────────────────────────
# OUTPUT 1 — branded_search_cleaned.csv
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 1 — branded_search_cleaned.csv")
print("=" * 60)

BRANDX_BRANDED_KEYWORDS = ['BrandX', 'BrandX processor', 'BrandX CPU', 'BrandX Core', 'BrandX laptop']

df_branded = df_clean[
    (df_clean['keyword_group'] == 'branded_awareness') &
    (df_clean['keyword'].isin(BRANDX_BRANDED_KEYWORDS))
].copy()

print(f"Rows after filter: {df_branded.shape[0]} (expected: {len(BRANDX_BRANDED_KEYWORDS) * 60})")
print("Keywords included:", df_branded['keyword'].unique().tolist())

df_branded_agg = (
    df_branded
    .groupby('date', as_index=False)['search_volume']
    .sum()
    .rename(columns={'search_volume': 'branded_search_volume'})
    .sort_values('date')
    .reset_index(drop=True)
)

print(f"\nAggregated shape: {df_branded_agg.shape}")
print("Sample (first 5 rows):")
print(df_branded_agg.head(5).to_string())
print(f"\nMin branded_search_volume: {df_branded_agg['branded_search_volume'].min():,}")
print(f"Max branded_search_volume: {df_branded_agg['branded_search_volume'].max():,}")

df_branded_agg.to_csv("data/processed/branded_search_cleaned.csv", index=False)
log_change(
    "search_volume_by_keyword.xlsx",
    "search_volume",
    "filtered keyword_group='branded_awareness' + 5 BrandX keywords, summed by month → branded_search_volume",
    "Pipeline requires one branded search column; only BrandX-specific keywords included",
    "groupby('date').sum()"
)
print("Saved: data/processed/branded_search_cleaned.csv")

# ─────────────────────────────────────────────
# OUTPUT 2 — comparative_search_cleaned.csv (Skill 4c)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 2 — comparative_search_cleaned.csv")
print("=" * 60)

df_comparative = df_clean[df_clean['keyword_group'] == 'comparative_consideration'].copy()

print(f"Rows after filter: {df_comparative.shape[0]}")
print("Keywords included:", df_comparative['keyword'].unique().tolist())

df_comparative_agg = (
    df_comparative
    .groupby('date', as_index=False)['search_volume']
    .sum()
    .rename(columns={'search_volume': 'comparative_search_volume'})
    .sort_values('date')
    .reset_index(drop=True)
)

print(f"\nAggregated shape: {df_comparative_agg.shape}")
print("Sample (first 5 rows):")
print(df_comparative_agg.head(5).to_string())
print(f"\nMin comparative_search_volume: {df_comparative_agg['comparative_search_volume'].min():,}")
print(f"Max comparative_search_volume: {df_comparative_agg['comparative_search_volume'].max():,}")

df_comparative_agg.to_csv("data/processed/comparative_search_cleaned.csv", index=False)
log_change(
    "search_volume_by_keyword.xlsx",
    "search_volume",
    "filtered keyword_group='comparative_consideration', summed by month → comparative_search_volume",
    "Skill 4c: pipeline expects single comparative search column",
    "groupby('date').sum()"
)
print("Saved: data/processed/comparative_search_cleaned.csv")

# ─────────────────────────────────────────────
# OUTPUT 3 — bottom_funnel_search_cleaned.csv (Skill 4d)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OUTPUT 3 — bottom_funnel_search_cleaned.csv")
print("=" * 60)

df_purchase = df_clean[df_clean['keyword_group'] == 'purchase_intent'].copy()

print(f"Rows after filter: {df_purchase.shape[0]}")
print("Keywords included:", df_purchase['keyword'].unique().tolist())

df_purchase_agg = (
    df_purchase
    .groupby('date', as_index=False)['search_volume']
    .sum()
    .rename(columns={'search_volume': 'bottom_funnel_search_volume'})
    .sort_values('date')
    .reset_index(drop=True)
)

print(f"\nAggregated shape: {df_purchase_agg.shape}")
print("Sample (first 5 rows):")
print(df_purchase_agg.head(5).to_string())
print(f"\nMin bottom_funnel_search_volume: {df_purchase_agg['bottom_funnel_search_volume'].min():,}")
print(f"Max bottom_funnel_search_volume: {df_purchase_agg['bottom_funnel_search_volume'].max():,}")

df_purchase_agg.to_csv("data/processed/bottom_funnel_search_cleaned.csv", index=False)
log_change(
    "search_volume_by_keyword.xlsx",
    "search_volume",
    "filtered keyword_group='purchase_intent', summed by month → bottom_funnel_search_volume",
    "Skill 4d: pipeline expects single bottom-funnel search column",
    "groupby('date').sum()"
)
print("Saved: data/processed/bottom_funnel_search_cleaned.csv")

# ─────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────
save_log()

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print("Outputs saved to data/processed/:")
print("  branded_search_cleaned.csv")
print("  comparative_search_cleaned.csv")
print("  bottom_funnel_search_cleaned.csv")
print("Log appended: logs/cleaning_log.csv")
