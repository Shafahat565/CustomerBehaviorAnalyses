# Generated from: 01_data_preprocessing.ipynb
# Converted at: 2026-07-18T17:09:32.591Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 01 — Data Preprocessing
# ### Phases 1–4: Raw Data Loading, Cleaning, Category Mapping & Customer-Level Feature Engineering
# 
# ---
# 
# **Project pipeline (this notebook is stage 1 of 5):**
# 
# | Stage | Notebook | Purpose |
# |---|---|---|
# | 1 | **`01_data_preprocessing.ipynb`** ⬅ *you are here* | Load raw data, validate schema, clean, engineer customer-level features |
# | 2 | `02_pca_lda.ipynb` | Dimensionality reduction (PCA / LDA) on the engineered features |
# | 3 | `03_classification.ipynb` | Classification modeling |
# | 4 | `04_regression.ipynb` | Regression modeling |
# | 5 | `05_qlearning_dqn.ipynb` | Reinforcement learning (Q-learning / DQN) |
# 
# **Dataset:** [Online Retail (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/352/online+retail) — a transnational e-commerce transaction log from a UK-based online retailer, covering **01-Dec-2010 to 09-Dec-2011** (~540,000 rows). Also mirrored on Kaggle as "Online Retail Dataset."
# 
# **This notebook does everything required *before* PCA/LDA, in this exact order:**
# 
# | Phase | What it does |
# |---|---|
# | **Phase 1** | Load the raw file, validate its schema with an automated test suite, profile data quality |
# | **Phase 2** | Apply the 5 mandatory cleaning steps, in order, each with hand-verifiable test cases |
# | **Phase 4** | Assign every product to one of 5 fixed categories via keyword matching (run *before* Phase 3 below — see note) |
# | **Phase 3** | Aggregate cleaned, categorized transactions into one row per customer (RFM + extras + category-spend-%) |
# 
# > ⚠️ **Note on ordering:** the source spec numbers these "Phase 2 → Phase 3 → Phase 4," but Phase 3's *Category Spend %* feature (step 3.4) requires every transaction to already have a `Category` label. That label is only produced in Phase 4. So this notebook **runs Phase 4 before finishing Phase 3**, while keeping each phase's original name/number so it's traceable back to the spec. Everything else stays in the mandated order.
# 
# > 💡 **How to use this notebook:** run cells top-to-bottom. Every code cell is preceded by markdown explaining *what* and *why*, and most are followed by a small, hand-verifiable **test case** (using tiny synthetic data, matching the official Txx.x test IDs) before the same logic is applied to the real dataset. This means you can trust each transformation is behaving correctly *before* seeing it run on 540,000 rows.


# ## 📦 Prerequisites
# 
# - Python 3.8+, `pandas`, `numpy`, `openpyxl` (for the Excel path)
# - Raw dataset at `../data/Online_Retail.xlsx` or `../data/Online_Retail.csv`
# 
# ### Raw data format reference (what we expect to receive)
# 
# The dataset arrives as **one flat transaction table** — every row is a single line item on an invoice, *not* a customer. This is the shape all of Phase 1–2's code assumes.
# 
# | Column | Type | Example | Notes |
# |---|---|---|---|
# | `InvoiceNo` | String | `536365` or `C536379` | `'C'` prefix = cancellation |
# | `StockCode` | String | `85123A` | Product code |
# | `Description` | String | `WHITE HANGING HEART T-LIGHT HOLDER` | Free-text product name, used for category mapping (Phase 4) |
# | `Quantity` | Integer | `6` or `-1` | Can be negative (returns/errors) |
# | `InvoiceDate` | Datetime | `2010-12-01 08:26:00` | Timestamp of transaction |
# | `UnitPrice` | Float | `2.55` or `0.00` | Can be zero (data errors) |
# | `CustomerID` | float/int (nullable) | `17850.0` or `NaN` | Missing for guest/unattributed orders |
# | `Country` | String | `United Kingdom` | Kept for reference, not used as a required feature |


import json
import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print(f"pandas version : {pd.__version__}")
print(f"numpy version  : {np.__version__}")

# ### A small helper: approximate-value checking
# 
# The official Phase 1 numbers (541,909 rows, 135,080 missing `CustomerID`) come from one specific mirror of the dataset and are checked **exactly**. But from Phase 2 onward, the spec itself says:
# 
# > *"Small differences (a few hundred rows) across dataset mirrors are normal. If you are off by tens of thousands of rows, re-check that you applied the 5 steps in order..."*
# 
# So instead of hard-failing on small mirror-to-mirror drift, we use a small helper that **passes within a tolerance** and only warns (rather than crashing the notebook) if a value is close-but-not-exact, while still catching genuinely wrong results.


def check_approx(label, actual, expected, tolerance, unit=""):
    '''Print PASS if actual is within `tolerance` of expected, else WARN.'''
    diff = abs(actual - expected)
    status = "PASS" if diff <= tolerance else "WARN"
    marker = "\u2705" if status == "PASS" else "\u26A0\uFE0F"
    print(f"{marker} [{status}] {label}: got {actual:,}{unit}, expected ~{expected:,}{unit} "
          f"(tolerance \u00b1{tolerance:,}{unit}, diff {diff:,}{unit})")
    return status == "PASS"

# ---
# # Phase 1 — Raw Data: Format and Loading
# 
# ## Step 1.1 & 1.2 — Loading the Raw Data
# 
# We try the **Excel (.xlsx)** file first (the original UCI distribution), falling back to **CSV** if it's not present. The CSV path uses `encoding="ISO-8859-1"` because this dataset contains non-ASCII characters (currency symbols, accented text) that the pandas UTF-8 default can't decode — Latin-1 matches the file's real original encoding and never raises decode errors.


# ==========================================
# 1.1 & 1.2: Raw Data Loading
# ==========================================
print("=== PHASE 1: LOADING RAW DATA ===")

excel_path = "../data/Online_Retail.xlsx"
csv_path = "../data/Online_Retail.csv"

try:
    print(f"Attempting to read raw data from: {excel_path}...")
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    print(f"Excel file not found. Attempting to load CSV instead from: {csv_path}...")
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

print(f"SUCCESS: Loaded dataset shape: {df.shape}\n")

# ### 🔍 Reading the output
# `df.shape` should read **`(541909, 8)`**. If you get a `FileNotFoundError` for both paths, the raw file isn't at `../data/` yet.
# 
# ## Step 1.3 — Automated Test Suite (T1.1 – T1.3)
# 
# | Test | Checks | Why it matters |
# |---|---|---|
# | **T1.1** | Exact row/column count (541,909 × 8) | Confirms we loaded the full, standard file. |
# | **T1.2** | Column names match the canonical schema (with auto-repair for known mirror naming, e.g. `Invoice`→`InvoiceNo`) | Every later notebook references these exact column names. |
# | **T1.3** | `InvoiceDate` parses as real `datetime64`, not a string | All Recency/date-based logic later depends on this. |


# ==========================================
# 1.3: Automated Test Suite (T1.1 to T1.3)
# ==========================================
print("=== RUNNING PHASE 1 TEST SUITE ===")

# --- Test T1.1: Row and Column Count Check ---
expected_rows = 541909
expected_cols = 8
t1_1_passed = df.shape[0] == expected_rows and df.shape[1] == expected_cols

print(f"[T1.1] Shape Check: {df.shape}")
if t1_1_passed:
    print("      \U0001F449 PASS: Dimensions match expectation (exactly 541,909 rows, 8 columns)")
else:
    print(f"      \u274C FAIL: Row count is {df.shape[0]} (Expected: {expected_rows}) or "
          f"Column count is {df.shape[1]} (Expected: {expected_cols})")

# --- Test T1.2: Column Name Alignment Check ---
required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                     'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
actual_columns = df.columns.tolist()

column_mapping = {
    'Invoice': 'InvoiceNo',
    'Price': 'UnitPrice',
    'Customer ID': 'CustomerID'
}
if any(col in actual_columns for col in column_mapping.keys()):
    print("      \u26A0\uFE0F Warning: Non-standard column headers detected. Auto-renaming columns...")
    df = df.rename(columns=column_mapping)
    actual_columns = df.columns.tolist()

t1_2_passed = actual_columns == required_columns

print(f"[T1.2] Columns Check: {actual_columns}")
if t1_2_passed:
    print("      \U0001F449 PASS: Column names conform to expected schema.")
else:
    print(f"      \u274C FAIL: Column mismatch! Expected {required_columns}")

# --- Test T1.3: Datetime Type Parse Verification ---
if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

t1_3_passed = pd.api.types.is_datetime64_any_dtype(df['InvoiceDate'])

print(f"[T1.3] InvoiceDate Dtype: {df['InvoiceDate'].dtype}")
if t1_3_passed:
    print("      \U0001F449 PASS: Datetime recognized properly as datetime64[ns].")
else:
    print("      \u274C FAIL: InvoiceDate is not a datetime column.")

# ## Raw Data Quality Profiling — *Before* Any Cleaning
# 
# We baseline nulls, ranges, and business-meaningful anomalies now, so every cleaning decision in Phase 2 can be justified against a documented "before" state.
# 
# | Anomaly | Real-world cause |
# |---|---|
# | `CustomerID` missing | Guest checkout / unattributed order |
# | `Description` missing | Manual adjustment or write-off entry |
# | `Quantity <= 0` | Return, refund, or stock adjustment |
# | `UnitPrice <= 0` | Free sample, goodwill credit, bookkeeping line |
# | `InvoiceNo` starts with `"C"` | Cancelled order |


# ==========================================
# Raw Data Quality Profiling (Before Cleaning)
# ==========================================
print("\n=== RAW DATA QUALITY REPORT (BEFORE CLEANING) ===")

print("\n--- 1. Schema & Data Types ---")
print(df.dtypes)

print("\n--- 2. Missing Value Profiling ---")
null_summary = df.isnull().sum()
for col in df.columns:
    null_count = null_summary[col]
    null_pct = (null_count / len(df)) * 100
    print(f"Column '{col}': {null_count:,} missing values ({null_pct:.2f}%)")

print("\n--- 3. Descriptives (Quantitative Ranges) ---")
print(df[['Quantity', 'UnitPrice']].describe())

print("\n--- 4. Data Anomaly Quick Summary ---")
negative_qty = (df['Quantity'] <= 0).sum()
zero_price = (df['UnitPrice'] <= 0).sum()
cancellations = df['InvoiceNo'].astype(str).str.startswith('C').sum()

print(f"Negative/Zero Quantities: {negative_qty:,} rows")
print(f"Zero/Negative Unit Prices: {zero_price:,} rows")
print(f"Canceled Orders ('C' prefix): {cancellations:,} rows")

# ## ✅ Phase 1 Checkpoint
# 
# Before continuing to Phase 2, confirm these **exact** reference values (official UCI mirror):


# ==========================================
# Phase 1 Final Checkpoint: Assert Reference Values
# ==========================================
assert df.shape == (541909, 8), f"Shape mismatch: got {df.shape}, expected (541909, 8)"
assert t1_1_passed and t1_2_passed and t1_3_passed, "One or more T1.x tests did not pass."

customer_id_missing = df['CustomerID'].isnull().sum()
customer_id_missing_pct = (customer_id_missing / len(df)) * 100

assert customer_id_missing == 135080, (
    f"CustomerID missing count mismatch: got {customer_id_missing:,}, expected 135,080"
)

raw_row_count = len(df)  # remember for the funnel summary at the very end

print("=== PHASE 1 CHECKPOINT PASSED ===")
print(f"Loaded Shape        : {df.shape}")
print(f"Test T1.1           : \U0001F449 PASS")
print(f"Test T1.2           : \U0001F449 PASS")
print(f"Test T1.3           : \U0001F449 PASS")
print(f"CustomerID missing  : {customer_id_missing:,} missing values (~{customer_id_missing_pct:.2f}%)")

# ---
# # Phase 2 — Data Cleaning: Step-by-Step
# 
# The 5 steps below **must be applied in exactly this order** — each one changes the population the next step operates on. For every step we first run a **tiny, hand-verifiable test** on synthetic data (matching the official test IDs), then apply the identical logic to the real `df`.


# ## Step 2.1 — Drop rows where `CustomerID` is null
# 
# **Rule:** any transaction with no `CustomerID` cannot be attributed to a customer and must be removed.
# 
# ```python
# df = df.dropna(subset=["CustomerID"])
# ```


# --- Test T2.1a / T2.1b (hand-verifiable, synthetic) ---
sample_2_1a = pd.DataFrame({
    'CustomerID': [17850.0, np.nan],
    'InvoiceNo': ['536365', '536366']
})
result_2_1a = sample_2_1a.dropna(subset=['CustomerID'])
assert len(result_2_1a) == 1 and result_2_1a.iloc[0]['CustomerID'] == 17850.0
print("[T2.1a] PASS: Row A (CustomerID=17850.0) kept, Row B (NaN) removed.")

sample_2_1b = pd.DataFrame({'CustomerID': [1.0, np.nan, 2.0, np.nan, 3.0]})
result_2_1b = sample_2_1b.dropna(subset=['CustomerID'])
assert len(result_2_1b) == 3
print("[T2.1b] PASS: 5-row sample with 2 NaNs -> 3 rows remain.")

# --- Apply to the real dataset ---
rows_before_2_1 = len(df)
df = df.dropna(subset=["CustomerID"])
rows_after_2_1 = len(df)

print(f"\nBefore Step 2.1: {rows_before_2_1:,} rows")
print(f"After  Step 2.1: {rows_after_2_1:,} rows  (removed {rows_before_2_1 - rows_after_2_1:,})")
check_approx("Rows after Step 2.1", rows_after_2_1, 406829, tolerance=1000)

# CustomerID can now be safely cast to a clean integer type (no more NaN to force float)
df["CustomerID"] = df["CustomerID"].astype(int)
print(f"CustomerID dtype now: {df['CustomerID'].dtype}")

# ## Step 2.2 — Remove cancelled orders
# 
# **Rule:** any `InvoiceNo` starting with `"C"` is a cancellation/refund, not a genuine sale, and must be removed.
# 
# ```python
# df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
# ```


# --- Test T2.2a / T2.2b / T2.2c (hand-verifiable, synthetic) ---
assert str('C536379').startswith('C')       # T2.2a: this row would be DROPPED
assert not str('536365').startswith('C')    # T2.2b: this row would be KEPT
print("[T2.2a] PASS: 'C536379' correctly flagged as a cancellation (would be dropped).")
print("[T2.2b] PASS: '536365' correctly flagged as a genuine invoice (would be kept).")

sample_2_2c = pd.DataFrame({'InvoiceNo': ['536365', 'C536379', '536366', 'C536380']})
result_2_2c = sample_2_2c[~sample_2_2c['InvoiceNo'].astype(str).str.startswith('C')]
assert list(result_2_2c['InvoiceNo']) == ['536365', '536366']
print("[T2.2c] PASS: 4-row sample -> filtered result is exactly ['536365', '536366'].")

# --- Apply to the real dataset ---
rows_before_2_2 = len(df)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
rows_after_2_2 = len(df)

print(f"\nBefore Step 2.2: {rows_before_2_2:,} rows")
print(f"After  Step 2.2: {rows_after_2_2:,} rows  (removed {rows_before_2_2 - rows_after_2_2:,})")
check_approx("Rows after Step 2.2", rows_after_2_2, 397924, tolerance=1000)

# ## Step 2.3 — Remove non-positive `Quantity` or `UnitPrice`
# 
# **Rule:** `Quantity <= 0` or `UnitPrice <= 0` indicates a data error or adjustment entry, not a genuine sale.
# 
# ```python
# df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
# ```


# --- Test T2.3a / T2.3b / T2.3c (hand-verifiable, synthetic) ---
sample_2_3 = pd.DataFrame({
    'Quantity':  [6,    -1,   6],
    'UnitPrice': [2.55, 2.55, 0.00],
})
result_2_3 = sample_2_3[(sample_2_3['Quantity'] > 0) & (sample_2_3['UnitPrice'] > 0)]
assert len(result_2_3) == 1
assert result_2_3.iloc[0]['Quantity'] == 6 and result_2_3.iloc[0]['UnitPrice'] == 2.55
print("[T2.3a] PASS: Quantity=6, UnitPrice=2.55 -> KEPT.")
print("[T2.3b] PASS: Quantity=-1 -> DROPPED (Quantity>0 fails).")
print("[T2.3c] PASS: UnitPrice=0.00 -> DROPPED (UnitPrice>0 fails).")

# --- Apply to the real dataset ---
rows_before_2_3 = len(df)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
rows_after_2_3 = len(df)

print(f"\nBefore Step 2.3: {rows_before_2_3:,} rows")
print(f"After  Step 2.3: {rows_after_2_3:,} rows  (removed {rows_before_2_3 - rows_after_2_3:,})")
check_approx("Rows after Step 2.3", rows_after_2_3, 392692, tolerance=1000)

# ## Step 2.4 — Create `TotalPrice`
# 
# **Formula:** `TotalPrice = Quantity × UnitPrice`
# 
# ```python
# df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
# ```


# --- Test T2.4a / T2.4b (hand-verifiable, synthetic) ---
sample_2_4 = pd.DataFrame({'Quantity': [6, 1], 'UnitPrice': [2.55, 7.65]})
sample_2_4['TotalPrice'] = sample_2_4['Quantity'] * sample_2_4['UnitPrice']
assert round(sample_2_4.loc[0, 'TotalPrice'], 2) == 15.30
assert round(sample_2_4.loc[1, 'TotalPrice'], 2) == 7.65
print("[T2.4a] PASS: Quantity=6, UnitPrice=2.55 -> TotalPrice=15.30")
print("[T2.4b] PASS: Quantity=1, UnitPrice=7.65 -> TotalPrice=7.65")

# --- Apply to the real dataset ---
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print(f"\n'TotalPrice' column added. Current shape: {df.shape}")
print(df[["Quantity", "UnitPrice", "TotalPrice"]].head())

# ## Step 2.5 — Set the snapshot date
# 
# **Formula:** `snapshot_date = max(InvoiceDate across cleaned data) + 1 day`
# 
# This is the reference "today" that Recency (in Phase 3) is measured against — one day after the last transaction, so even a customer who purchased on the very last day gets a Recency of at least 1.
# 
# ```python
# snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
# ```


# --- Test T2.5a (hand-verifiable, synthetic) ---
sample_max_date = pd.Timestamp('2011-12-09 12:50:00')
sample_snapshot = sample_max_date + pd.Timedelta(days=1)
assert sample_snapshot == pd.Timestamp('2011-12-10 12:50:00')
print("[T2.5a] PASS: max=2011-12-09 12:50:00 -> snapshot=2011-12-10 12:50:00")

# --- Apply to the real dataset ---
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"\nLast InvoiceDate in cleaned data : {df['InvoiceDate'].max()}")
print(f"snapshot_date                    : {snapshot_date}")

# ## Phase 2 — Cumulative Before/After Summary
# 
# If your numbers differ from the reference by **a few hundred rows**, that's normal mirror-to-mirror variation. If you're off by **tens of thousands**, re-check that the 5 steps ran in order and that you didn't accidentally reload the unfiltered file partway through.


# ==========================================
# Phase 2 — Cumulative Row-Count Funnel
# ==========================================
funnel = pd.DataFrame({
    "Stage": [
        "Raw load",
        "After 2.1 (drop null CustomerID)",
        "After 2.2 (remove cancellations)",
        "After 2.3 (remove Qty/Price <= 0)",
        "After 2.4 (add TotalPrice)",
    ],
    "Row Count": [raw_row_count, rows_after_2_1, rows_after_2_2, rows_after_2_3, len(df)],
    "Columns": [8, 8, 8, 8, df.shape[1]],
})
print(funnel.to_string(index=False))

print("\n--- Reference (approximate) values from the spec ---")
check_approx("After 2.1", rows_after_2_1, 406829, tolerance=1000)
check_approx("After 2.2", rows_after_2_2, 397924, tolerance=1000)
check_approx("After 2.3", rows_after_2_3, 392692, tolerance=1000)

# ---
# # Phase 4 — Product Category Mapping
# 
# *(Run here, ahead of Phase 3's category-spend-% feature, since that feature needs every transaction already labeled with a `Category`.)*
# 
# ### Rule
# Assign every `StockCode` to exactly **one** of 5 fixed categories, using **case-insensitive keyword matching on `Description`**, in this **priority order** — first match wins. If nothing matches, assign `"Other"`.
# 
# | Priority | Category | Keywords |
# |---|---|---|
# | 1 | Homeware | `HOME`, `MUG`, `CANDLE`, `LANTERN`, `CUSHION` |
# | 2 | Stationery | `CARD`, `NOTEBOOK`, `PEN`, `PAPER`, `ENVELOPE` |
# | 3 | Gadgets | `LIGHT`, `CLOCK`, `BATTERY`, `ALARM` |
# | 4 | Decorations | `CHRISTMAS`, `DECORATION`, `BUNTING`, `GARLAND` |
# | 5 | Kitchenware | `BAKING`, `CAKE`, `TIN`, `JAR`, `BOWL` |
# | 6 | Other | *(fallback — nothing else matched)* |
# 
# > Do **not** invent your own categories — this fixed list resolves the "manual mapping" ambiguity, and we export it to `category_map.json` so the mapping is fully reproducible outside this notebook.
# 
# ### Why build a `stock_lookup` instead of applying the function per row?
# `assign_category()` only depends on `Description`, and the same `StockCode` repeats across many transaction rows. Building the lookup **once per unique `StockCode`** (a few thousand rows) and then merging it back onto the full transaction table (hundreds of thousands of rows) is dramatically faster than re-running the keyword search on every single transaction row.


CATEGORY_MAP = {
    "Homeware":    ["HOME", "MUG", "CANDLE", "LANTERN", "CUSHION"],
    "Stationery":  ["CARD", "NOTEBOOK", "PEN", "PAPER", "ENVELOPE"],
    "Gadgets":     ["LIGHT", "CLOCK", "BATTERY", "ALARM"],
    "Decorations": ["CHRISTMAS", "DECORATION", "BUNTING", "GARLAND"],
    "Kitchenware": ["BAKING", "CAKE", "TIN", "JAR", "BOWL"],
}

def assign_category(description: str) -> str:
    desc = str(description).upper()
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Other"

# ### 🔍 Test cases T4.1 – T4.6
# 
# These six examples are the official reference cases from the spec — each checks a specific priority-ordering edge case (e.g. a false-positive-looking substring match, two categories matching the same description, nothing matching at all).


# --- Test T4.1 - T4.6 (hand-verifiable) ---
tests_4 = [
    ("T4.1", "WHITE HANGING HEART T-LIGHT HOLDER", "Gadgets",
     "no Homeware/Stationery keyword hits; 'T-LIGHT' contains 'LIGHT' -> Gadgets"),
    ("T4.2", "RED CHRISTMAS TREE BAUBLE", "Decorations",
     "matches CHRISTMAS before any lower-priority keyword"),
    ("T4.3", "PACK OF 12 PAPER NAPKINS", "Stationery",
     "matches PAPER"),
    ("T4.4", "RECIPE BOX BAKING SET", "Kitchenware",
     "matches BAKING"),
    ("T4.5", "SILVER KEYRING", "Other",
     "no keyword from any category matches"),
    ("T4.6", "CANDLE HOLDER HOME DECOR", "Homeware",
     "matches both HOME and CANDLE, both = Homeware (priority 1) -> first matching category wins"),
]

all_passed = True
for test_id, description, expected, reason in tests_4:
    actual = assign_category(description)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        all_passed = False
    print(f"[{test_id}] {status}: '{description}' -> '{actual}' (expected '{expected}') | {reason}")

assert all_passed, "One or more Phase 4 category-assignment tests failed."
print("\nAll Phase 4 category-assignment tests PASSED.")

# ==========================================
# Apply category mapping to the real dataset
# ==========================================

# Build the lookup ONCE per unique StockCode (fast), not per transaction row
stock_lookup = (df[["StockCode", "Description"]].drop_duplicates("StockCode")
                 .assign(Category=lambda d: d["Description"].map(assign_category)))

df = df.merge(stock_lookup[["StockCode", "Category"]], on="StockCode", how="left")

# Saves directly to the project root folder (one level up from 'notebooks/')
category_map_path = os.path.join("..", "category_map.json")
with open(category_map_path, "w") as f:
    json.dump(CATEGORY_MAP, f, indent=2)

print(f"Unique StockCodes categorized : {len(stock_lookup):,}")
print(f"'Category' column added to df. Current shape: {df.shape}")
print("\nCategory distribution across all cleaned transactions:")
print(df["Category"].value_counts())
print(f"\ncategory_map.json written successfully to Project Root: {category_map_path}")

# ### 🔍 Reading the output
# - Every row in `df` now has a `Category` in `{Homeware, Stationery, Gadgets, Decorations, Kitchenware, Other}`.
# - A large `"Other"` bucket is expected and fine — this keyword list is intentionally small and fixed by the spec, not an exhaustive product taxonomy.
# - `category_map.json` now exists alongside this notebook (i.e. in the `notebooks/` folder) — check it into your repository so the category assignment is auditable and reproducible without re-running this cell.


# ---
# # Phase 3 — Customer-Level Feature Engineering
# 
# We now collapse the cleaned, categorized **transaction-level** table (1 row = 1 invoice line) into a **customer-level** table (1 row = 1 customer) using RFM (**R**ecency, **F**requency, **M**onetary) plus a few extras.
# 
# | Feature | Exact definition |
# |---|---|
# | **Recency (R)** | `snapshot_date − date of customer's last invoice`, in days |
# | **Frequency (F)** | Count of *distinct* `InvoiceNo` values for the customer |
# | **Monetary (M)** | Sum of `TotalPrice` across all the customer's transactions |
# | **Product Diversity** | Count of distinct `StockCode` values purchased |
# | **Avg. Spend / Transaction** | `Monetary ÷ Frequency` |
# | **Category Spend %** | For each of the 5 categories: `(spend in category ÷ Monetary) × 100` |
# 
# ### 🔍 Test cases T3.1 – T3.5 (hand-verifiable, synthetic customer "X")
# 
# We build one synthetic customer with a known purchase history and check that the *exact same aggregation code* we're about to run on the real data produces the documented expected values.


# --- Synthetic customer "X" matching the spec's worked example ---
sample_txn = pd.DataFrame({
    'CustomerID':  ['X', 'X', 'X'],
    'InvoiceDate': [pd.Timestamp('2011-11-20'), pd.Timestamp('2011-11-25'), pd.Timestamp('2011-11-28')],
    'InvoiceNo':   ['536365', '536390', '536365'],   # 536365 repeats -> nunique = 2
    'TotalPrice':  [15.30, 25.00, 9.90],              # sums to 50.20
    'StockCode':   ['85123A', '85123A', '71053'],      # 85123A repeats -> nunique = 2
})
sample_snapshot_date = pd.Timestamp('2011-12-10')

sample_agg = sample_txn.groupby('CustomerID').agg(
    last_purchase=('InvoiceDate', 'max'),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum'),
    ProductDiversity=('StockCode', 'nunique'),
)
sample_agg['Recency'] = (sample_snapshot_date - sample_agg['last_purchase']).dt.days
sample_agg['AvgSpendPerTxn'] = sample_agg['Monetary'] / sample_agg['Frequency']

assert sample_agg.loc['X', 'Recency'] == 12
print("[T3.1] PASS: last invoice 2011-11-28, snapshot 2011-12-10 -> Recency = 12 days")

assert sample_agg.loc['X', 'Frequency'] == 2
print("[T3.2] PASS: invoices {536365, 536390, 536365} -> Frequency (nunique) = 2")

assert round(sample_agg.loc['X', 'Monetary'], 2) == 50.20
print("[T3.3] PASS: TotalPrice values 15.30 + 25.00 + 9.90 -> Monetary = 50.20")

assert sample_agg.loc['X', 'ProductDiversity'] == 2
print("[T3.4] PASS: StockCodes {85123A, 85123A, 71053} -> ProductDiversity (nunique) = 2")

assert round(sample_agg.loc['X', 'AvgSpendPerTxn'], 2) == 25.10
print("[T3.5] PASS: Monetary=50.20 / Frequency=2 -> AvgSpendPerTxn = 25.10")


# ==============================================================================
# 🛠️ REFACTOR BLOCK: ISOLATE ROLLING TIME-WINDOWS (OCTOBER 1, 2011 CUTOFF)
# ==============================================================================
print("\n=== ISOLATING TEMPORAL SPLITS (OCTOBER 1, 2011 CUTOFF) ===")
cutoff_date = '2011-10-01'

# 1. Feature Window: Historical transactions used to compute features
feature_df = df[df['InvoiceDate'] < cutoff_date].copy()

# 2. Target Window: Future transactions used to isolate actual upcoming spend
target_df = df[df['InvoiceDate'] >= cutoff_date].copy()

print(f"Feature Window transactions (Before {cutoff_date}): {feature_df.shape[0]:,} rows")
print(f"Target Window transactions (After {cutoff_date}):  {target_df.shape[0]:,} rows")


# ==============================================================================
# 3.1: RFM + extras aggregation on the real dataset (Run strictly on Feature Window)
# ==============================================================================
print("\n=== PHASE 3: CUSTOMER FEATURE ENGINEERING (HISTORICAL WINDOW) ===")

# Reference snapshot date is locked exactly to the boundary edge of the history frame
snapshot_date = feature_df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Aggregate RFM features ONLY from historical data
agg = feature_df.groupby("CustomerID").agg(
    last_purchase=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    ProductDiversity=("StockCode", "nunique"),
)
agg["Recency"] = (snapshot_date - agg["last_purchase"]).dt.days
agg["AvgSpendPerTxn"] = agg["Monetary"] / agg["Frequency"]
agg = agg.drop(columns=["last_purchase"])

print(f"Customer-level table shape: {agg.shape}   (expected ~4,300 rows x 5 columns)")
check_approx("Unique customers", agg.shape[0], 4300, tolerance=600)
agg.head()


# ## 3.4 — Category Spend % feature (Synthetic Validation)
# 
# Reusing synthetic customer "X" (Monetary = 50.20 from T3.3), we split that spend across categories and confirm both the percentage formula and the "percentages sum to ~100%" invariant.

# --- Synthetic categorized transactions for customer "X" ---
# Homeware=20.00, Stationery=10.00, Other=20.20  -> sums to 50.20 (matches T3.3's Monetary)
sample_txn_cat = pd.DataFrame({
    'CustomerID': ['X', 'X', 'X'],
    'Category':   ['Homeware', 'Stationery', 'Other'],
    'TotalPrice': [20.00, 10.00, 20.20],
})
sample_monetary = pd.Series({'X': 50.20})

sample_cat_pivot = sample_txn_cat.pivot_table(
    index='CustomerID', columns='Category', values='TotalPrice', aggfunc='sum', fill_value=0
)
sample_cat_pct = sample_cat_pivot.div(sample_monetary, axis=0) * 100

assert round(sample_cat_pct.loc['X', 'Homeware'], 2) == 39.84
print("[T3.6] PASS: Homeware spend=20.00, Monetary=50.20 -> Homeware_Pct = 39.84%")

assert round(sample_cat_pct.sum(axis=1).loc['X'], 2) == 100.00
print("[T3.7] PASS: category percentages for customer X sum to 100.00% ")


# ==============================================================================
# 3.4: Category Spend % on the real dataset + final assembly
# ==============================================================================
cat_pivot = feature_df.pivot_table(index="CustomerID", columns="Category",
                           values="TotalPrice", aggfunc="sum", fill_value=0)
cat_pct = cat_pivot.div(agg["Monetary"], axis=0) * 100
cat_pct.columns = [c + "_Pct" for c in cat_pct.columns]

customer_features = agg.join(cat_pct).fillna(0)

print(f"Final customer_features shape: {customer_features.shape}")
print(f"Columns: {customer_features.columns.tolist()}")

pct_cols = [c for c in customer_features.columns if c.endswith("_Pct")]
pct_row_sums = customer_features[pct_cols].sum(axis=1)
print(f"\nCategory-%% columns sum per customer -- min: {pct_row_sums.min():.2f}, "
      f"max: {pct_row_sums.max():.2f} (should be ~100.00 for all customers, small float drift is normal)")

print(customer_features.head())
print(customer_features.describe())


# ==============================================================================
# 🛠️ COMPUTE TRUE TARGET VARIABLE & JOIN COHORTS
# ==============================================================================
print("\n=== CALCULATING TRUE NEXT QUARTER SPEND & ALIGNING MATRICES ===")

# Group the target window transactions by CustomerID to obtain real future spend values
true_target = (
    target_df.groupby('CustomerID')['TotalPrice']
    .sum()
    .reset_index()
    .rename(columns={'TotalPrice': 'NextQuarterSpend'})
)

# Left join features with target variables to retain customers who did not buy in Q4
final_dataset = pd.merge(customer_features, true_target, on='CustomerID', how='left')

# Fill NaN values with 0 for customers who generated zero spend in the future target quarter
final_dataset['NextQuarterSpend'] = final_dataset['NextQuarterSpend'].fillna(0)

print(f"Final Preprocessed Dataset Shape: {final_dataset.shape}")
print(final_dataset[['Frequency', 'Monetary', 'NextQuarterSpend']].head())


# ==============================================================================
# 🛠️ SAVE REFACTOR PIPELINE ARTIFACTS
# ==============================================================================
print("\n=== SAVING INTERIM PIPELINE ARTIFACTS ===")
output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)

# Save the unified feature matrix with targets included to disk
features_csv_path = os.path.join(output_dir, "customer_features.csv")
final_dataset.to_csv(features_csv_path, index=False)

print(f"✅ Saved Preprocessed Feature Dataset to: {features_csv_path}")


# ---
# # ✅ Phases 1–4 — Final Summary & Checkpoint
# 
# Before moving on to `02_pca_lda.ipynb`, review the consolidated funnel and feature summary below. `final_dataset` is the artifact the rest of the pipeline will build on.

print("=" * 60)
print("PHASES 1-4 COMPLETE - PIPELINE SUMMARY")
print("=" * 60)

print("\n--- Row-count funnel (transaction level) ---")
print(funnel.to_string(index=False))

print("\n--- Category mapping ---")
print(f"category_map.json written with {len(CATEGORY_MAP)} fixed categories (+ 'Other' fallback)")
print(df["Category"].value_counts())

print("\n--- Customer-level feature table ---")
print(f"final_dataset shape : {final_dataset.shape}")
print(f"columns             : {final_dataset.columns.tolist()}")

print("\n--- Sanity checks ---")
assert final_dataset.isnull().sum().sum() == 0, "final_dataset contains unexpected NaNs"
print("\u2705 No missing values in final_dataset")

print("\nPreprocessing Stage Complete! Ready for 02_pca_lda.ipynb execution.")