import pandas as pd
import numpy as np

# --- FILE NAMES ---
stocks_file = "omxs30_close_prices_raw.csv"
index_file = "omxs30_index_close.csv"
snapshot_file = "omxs30_snapshot_esg_fundamentals.csv"

# ---------- HELPER FUNCTIONS ----------
def clean_number(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()

    # Remove currency and percentage markers.
    for token in ["SEK", "$", "%"]:
        x = x.replace(token, "")

    # Remove non-breaking spaces and surrounding whitespace.
    x = x.replace("\xa0", " ").strip()

    # Remove spaces used as thousands separators.
    x = x.replace(" ", "")

    # Convert decimal commas to decimal points.
    x = x.replace(",", ".")

    if x == "" or x == "-":
        return np.nan

    try:
        return float(x)
    except ValueError:
        return np.nan


# ---------- 1. LOAD STOCK PRICE FILE ----------
stocks = pd.read_csv(stocks_file, sep=";")
stocks.rename(columns={stocks.columns[0]: "Date"}, inplace=True)
stocks["Date"] = pd.to_datetime(stocks["Date"], format="%d-%b-%Y", errors="coerce")

for col in stocks.columns[1:]:
    stocks[col] = stocks[col].apply(clean_number)

# Convert to long format.
stocks_long = stocks.melt(id_vars="Date", var_name="ticker", value_name="close")
stocks_long = stocks_long.dropna(subset=["Date", "close"])
stocks_long = stocks_long.sort_values(["ticker", "Date"])

# Calculate return features.
stocks_long["ret_1d"] = stocks_long.groupby("ticker")["close"].pct_change()
stocks_long["ret_1m"] = stocks_long.groupby("ticker")["close"].pct_change(21)
stocks_long["ret_3m"] = stocks_long.groupby("ticker")["close"].pct_change(63)
stocks_long["ret_fwd_1m"] = stocks_long.groupby("ticker")["close"].shift(-21) / stocks_long["close"] - 1


# ---------- 2. LOAD INDEX FILE ----------
index_df = pd.read_csv(index_file, sep=";")
index_df.columns = ["Date", "omxs30_close"]
index_df["Date"] = pd.to_datetime(index_df["Date"], format="%d-%b-%Y", errors="coerce")
index_df["omxs30_close"] = index_df["omxs30_close"].apply(clean_number)
index_df = index_df.dropna(subset=["Date", "omxs30_close"]).sort_values("Date")

index_df["omxs30_ret_1d"] = index_df["omxs30_close"].pct_change()
index_df["omxs30_ret_1m"] = index_df["omxs30_close"].pct_change(21)
index_df["omxs30_ret_fwd_1m"] = index_df["omxs30_close"].shift(-21) / index_df["omxs30_close"] - 1


# ---------- 3. LOAD SNAPSHOT FILE ----------
# The CSV already has a single header row. Some column names contain embedded
# line breaks, so we normalize them after reading.
snapshot = pd.read_csv(snapshot_file, sep=";", encoding="utf-8-sig")
snapshot.columns = [col.replace("\n", " ").strip() for col in snapshot.columns]

# Remove the index summary row and the totals row.
snapshot = snapshot[snapshot["Identifier (RIC)"] != "Totals (30)"].copy()
snapshot = snapshot[snapshot["Identifier (RIC)"] != ".OMXS30"].copy()

# Rename key columns when present.
rename_map = {
    "Identifier (RIC)": "ticker",
    "Company Name": "company_name",
    "ESG Score (FY0) (Σ=None)": "esg_score",
    "ESG Combined Score (FY0) (Σ=None)": "esgc_score",
    "Price Close (Σ=None)": "price_close",
    "P/E (LTM) - IBES Actual (Σ=None)": "pe",
    "Price To Book (Σ=None)": "pb",
    "ROE (Σ=None)": "roe",
    "Total Debt Percentage of Total Equity (FY0) (Σ=None)": "debt_equity",
    "ICB Sector name": "sector"
}
snapshot.rename(columns={k: v for k, v in rename_map.items() if k in snapshot.columns}, inplace=True)

# Clean numeric snapshot columns.
for col in ["esg_score", "esgc_score", "price_close", "pe", "pb", "roe", "debt_equity"]:
    if col in snapshot.columns:
        snapshot[col] = snapshot[col].apply(clean_number)

snapshot = snapshot[["ticker"] + [c for c in ["company_name", "sector", "esg_score", "esgc_score", "pe", "pb", "roe", "debt_equity"] if c in snapshot.columns]]


# ---------- 4. MERGE DATASETS ----------
df = stocks_long.merge(index_df, on="Date", how="left")
df = df.merge(snapshot, on="ticker", how="left")

# Create the excess return target.
df["excess_ret_fwd_1m"] = df["ret_fwd_1m"] - df["omxs30_ret_fwd_1m"]

print(df.head())
print(df.columns)

# Save the cleaned outputs.
df.to_csv("omxs30_model_dataset.csv", index=False)
snapshot.to_csv("omxs30_snapshot_clean.csv", index=False)
index_df.to_csv("omxs30_index_clean.csv", index=False)
stocks_long.to_csv("omxs30_prices_long.csv", index=False)
