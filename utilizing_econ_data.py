import os
import pandas as pd
import re

# =========================
# User-defined File Paths
# =========================
financial_file = r"C:\Users\kalis\OneDrive\Documents\CFA2\Cleaned_Financials_Output_WideOnly.xlsx"
macro_file     = r"C:\Users\kalis\Downloads\Macroeconomic Data FRED.xls"

# Check that both files exist
if not os.path.exists(financial_file):
    raise FileNotFoundError(f"Financial file not found: {financial_file}")
if not os.path.exists(macro_file):
    raise FileNotFoundError(f"Macro file not found: {macro_file}")

# ========================================
# Helper function: extract year from a string
# ========================================
def extract_year(col_val):
    """
    Extracts a 4-digit year from strings like "SEP '24".
    Assumes the 2-digit year is in the 2000s.
    """
    match = re.search(r"(\d{2})$", str(col_val).strip())
    return 2000 + int(match.group(1)) if match else None

# ========================================
# PART 1: LOAD & MERGE FINANCIAL DATA
# ========================================

financial_sheets = ['BS - Wide', 'IS - Wide', 'CFS - Wide', 'Ratio Analysis - Wide']
financial_merged = None  # To hold combined financial data

for sheet in financial_sheets:
    try:
        df = pd.read_excel(financial_file, sheet_name=sheet)
    except Exception as e:
        print(f"Error reading sheet {sheet}: {e}")
        continue

    # The first column is used for year information (e.g., "SEP '24")
    year_col = df.columns[0]
    df["Year"] = df[year_col].apply(extract_year)

    # Rename all other columns with a prefix derived from the sheet name
    sheet_prefix = sheet.split()[0]  # For example, "BS" from "BS - Wide"
    renamed_columns = {col: f"{sheet_prefix}_{col}" for col in df.columns if col != "Year"}
    df.rename(columns=renamed_columns, inplace=True)

    # Optionally, drop the original year column (now prefixed) if desired
    orig_year_col = f"{sheet_prefix}_{year_col}"
    if orig_year_col in df.columns:
        df.drop(columns=[orig_year_col], inplace=True)

    # Ensure 'Year' is present and of type int
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # Merge the current financial data with the accumulated data
    if financial_merged is None:
        financial_merged = df.copy()
    else:
        financial_merged = pd.merge(financial_merged, df, on="Year", how="inner")

print("✅ Combined Financial Data Preview:")
print(financial_merged.head())

# ========================================
# PART 2: LOAD & MERGE MACROECONOMIC DATA
# ========================================

# We will process all sheets in the macro file.
macro_xls = pd.ExcelFile(macro_file)
macro_merged = None  # To accumulate all macro data

for sheet in macro_xls.sheet_names:
    try:
        # Read sheet starting at row 11 (so skip first 10 rows)
        df_macro = pd.read_excel(macro_file, sheet_name=sheet, skiprows=10)
    except Exception as e:
        print(f"Error reading macro sheet {sheet}: {e}")
        continue

    # Rename the first column to "observation_date" if not already set
    date_col = df_macro.columns[0]
    df_macro.rename(columns={date_col: "observation_date"}, inplace=True)

    # Convert observation_date to datetime and drop rows with invalid dates
    df_macro["observation_date"] = pd.to_datetime(df_macro["observation_date"], errors="coerce")
    df_macro.dropna(subset=["observation_date"], inplace=True)

    # Set the observation_date as the index to facilitate resampling
    df_macro.set_index("observation_date", inplace=True)

    # Resample the monthly data to annual data using the mean (or use .last() for December values)
    df_macro_annual = df_macro.resample("A").mean()

    # Create a "Year" column from the datetime index (the index now represents the end-of-year dates)
    df_macro_annual["Year"] = df_macro_annual.index.year

    # Select only numeric columns and the "Year" column.
    # Remove "Year" from numeric columns if it appears to avoid duplication.
    numeric_cols = df_macro_annual.select_dtypes(include=["number"]).columns.tolist()
    if "Year" in numeric_cols:
        numeric_cols.remove("Year")
    df_macro_annual = df_macro_annual[numeric_cols + ["Year"]]

    # Rename indicator columns using a prefix based on the sheet name
    indicator_cols = [col for col in df_macro_annual.columns if col != "Year"]
    prefix = sheet.replace(" ", "_")
    df_macro_annual.rename(columns={col: f"{prefix}_{col}" for col in indicator_cols}, inplace=True)

    # Merge this sheet's macro data with the cumulative macro_merged dataset
    if macro_merged is None:
        macro_merged = df_macro_annual.copy()
    else:
        macro_merged = pd.merge(macro_merged, df_macro_annual, on="Year", how="outer")

print("✅ Combined Macro Data Preview:")
print(macro_merged.sort_values("Year").head(10))

# ========================================
# PART 3: MERGE FINANCIAL & MACRO DATA INTO FINAL DATASET
# ========================================
final_merged = pd.merge(financial_merged, macro_merged, on="Year", how="inner")

print("✅ Final Merged Dataset Preview:")
print(final_merged.head(10))

# ========================================
# PART 4: SAVE THE FINAL DATASET
# ========================================
output_path = r"C:\Users\kalis\OneDrive\Documents\CFA2\Merged_AllData.xlsx"
final_merged.to_excel(output_path, index=False)
print(f"✅ Final merged dataset saved to {output_path}")
