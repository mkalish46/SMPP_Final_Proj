import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json

# Use a non-interactive backend so figures are saved and not displayed
plt.ioff()


# ---------- Helper Functions ----------

def extract_year(col_name):
    """
    Extract a four-digit year from a column value like "SEP '24".
    Assumes the 2-digit year is in the 2000s.
    """
    match = re.search(r"(\d{2})$", col_name.strip())
    if match:
        year = int(match.group(1))
        return 2000 + year
    return None


def compute_summary_stats(series):
    """Compute basic summary statistics for a pandas Series."""
    stats = {}
    stats['mean'] = series.mean()
    stats['std'] = series.std()
    stats['min'] = series.min()
    stats['max'] = series.max()
    stats['median'] = series.median()
    mode = series.mode()
    stats['mode'] = mode.iloc[0] if not mode.empty else np.nan
    return stats


def plot_trend_line(years, values, title, save_path=None):
    """
    Plot a time series with a fitted linear trend line.
    If the trend line computation fails (e.g., SVD does not converge),
    only the raw data is plotted.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(years, values, marker='o', label='Data')
    mask = ~np.isnan(values)
    if np.sum(mask) >= 2:
        try:
            coeffs = np.polyfit(np.array(years)[mask], np.array(values)[mask], 1)
            trend_line = np.poly1d(coeffs)(years)
            plt.plot(years, trend_line, color='red', linestyle='--', label='Trend line')
        except np.linalg.LinAlgError as e:
            print(f"Warning: polyfit failed for {title} with error: {e}")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()  # Close the figure to free memory


def normalize(text):
    """Convert text to lowercase and remove non-alphanumeric characters."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_metric_column(df, candidates, exact=False):
    """
    Search among the column headers (excluding the first column) for one
    that matches one of the candidate strings.
    If exact=True, perform an exact match (after normalizing) otherwise perform a substring match.
    Returns the column name if found; otherwise, returns None.
    """
    for col in df.columns[1:]:
        norm_col = normalize(str(col))
        for cand in candidates:
            norm_cand = normalize(cand)
            if exact:
                if norm_col == norm_cand:
                    return col
            else:
                if norm_cand in norm_col:
                    return col
    return None


def sanitize_filename(name):
    """Sanitize a string for use as a filename by replacing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


# ---------- Diagnostic: Print Available Column Headers ----------
input_file = 'C:/Users/kalis/OneDrive/Documents/CFA2/Cleaned_Financials_Output_WideOnly.xlsx'
for sheet in ['BS - Wide', 'IS - Wide', 'CFS - Wide', 'Ratio Analysis - Wide']:
    try:
        df_diag = pd.read_excel(input_file, sheet_name=sheet)
        print(f"\nSheet: {sheet}\nColumn headers:")
        print(df_diag.columns.tolist())
    except Exception as e:
        print(f"Error reading sheet {sheet}: {e}")

# ---------- Search Alternatives ----------
# For BS, we require an exact match for key metrics.
search_alternatives = {
    'BS': {
        'Total Assets': ['Total Assets'],  # Exact match only
        'Total Liabilities': ['Total Liabilities'],  # Exact match only
        "Total Shareholders' Equity": ["Total Shareholders' Equity"],  # Exact match only
        'BV/Share': ['BV/Share', 'Book Value']
    },
    'IS': {
        'Sales': ['Sales', 'Revenue'],
        'Net Income': ['Net Income', 'NI'],
        'EPS (basic)': ['EPS (basic)'],  # Only match "EPS (basic)"
        'EPS (diluted)': ['EPS (diluted)']  # Only match "EPS (diluted)"
    },
    'CFS': {
        'NI': ['NI', 'Net Income'],
        'D&A': ['D&A', 'Depreciation'],
        'Change in WC': ['Change in WC', 'Working Capital'],
        'Cap Exp': ['Cap Exp', 'Capital Expenditure'],
        'Free Cash Flow': ['Free Cash Flow']
    },
    'Ratio Analysis': {
        'Gross Margin': ['Gross Margin'],
        'Return on Equity': ['Return on Equity', 'ROE'],
        'Return on Invested Capital': ['Return on Invested Capital', 'ROIC'],
        'Debt to Equity Ratio': ['Debt to Equity Ratio', 'Debt/Equity'],
        'Current Ratio': ['Current Ratio'],
        'Quick Ratio': ['Quick Ratio'],
        'Dividend per share': ['Dividend per share', 'Dividends'],
        'Payout Ratio': ['Payout Ratio']
    }
}

# Specify which metrics should have growth computed.
growth_options = {
    'BS': ['Total Assets', "Total Shareholders' Equity"],
    'IS': ['Sales', 'Net Income', 'EPS (basic)', 'EPS (diluted)'],
    'CFS': [],  # Not computing growth for CFS items here
    'Ratio Analysis': ['Dividend per share']
}

# ---------- Output Paths ----------
output_plot_dir = 'C:/Users/kalis/OneDrive/Documents/CFA2/Plots'
os.makedirs(output_plot_dir, exist_ok=True)
summary_json_path = 'C:/Users/kalis/OneDrive/Documents/CFA2/summary_statistics.json'

# ---------- Main Processing ----------
summary_results = {}

with pd.ExcelFile(input_file) as xls:
    for tab_key in ['BS', 'IS', 'CFS', 'Ratio Analysis']:
        sheet_name = f"{tab_key} - Wide"
        print(f"\nProcessing sheet: {sheet_name}")
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
            continue

        # Assume the first column is the "Year" column.
        year_col_name = df.columns[0]
        years_raw = df[year_col_name].astype(str).tolist()
        years_extracted = [extract_year(y) for y in years_raw]
        time_index = pd.Series(years_extracted)

        summary_results[tab_key] = {}
        # For BS, use exact matching; for others, substring matching.
        exact_match = (tab_key == 'BS')

        # Loop over each desired metric for this tab.
        for metric, candidates in search_alternatives[tab_key].items():
            col_found = find_metric_column(df, candidates, exact=exact_match)
            if col_found is None:
                print(f"Metric '{metric}' not found in {sheet_name}.")
                continue

            # Extract the metric data from the found column.
            series_data = df[col_found]
            filtered = [(y, v) for y, v in zip(time_index, series_data) if y is not None and pd.notna(v)]
            if not filtered:
                print(f"No valid data for metric '{metric}' in {sheet_name}.")
                continue

            years, values = zip(*filtered)
            years = list(years)
            level_series = pd.Series(values, index=years).astype(float)

            # Compute summary statistics for level data.
            stats_level = compute_summary_stats(level_series)
            summary_results[tab_key][metric] = {'Level': stats_level}
            print(f"Summary for {tab_key} - {metric} (Level): {stats_level}")

            # Plot trend for level data.
            plot_title = f"{tab_key} - {metric} (Level)"
            safe_metric = sanitize_filename(metric)
            plot_save_path = os.path.join(output_plot_dir, f"{tab_key}_{safe_metric}_Level.png")
            plot_trend_line(years, level_series.values, plot_title, save_path=plot_save_path)

            # Compute and plot growth if requested.
            if metric in growth_options.get(tab_key, []):
                growth_series = level_series.pct_change() * 100
                growth_stats = compute_summary_stats(growth_series.dropna())
                summary_results[tab_key][metric]['Growth'] = growth_stats
                print(f"Growth Summary for {tab_key} - {metric}: {growth_stats}")
                plot_title_growth = f"{tab_key} - {metric} (Growth %)"
                plot_save_path_growth = os.path.join(output_plot_dir, f"{tab_key}_{safe_metric}_Growth.png")
                plot_trend_line(years, growth_series.values, plot_title_growth, save_path=plot_save_path_growth)

# Save all summary statistics to a JSON file.
with open(summary_json_path, 'w') as f:
    json.dump(summary_results, f, indent=4)

print("✅ Processing complete. Summary statistics saved and trend plots generated.")
