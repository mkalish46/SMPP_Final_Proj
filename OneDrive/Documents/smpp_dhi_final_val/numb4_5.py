import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ============================================================
# File Paths – Update these to your actual file locations.
# ============================================================
merged_file = r"C:\Users\kalis\OneDrive\Documents\CFA2\Merged_AllData.xlsx"

if not os.path.exists(merged_file):
    raise FileNotFoundError(f"Merged file not found: {merged_file}")

# ============================================================
# Load the merged dataset and normalize column names
# ============================================================
df = pd.read_excel(merged_file)
df.sort_values('Year', inplace=True)
# Strip extra whitespace from all column names
df.columns = df.columns.str.strip()

print("✅ Merged file loaded. Columns in the dataset:")
print(df.columns.tolist())

# ============================================================
# Compute Additional Variables (Growth Rates)
# ============================================================
# Revenue Growth: assuming revenue is in "IS_Sales"
if 'IS_Sales' in df.columns:
    df['Revenue_growth'] = df['IS_Sales'].pct_change() * 100
else:
    print("Column 'IS_Sales' not found; cannot compute revenue growth.")

# GDP Growth: Use the column that contains "GDP"
gdp_candidates = [col for col in df.columns if 'GDP' in col.upper() and 'GROWTH' not in col.upper()]
if len(gdp_candidates) > 0:
    gdp_col = gdp_candidates[0]
    df['GDP_growth'] = df[gdp_col].pct_change() * 100
    print(f"Using '{gdp_col}' as GDP measure.")
else:
    print("No GDP variable found; cannot compute GDP growth.")

# CPI Growth: Look for a column containing 'CPI'
cpi_candidates = [col for col in df.columns if 'CPI' in col.upper()]
if len(cpi_candidates) > 0:
    cpi_col = cpi_candidates[0]
    df['CPI_growth'] = df[cpi_col].pct_change() * 100
    print(f"Using '{cpi_col}' as CPI measure.")
else:
    print("No CPI variable found; cannot compute CPI growth.")

# Interest Rate: Explicitly use "10_Yr_Treasury_BOGZ1FL073161113Q"
if '10_Yr_Treasury_BOGZ1FL073161113Q' in df.columns:
    df['Interest_rate'] = df['10_Yr_Treasury_BOGZ1FL073161113Q']  # levels, not growth
    print("Using '10_Yr_Treasury_BOGZ1FL073161113Q' as Interest Rate measure.")
else:
    print("Interest Rate column '10_Yr_Treasury_BOGZ1FL073161113Q' not found.")

# Oil Prices: Look for a column containing 'OIL'
oil_candidates = [col for col in df.columns if 'OIL' in col.upper()]
if len(oil_candidates) > 0:
    oil_col = oil_candidates[0]
    df['Oil_price'] = df[oil_col]  # levels; optionally, you could compute growth
    print(f"Using '{oil_col}' as Oil Price measure.")
else:
    print("No Oil Price variable found.")

# ============================================================
# Correlation Analysis
# ============================================================
# Select variables of interest for correlation:
corr_vars = ['Revenue_growth', 'GDP_growth', 'CPI_growth', 'Interest_rate', 'Oil_price']
# Also include key firm-specific metrics (if available)
for candidate in ['IS_Net Income', 'BS_Total Assets', 'Ratio_Dividend per share', 'IS_EPS (basic)']:
    if candidate in df.columns:
        corr_vars.append(candidate)

df_corr = df[corr_vars].dropna()
corr_matrix = df_corr.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Selected Variables")
plt.show()

# ============================================================
# Regression Analysis: Revenue Growth vs. Macro Variables
# ============================================================
if 'Revenue_growth' in df_corr.columns:
    X = df_corr[['GDP_growth', 'CPI_growth', 'Interest_rate', 'Oil_price']]
    X = sm.add_constant(X)
    y = df_corr['Revenue_growth']
    rev_model = sm.OLS(y, X).fit()
    print("Regression Model: Revenue Growth vs. Macro Variables")
    print(rev_model.summary())
else:
    print("Revenue_growth not available for regression.")

# ============================================================
# Accounting Riddles – Testing Hypotheses with Regression Models
# ============================================================

# Riddle 1: “Cash and equivalents increases with increases in revenues.”
# Updated to use "BS_Cash Only" (found in your file) instead of "BS_Cash and equivalents"
if 'BS_Cash Only' in df.columns and 'IS_Sales' in df.columns:
    df['Sales_change'] = df['IS_Sales'].pct_change() * 100
    df['Cash_change'] = df['BS_Cash Only'].pct_change() * 100
    temp = df[['Sales_change', 'Cash_change']].dropna()
    reg1 = sm.OLS(temp['Cash_change'], sm.add_constant(temp['Sales_change'])).fit()
    print("\nRiddle 1: Cash and equivalents increases with increases in revenues:")
    print(reg1.summary())
else:
    print("Required columns for Riddle 1 not found.")

# Riddle 2: “Sales and profit margins fall with higher inflation.”
# Using "Ratio_Net Margin" as a proxy for profit margins.
if 'Ratio_Net Margin' in df.columns and 'CPI_growth' in df.columns:
    df['Margin_change'] = df['Ratio_Net Margin'].pct_change() * 100
    temp2 = df[['Margin_change', 'CPI_growth']].dropna()
    reg2 = sm.OLS(temp2['Margin_change'], sm.add_constant(temp2['CPI_growth'])).fit()
    print("\nRiddle 2: Sales and profit margins fall with higher inflation:")
    print(reg2.summary())
else:
    print("Required columns for Riddle 2 not found.")

# Riddle 3: “Return on Equity falls during recessions but rises with inflation.”
if 'Ratio_Return on Equity' in df.columns and 'GDP_growth' in df.columns and 'CPI_growth' in df.columns:
    temp3 = df[['Ratio_Return on Equity', 'GDP_growth', 'CPI_growth']].dropna()
    X_r3 = sm.add_constant(temp3[['GDP_growth', 'CPI_growth']])
    y_r3 = temp3['Ratio_Return on Equity']
    reg3 = sm.OLS(y_r3, X_r3).fit()
    print("\nRiddle 3: Return on Equity falls during recessions but rises with inflation:")
    print(reg3.summary())
else:
    print("Required columns for Riddle 3 not found.")

# Riddle 4: “Firms accumulate cash during periods of economic growth.”
if 'BS_Cash Only' in df.columns and 'GDP_growth' in df.columns:
    df['Cash_change'] = df['BS_Cash Only'].pct_change() * 100
    temp4 = df[['Cash_change', 'GDP_growth']].dropna()
    reg4 = sm.OLS(temp4['Cash_change'], sm.add_constant(temp4['GDP_growth'])).fit()
    print("\nRiddle 4: Firms accumulate cash during periods of economic growth:")
    print(reg4.summary())
else:
    print("Required columns for Riddle 4 not found.")

# Riddle 5: “Good managers generate higher net income even when the economy is in recession.”
if 'IS_Net Income' in df.columns and 'GDP_growth' in df.columns:
    df['NetIncome_change'] = df['IS_Net Income'].pct_change() * 100
    temp5 = df[['NetIncome_change', 'GDP_growth']].dropna()
    reg5 = sm.OLS(temp5['NetIncome_change'], sm.add_constant(temp5['GDP_growth'])).fit()
    print("\nRiddle 5: Good managers generate higher net income even when the economy is in recession:")
    print(reg5.summary())
else:
    print("Required columns for Riddle 5 not found.")

# Riddle 6: “Good managers grow their dividend when the economy is in recession by issuing debt.”
# Using "IS_Dividends per Share" as the dividend measure.
if 'IS_Dividends per Share' in df.columns and 'BS_Total Debt' in df.columns and 'GDP_growth' in df.columns:
    df['Dividends_change'] = df['IS_Dividends per Share'].pct_change() * 100
    df['Debt_change'] = df['BS_Total Debt'].pct_change() * 100
    temp6 = df[['Dividends_change', 'GDP_growth', 'Debt_change']].dropna()
    X_r6 = sm.add_constant(temp6[['GDP_growth', 'Debt_change']])
    y_r6 = temp6['Dividends_change']
    reg6 = sm.OLS(y_r6, X_r6).fit()
    print("\nRiddle 6: Good managers grow their dividend during recession by issuing debt:")
    print(reg6.summary())
else:
    print("Required columns for Riddle 6 not found.")

# Riddle 7: “Good managers reduce the debt to equity ratio during periods of economic growth.”
# Using "Ratio_Total Debt/Equity (%)" as our measure.
if 'Ratio_Total Debt/Equity (%)' in df.columns and 'GDP_growth' in df.columns:
    df['DebtEquity_change'] = df['Ratio_Total Debt/Equity (%)'].pct_change() * 100
    temp7 = df[['DebtEquity_change', 'GDP_growth']].dropna()
    reg7 = sm.OLS(temp7['DebtEquity_change'], sm.add_constant(temp7['GDP_growth'])).fit()
    print("\nRiddle 7: Good managers reduce the debt to equity ratio during periods of economic growth:")
    print(reg7.summary())
else:
    print("Required columns for Riddle 7 not found.")

# ============================================================
# Forecasting: Next 10 Years for Key Variables
# ============================================================
# We forecast the following key variables:
# • Dividend per share – from "Ratio_Dividend per share" (if available)
# • EPS – from "IS_EPS (basic)"
# • FCFF/share – from "CFS_Free Cash Flow"
# • FCFE/share – using FCFF as a proxy (update if you have a different measure)
last_year = int(df['Year'].max())
future_years = pd.DataFrame({'Year': range(last_year + 1, last_year + 11)})
future_X = sm.add_constant(future_years['Year'])

# Forecast Dividend per share (if available)
if 'Ratio_Dividend per share' in df.columns:
    div_df = df[['Year', 'Ratio_Dividend per share']].dropna()
    X_div = sm.add_constant(div_df['Year'])
    y_div = div_df['Ratio_Dividend per share']
    model_div = sm.OLS(y_div, X_div).fit()
    print("\nForecasting Model for Dividend per share:")
    print(model_div.summary())
    future_years['Forecast_Dividend_per_share'] = model_div.predict(future_X)
    print("\nDividend per share forecast for next 10 years:")
    print(future_years[['Year', 'Forecast_Dividend_per_share']])
else:
    print("Column for Dividend per share not found for forecasting.")

# Forecast EPS (from "IS_EPS (basic)")
if 'IS_EPS (basic)' in df.columns:
    eps_df = df[['Year', 'IS_EPS (basic)']].dropna()
    X_eps = sm.add_constant(eps_df['Year'])
    y_eps = eps_df['IS_EPS (basic)']
    model_eps = sm.OLS(y_eps, X_eps).fit()
    print("\nForecasting Model for EPS (basic):")
    print(model_eps.summary())
    future_years['Forecast_EPS'] = model_eps.predict(future_X)
    print("\nEPS forecast for next 10 years:")
    print(future_years[['Year', 'Forecast_EPS']])
else:
    print("Column for EPS (basic) not found for forecasting.")

# Forecast FCFF/share (from "CFS_Free Cash Flow")
if 'CFS_Free Cash Flow' in df.columns:
    fcff_df = df[['Year', 'CFS_Free Cash Flow']].dropna()
    X_fcff = sm.add_constant(fcff_df['Year'])
    y_fcff = fcff_df['CFS_Free Cash Flow']
    model_fcff = sm.OLS(y_fcff, X_fcff).fit()
    print("\nForecasting Model for FCFF (Free Cash Flow):")
    print(model_fcff.summary())
    future_years['Forecast_FCFF'] = model_fcff.predict(future_X)
    print("\nFCFF forecast for next 10 years:")
    print(future_years[['Year', 'Forecast_FCFF']])
else:
    print("Column for Free Cash Flow not found for forecasting FCFF.")

# Forecast FCFE/share – if not separately available, using FCFF as a proxy
if 'CFS_Free Cash Flow' in df.columns:
    fcfe_df = df[['Year', 'CFS_Free Cash Flow']].dropna()
    X_fcfe = sm.add_constant(fcfe_df['Year'])
    y_fcfe = fcfe_df['CFS_Free Cash Flow']
    model_fcfe = sm.OLS(y_fcfe, X_fcfe).fit()
    print("\nForecasting Model for FCFE (assumed similar to FCFF):")
    print(model_fcfe.summary())
    future_years['Forecast_FCFE'] = model_fcfe.predict(future_X)
    print("\nFCFE forecast for next 10 years:")
    print(future_years[['Year', 'Forecast_FCFE']])
else:
    print("Column for FCFE or equivalent not found for forecasting.")

# ============================================================
# End of Analysis
# ============================================================
