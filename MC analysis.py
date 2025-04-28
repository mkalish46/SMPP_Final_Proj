import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =========================
# File Path
# =========================
merged_file = r"C:\Users\kalis\OneDrive\Documents\CFA2\Merged_AllData.xlsx"
if not os.path.exists(merged_file):
    raise FileNotFoundError(f"Merged file not found: {merged_file}")

# =========================
# Load Merged Data and Preprocess
# =========================
df = pd.read_excel(merged_file)
df.sort_values('Year', inplace=True)
df.columns = df.columns.str.strip()  # Normalize column names

# =========================
# FCFF Regression Model
# =========================
fcff_df = df[['Year', 'CFS_Free Cash Flow']].dropna()
X_fcff = sm.add_constant(fcff_df['Year'])
y_fcff = fcff_df['CFS_Free Cash Flow']
fcff_model = sm.OLS(y_fcff, X_fcff).fit()
resid_std = fcff_model.resid.std()

const = fcff_model.params['const']
coef = fcff_model.params['Year']

print("FCFF Regression Model:")
print(fcff_model.summary())

# =========================
# Determine Shares Outstanding
# =========================
if 'IS_Basic Shares Outstanding' in df.columns:
    share_count = df.loc[df['Year'] == df['Year'].max(), 'IS_Basic Shares Outstanding'].iloc[0]
else:
    share_count = 330  # per the latest simulation result
print(f"Using {share_count:.0f} shares outstanding for per-share valuation.")

# =========================
# Set Simulation Parameters
# =========================
n_sim = 10000  # Number of Monte Carlo iterations
forecast_horizon = 10  # Forecast period in years
current_year = int(df['Year'].max())

# For a tighter interval, reduce uncertainty:
error_factor = 0.5  # Scale down forecast residual uncertainty (0 < factor < 1)


# Update discount rate simulation: use lower standard deviation (e.g., 1% instead of 2%)
def simulate_discount_rate():
    dr = np.random.normal(loc=0.10, scale=0.01)  # mean = 10%, std = 1%
    return np.clip(dr, 0.05, 0.20)


def simulate_terminal_growth(dr):
    g = np.random.normal(loc=0.03, scale=0.01)  # mean = 3%, std = 1%
    if g >= dr:
        g = dr * 0.9
    return g


# =========================
# Monte Carlo Simulation Loop
# =========================
simulated_intrinsic_values = []

for i in range(n_sim):
    # Simulate inputs
    dr = simulate_discount_rate()
    g_terminal = simulate_terminal_growth(dr)

    fcff_forecasts = []
    for t in range(1, forecast_horizon + 1):
        year_t = current_year + t
        pred_fcff = const + coef * year_t
        # Scale the error term to reduce overall uncertainty
        error = np.random.normal(loc=0, scale=error_factor * resid_std)
        fcff_forecasts.append(pred_fcff + error)

    # Discount the forecasted FCFF
    pv_fcff = sum([
        fcff_forecasts[t - 1] / ((1 + dr) ** t)
        for t in range(1, forecast_horizon + 1)
    ])

    # Terminal value calculation
    fcff_last = fcff_forecasts[-1]
    terminal_value = (fcff_last * (1 + g_terminal)) / (dr - g_terminal)
    pv_terminal = terminal_value / ((1 + dr) ** forecast_horizon)

    total_value = pv_fcff + pv_terminal
    intrinsic_value_per_share = total_value / share_count
    simulated_intrinsic_values.append(intrinsic_value_per_share)

simulated_intrinsic_values = np.array(simulated_intrinsic_values)

# =========================
# Analyze Simulation Results
# =========================
mean_value = np.mean(simulated_intrinsic_values)
median_value = np.median(simulated_intrinsic_values)
ci_lower = np.percentile(simulated_intrinsic_values, 2.5)
ci_upper = np.percentile(simulated_intrinsic_values, 97.5)

print("\nMonte Carlo Simulation Results:")
print(f"Simulated mean intrinsic value per share: ${mean_value:,.2f}")
print(f"Simulated median intrinsic value per share: ${median_value:,.2f}")
print(f"95% Confidence Interval: ${ci_lower:,.2f} to ${ci_upper:,.2f}")

plt.figure(figsize=(10, 6))
plt.hist(simulated_intrinsic_values, bins=50, color='skyblue', edgecolor='black')
plt.axvline(ci_lower, color='red', linestyle='--', label=f"2.5th Percentile (${ci_lower:,.2f})")
plt.axvline(ci_upper, color='red', linestyle='--', label=f"97.5th Percentile (${ci_upper:,.2f})")
plt.axvline(mean_value, color='green', linestyle='-', label=f"Mean (${mean_value:,.2f})")
plt.xlabel("Intrinsic Value per Share ($)")
plt.ylabel("Frequency")
plt.title("Distribution of Simulated Intrinsic Stock Values")
plt.legend()
plt.show()

# =========================
# Make Recommendations
# =========================
current_market_price = 121  # Update with current market price if needed

print(f"\nCurrent Market Price: ${current_market_price:,.2f}")
if ci_lower > current_market_price:
    recommendation = "Buy"
elif ci_upper < current_market_price:
    recommendation = "Sell"
else:
    recommendation = "Hold"

print(f"Recommendation: {recommendation}")
print(f"Based on a 95% confidence interval of simulated intrinsic values from ${ci_lower:,.2f} to ${ci_upper:,.2f}.")
