# Importing necessary libraries
import streamlit as st # Web app framework
import plotly.express as px  #For interactive plots (not used yet in this snippet)
import numpy as np # Numerical computations
import plotly.graph_objects as go # For more customized interactive plots
import math # Math utilities
from scipy.stats import norm # For normal distribution functions used in BSM
import yfinance as yf # Yahoo Finance API to fetch historical stock data
import seaborn as sns # For heatmaps
import matplotlib.pyplot as plt # For plotting with matplotlib


# Streamlit app configuration
st.set_page_config(page_title="Options Visualizer", layout="wide")
st.title("📈 Real-Time Option Pricing & Greeks Dashboard")


# ------------------------------
# Functions for computing Greeks
# ------------------------------

# Black-Scholes-Merton Delta (Δ)
def BSMd(S, K, T, r, vol, call = True ):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    delta_c = norm.cdf(d1)
    delta_p = -norm.cdf(-d1)

    if call == True :
        return delta_c
    else:
        return delta_p

# Gamma (Γ): Second derivative w.r.t. underlying price
def BSMg(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    gamma_c = norm.pdf(d1) / (S * vol * np.sqrt(T))
    gamma_p = norm.pdf(d1) / (S * vol * math.sqrt(T))
    
    if call == True :
        return gamma_c
    else:
        return gamma_p

# Vega (ν): Sensitivity to volatility (per 1% change)
def BSMv(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    vega_c = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change
    vega_p = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% change
    
    if call == True :
        return vega_c
    else:
        return vega_p

# Theta (Θ): Sensitivity to time decay (per day)
def BSMt(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    theta_c = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_p = (-S * norm.pdf(d1) * vol / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    if call == True :
        return theta_c
    else:
        return theta_p

# Rho (ρ): Sensitivity to interest rates (per 1% change)
def BSMr(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    rho_c = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_p = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    if call == True :
        return rho_c
    else:
        return rho_p

# ------------------------------
# Helper function to plot a Greek value over time
# ------------------------------

def plot_greek(data, greek_name, color="skyblue", dot_color="gray"):
    fig, ax = plt.subplots(figsize=(6, 4))

    y_vals = data[greek_name]
    x_vals = data.index

    # Plot Greek
    ax.plot(x_vals, y_vals, color=color, label=greek_name[:-5])

    # Mark current value with bullet & dotted line
    ax.scatter(x_vals[-1], y_vals.iloc[-1], color=dot_color, s=50, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.vlines(x_vals[-1], 0, y_vals.iloc[-1], linestyle="dotted", color=dot_color, linewidth=1)

    # Axis bounds (auto-fit within Greek value range)
    y_margin = (y_vals.max() - y_vals.min()) * 0.1
    if y_margin > 0:
        ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig

# ------------------------------
# Black-Scholes-Merton Price Formula for European Options
# ------------------------------

def BSM_EU (S, K, T, r, vol, call = True) :
    
    d1 = (math.log(S/K) + (r + 0.5 * vol **2) * T ) / (vol * math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    
    price_EUcall = S * norm.cdf(d1) - K* math.exp(-r*T)*norm.cdf(d2)  
    price_EUput =  K* math.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)
    
    if call == True:
        return price_EUcall
    else:
        return price_EUput


# ------------------------------
# Heatmap computation for call prices
# ------------------------------

def compute_heatmap_call(S_vals, vol_vals, K, T, r):
    prices = np.zeros((len(vol_vals), len(S_vals)))
    for i, sigma in enumerate(vol_vals):
        for j, S in enumerate(S_vals):
            prices[i, j] = BSM_EU (S, K, T, r, sigma, True)
    return prices


# ------------------------------
# Heatmap computation for put prices
# ------------------------------

def compute_heatmap_put(S_vals, vol_vals, K, T, r):
    prices = np.zeros((len(vol_vals), len(S_vals)))
    for i, sigma in enumerate(vol_vals):
        for j, S in enumerate(S_vals):
            prices[i, j] = BSM_EU (S, K, T, r, sigma, False)
    return prices


# ------------------------------
# American put pricing using Monte Carlo simulation with early exercise (not used in the app).
# ------------------------------

def bsm_american_put(S0, K, T, r, vol, steps=50, paths=10000):
    dt = T / steps
    discount = np.exp(-r * dt)
    np.random.seed(0)

    # Simulate asset price paths using GBM
    Z = np.random.randn(paths, steps)
    S = np.zeros((paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z[:, t - 1])

    # Calculate payoff at all steps (American put)
    payoff = np.maximum(K - S, 0)

    # Start with terminal cashflow
    cashflow = payoff[:, -1]

    # Backward induction for early exercise decision
    for t in range(steps - 1, 0, -1):
        in_the_money = payoff[:, t] > 0
        X = S[in_the_money, t]
        Y = cashflow[in_the_money] * discount

        if len(X) == 0:
            continue

        # Polynomial regression (2nd degree)
        A = np.vstack([np.ones_like(X), X, X**2]).T
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2

        # Exercise decision
        exercise = payoff[in_the_money, t] > continuation
        cashflow[in_the_money] = np.where(exercise, payoff[in_the_money, t], cashflow[in_the_money] * discount)

    # Final discounted expected value
    price = np.mean(cashflow) * np.exp(-r * dt)
    return price


# ------------------------------
# Fetch 1-year historical stock data using yfinance
# ------------------------------

def get_price_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    return data


# ------------------------------
# streamlit webapp
# ------------------------------

with st.sidebar :
    
    st.title("🌐 Black-Scholes Model Inputs and Assumptions")
    st.write("`Created by:`")
    linkedin_url = "www.linkedin.com/in/ethan-blondin-10432a203"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Ethan BLONDIN`</a>', unsafe_allow_html=True)
    # Dropdown menu
    
    assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "^GSPC", "^IXIC", "^STOXX50E", "SAP", "ASML", "SIE.DE", "SU.PA", "ALV.DE"]  
    selected_asset = st.selectbox("Select Stock or Index", assets)
    
    
    
    data = get_price_data(selected_asset)
    data["Close"] = data["Close"].astype(float)

    
    last_close =  float(data["Close"].iloc[-1].item())
    
    # Placeholder: set up volatility and time range
    vol_range = st.slider("Aribtrary Volatility", 0.0, 1.0, 0.15, step = 0.05)
    
    # Decide which currency to use
    if selected_asset in assets[0:8]:  # US-based assets
        currency = "$"
    else:  # European assets
        currency = "€"
    
    spot_price = st.metric(label = f"Latest Spot Price on {data.index[-1].date()}", value = f"{last_close:.2f} {currency}")


    # Function to avoid auto-reinitialization of the strike price after modification
    def get_float(value):
        return float(value)
    
    # Option parameters

    lc_dummy = get_float(last_close)
    

    K = st.number_input("Strike Price", value = lc_dummy, min_value = 0.00, key ="strike_price") # mettre initial value à close et permettre un changement arbitraire ainsi qu'un bouton réinitialiser.
    T = st.number_input("Time to Maturity (in years)", value=0.5, step=0.25) 
    r = st.number_input("Risk-free rate", value=0.03)

  
# Visualizing the heatmaps

S_vals = np.linspace(last_close * 0.8 , last_close * 1.2, 9)
vol_vals = np.linspace(max(vol_range-0.2,0.01),min(vol_range+0.2, 1), 9)


col1, col2 = st.columns([1,1], gap="small")


with col1:

    price_matrix_call = compute_heatmap_call(S_vals, vol_vals, K, T, r)
    
    # Plotting Call Price Heatmap
    st.subheader("Call Option Price Heatmap")
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(price_matrix_call, xticklabels=np.round(S_vals, 2), yticklabels=np.round(vol_vals, 2), annot=True, fmt=".2f", cmap="Spectral", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')

    fig_call



with col2:

    price_matrix_put = compute_heatmap_put(S_vals, vol_vals, K, T, r)

    # Plotting Put Price Heatmap
    st.subheader("Put Option Price Heatmap")
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(price_matrix_put, xticklabels=np.round(S_vals, 2), yticklabels=np.round(vol_vals, 2), annot=True, fmt=".2f", cmap="Spectral", ax=ax_put)
    ax_put.set_title('Put')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    fig_put


# Plotting the option greeks
col3, col4, col5, col6, col7 = st.columns(5)

data["Delta_Call"] = data["Close"].apply(lambda S: BSMd(S, K, T, r, vol_range, True))
with col3:
    st.subheader("Delta")
    st.pyplot(plot_greek(data, "Delta_Call", color="#1E90FF"))

data["Gamma_Call"] = data["Close"].apply(lambda S: BSMg(S, K, T, r, vol_range, True))
with col4:
    st.subheader("Gamma")
    st.pyplot(plot_greek(data, "Gamma_Call", color="#7FDBFF"))

data["Vega_Call"] = data["Close"].apply(lambda S: BSMv(S, K, T, r, vol_range, True))
with col5:
    st.subheader("Vega")
    st.pyplot(plot_greek(data, "Vega_Call", color="#A29BFE"))

data["Theta_Call"] = data["Close"].apply(lambda S: BSMt(S, K, T, r, vol_range, True))
with col6:
    st.subheader("Theta")
    st.pyplot(plot_greek(data, "Theta_Call", color="#8F94FA"))

data["Rho_Call"] = data["Close"].apply(lambda S: BSMr(S, K, T, r, vol_range, True))
with col7:
    st.subheader("Rho")
    st.pyplot(plot_greek(data, "Rho_Call", color="#81ecec"))

