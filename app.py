import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import math
from scipy.stats import norm
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


# Greeks for european call options

def BSMd(S, K, T, r, vol, call = True ):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    delta_c = norm.cdf(d1)
    delta_p = -norm.cdf(-d1)

    if call == True :
        return delta_c
    else:
        return delta_p


def BSMg(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    gamma_c = norm.pdf(d1) / (S * vol * np.sqrt(T))
    gamma_p = norm.pdf(d1) / (S * vol * math.sqrt(T))
    
    if call == True :
        return gamma_c
    else:
        return gamma_p

def BSMv(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    vega_c = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change
    vega_p = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% change
    
    if call == True :
        return vega_c
    else:
        return vega_p


def BSMt(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    theta_c = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_p = (-S * norm.pdf(d1) * vol / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    if call == True :
        return theta_c
    else:
        return theta_p

def BSMr(S, K, T, r, vol, call = True):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    rho_c = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_p = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    if call == True :
        return rho_c
    else:
        return rho_p


#___________________________________________________________________________________________________________


# Calculation of d1 and d2 & computation of the BSM formula for pricing call/put options. 
def BSM_EU (S, K, T, r, vol, call = True) :
    
    d1 = (math.log(S/K) + (r + 0.5 * vol **2) * T ) / (vol * math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    
    price_EUcall = S * norm.cdf(d1) - K* math.exp(-r*T)*norm.cdf(d2)  
    price_EUput =  K* math.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)
    
    if call == True:
        return price_EUcall
    else:
        return price_EUput


# Computing the heatmap for European call options.

def compute_heatmap_call(S_vals, vol_vals, K, T, r):
    prices = np.zeros((len(vol_vals), len(S_vals)))
    for i, sigma in enumerate(vol_vals):
        for j, S in enumerate(S_vals):
            prices[i, j] = BSM_EU (S, K, T, r, sigma, True)
    return prices


# Computing the heatmap for European put options.

def compute_heatmap_put(S_vals, vol_vals, K, T, r):
    prices = np.zeros((len(vol_vals), len(S_vals)))
    for i, sigma in enumerate(vol_vals):
        for j, S in enumerate(S_vals):
            prices[i, j] = BSM_EU (S, K, T, r, sigma, False)
    return prices


# Pricing of an american put option using a brownian motion based on the BMS model.

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
        cashflow[in_the_money] = np.where(exercise, payoff[in_the_money, t],
                                          cashflow[in_the_money] * discount)

    # Final discounted expected value
    price = np.mean(cashflow) * np.exp(-r * dt)
    return price


#___________________________________________________________________________________________________________


def get_price_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    return data

#___________________________________________________________________________________________________________

st.set_page_config(page_title="Options Visualizer", layout="wide")
st.title("📈 Real-Time Option Pricing & Greeks Dashboard")

with st.sidebar :
    
    st.title("🌐 Black-Scholes Model Inputs and Assumptions")
    
    # Dropdown menu
    assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "^GSPC", "^IXIC", "^STOXX50E", "SAP", "ASML", "SIE.DE", "SU.PA", "ALV.DE"]  # Add Eurostoxx top 5
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


    # function to avoid auto-reinitialization of the strike price after modification
    def get_float(value):
        return float(value)
    
    # Option parameters

    lc_dummy = get_float(last_close)
    
    col1sb, col2sb = st.columns([3, 1])

    with col1sb :
        K = st.number_input("Strike Price", value = lc_dummy, min_value = 0.00, key ="strike_price") # mettre initial value à close et permettre un changement arbitraire ainsi qu'un bouton réinitialiser.
    
    with col2sb :
        if st.button("Reset Strike") :
            st.session_state["strike_price"] = lc_dummy
    
    T = st.number_input("Time to Maturity (in years)", value=0.5, step=0.25) 
    r = st.number_input("Risk-free rate", value=0.03)

   # style = st.selectbox("Option Style", ["European", "American"])




## Visualizing the heatmaps

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


#data.assign(Delta_Call = lambda S: BSMd(S.Close, K, T, r, vol_range,True))  
data["Delta_Call"] = data["Close"].apply(lambda S: BSMd(S, K, T, r, vol_range, True))

fig_dc, ax_dc = plt. subplots(figsize = (10, 8))
ax_dc.plot(data.index, data["Delta_Call"], label="Delta", color="blue")
ax_dc.set_xlabel("Date")
ax_dc.set_ylabel("Delta")
ax_dc.set_title("Call Option Delta Over Time")

fig_dc


#fig_gc
data["Gamma_Call"] = data["Close"].apply(lambda S: BSMg(S, K, T, r, vol_range, True))

fig_gc, ax_gc = plt. subplots(figsize = (10, 8))
ax_gc.plot(data.index, data["Gamma_Call"], label="Gamma", color="blue")
ax_gc.set_xlabel("Date")
ax_gc.set_ylabel("Gamma")
ax_gc.set_title("Call Option Gamma Over Time")

fig_gc

#fig_vc
data["Vega_Call"] = data["Close"].apply(lambda S: BSMv(S, K, T, r, vol_range, True))

fig_vc, ax_vc = plt. subplots(figsize = (10, 8))
ax_vc.plot(data.index, data["Vega_Call"], label="Vega", color="blue")
ax_vc.set_xlabel("Date")
ax_vc.set_ylabel("Vega")
ax_vc.set_title("Call Option Vega Over Time")

fig_vc

#fig_tc
data["Theta_Call"] = data["Close"].apply(lambda S: BSMt(S, K, T, r, vol_range, True))

fig_tc, ax_tc = plt. subplots(figsize = (10, 8))
ax_tc.plot(data.index, data["Theta_Call"], label="Theta", color="blue")
ax_tc.set_xlabel("Date")
ax_tc.set_ylabel("Theta")
ax_tc.set_title("Call Option Theta Over Time")

fig_tc

#fig_rc
data["Rho_Call"] = data["Close"].apply(lambda S: BSMr(S, K, T, r, vol_range, True))


fig_rc, ax_rc = plt. subplots(figsize = (10, 8))
ax_rc.plot(data.index, data["Rho_Call"], label="Rho", color="blue")
ax_rc.set_xlabel("Date")
ax_rc.set_ylabel("Rho")
ax_rc.set_title("Call Option Rho Over Time")

fig_rc
