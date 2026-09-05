
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="AgriMarket AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Demo data
# -----------------------------
np.random.seed(42)
dates = pd.date_range("2026-01-01", periods=180, freq="D")
base = 2200 + 180*np.sin(np.arange(180)/14) + np.linspace(0, 220, 180)
price = base + np.random.normal(0, 75, 180)
arrivals = 100 + 25*np.sin(np.arange(180)/11) + np.random.normal(0, 8, 180)
rainfall = np.maximum(0, 8 + 10*np.sin(np.arange(180)/18) + np.random.normal(0, 5, 180))

df = pd.DataFrame({
    "date": dates,
    "price": price,
    "arrivals": arrivals,
    "rainfall": rainfall
})

# Lag features for demo ML model
df["lag_1"] = df["price"].shift(1)
df["lag_7"] = df["price"].shift(7)
df["arrival_7"] = df["arrivals"].rolling(7).mean()
train = df.dropna().copy()

X = train[["lag_1", "lag_7", "arrivals", "rainfall", "arrival_7"]]
y = train["price"]

model = RandomForestRegressor(n_estimators=180, random_state=42)
model.fit(X, y)

last = df.iloc[-1]
forecast_input = pd.DataFrame([{
    "lag_1": last["price"],
    "lag_7": df.iloc[-7]["price"],
    "arrivals": last["arrivals"],
    "rainfall": last["rainfall"],
    "arrival_7": df["arrivals"].tail(7).mean()
}])
predicted_price = float(model.predict(forecast_input)[0])

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🌾 AgriMarket AI")
st.sidebar.caption("SIH • Market Linkage & Price Discovery")

crop = st.sidebar.selectbox(
    "Select Crop",
    ["Tomato", "Onion", "Potato", "Wheat", "Soybean", "Cotton"]
)
location = st.sidebar.selectbox(
    "Farmer Location",
    ["Pune", "Nashik", "Ahmednagar", "Satara", "Solapur"]
)
days = st.sidebar.selectbox("Forecast Horizon", [7, 15, 30])

st.sidebar.markdown("---")
st.sidebar.info(
    "Demo mode: replace the generated data with official mandi, "
    "weather and arrivals datasets before deployment."
)

# -----------------------------
# Header
# -----------------------------
st.title("🌾 AgriMarket AI Dashboard")
st.markdown(
    f"### Strengthening Market Linkages & Price Discovery for Farmers"
)
st.caption(f"Selected crop: **{crop}**  •  Location: **{location}**")

# -----------------------------
# Mandi recommendation
# -----------------------------
mandis = pd.DataFrame({
    "Mandi": ["Pune", "Nashik", "Mumbai", "Ahmednagar", "Satara"],
    "Current Price": [2500, 2720, 2860, 2580, 2640],
    "Transport Cost": [80, 170, 260, 120, 110]
})
mandis["Net Value"] = mandis["Current Price"] - mandis["Transport Cost"]
best = mandis.loc[mandis["Net Value"].idxmax()]

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"₹{last['price']:,.0f}/q")
c2.metric("AI Predicted Price", f"₹{predicted_price:,.0f}/q",
          f"{((predicted_price-last['price'])/last['price'])*100:+.1f}%")
c3.metric("Best Mandi", best["Mandi"], f"Net ₹{best['Net Value']:,.0f}/q")
c4.metric("Market Signal", "BUY / WAIT" if predicted_price > last["price"] else "SELL NOW")

st.markdown("---")

# -----------------------------
# Price chart
# -----------------------------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("📈 Market Price & AI Forecast")

    chart_df = df.tail(90).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["date"], y=chart_df["price"],
        mode="lines", name="Historical Price"
    ))

    future_dates = pd.date_range(
        df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=days
    )
    future_prices = []
    rolling_price = last["price"]

    for i in range(days):
        lag7 = df.iloc[-7]["price"]
        inp = pd.DataFrame([{
            "lag_1": rolling_price,
            "lag_7": lag7,
            "arrivals": max(50, last["arrivals"] + np.random.normal(0, 4)),
            "rainfall": max(0, last["rainfall"] + np.random.normal(0, 2)),
            "arrival_7": df["arrivals"].tail(7).mean()
        }])
        p = float(model.predict(inp)[0])
        future_prices.append(p)
        rolling_price = p

    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines+markers", name=f"{days}-Day AI Forecast",
        line=dict(dash="dash")
    ))
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🚨 AI Recommendation")
    if predicted_price > last["price"] * 1.05:
        st.success(
            f"**Consider waiting.**\n\n"
            f"AI predicts approximately ₹{predicted_price:,.0f}/q."
        )
    elif predicted_price < last["price"] * 0.97:
        st.warning(
            f"**Selling opportunity.**\n\n"
            f"AI predicts a possible decline to ₹{predicted_price:,.0f}/q."
        )
    else:
        st.info(
            f"**Stable market.**\n\n"
            f"Expected price: ₹{predicted_price:,.0f}/q."
        )

    st.markdown("#### 📊 Market Factors")
    st.write(f"• Recent arrivals: **{last['arrivals']:.0f} q/day**")
    st.write(f"• Rainfall indicator: **{last['rainfall']:.1f} mm**")
    st.write(f"• Forecast horizon: **{days} days**")

st.markdown("---")

# -----------------------------
# Mandi comparison
# -----------------------------
st.subheader("🏪 Mandi Price Comparison")

display = mandis.copy()
display["Recommendation"] = np.where(
    display["Mandi"].eq(best["Mandi"]), "⭐ Best Net Value", ""
)
st.dataframe(
    display.rename(columns={
        "Current Price": "Price (₹/q)",
        "Transport Cost": "Transport (₹/q)",
        "Net Value": "Net Value (₹/q)"
    }),
    use_container_width=True,
    hide_index=True
)

# Bar chart
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=mandis["Mandi"],
    y=mandis["Net Value"],
    text=mandis["Net Value"].round(0),
    textposition="auto",
    name="Net Value"
))
fig2.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis_title="Expected Net Value (₹/quintal)"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Supply / weather
# -----------------------------
a, b = st.columns(2)

with a:
    st.subheader("📦 Supply / Arrival Trend")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df.tail(60)["date"],
        y=df.tail(60)["arrivals"],
        mode="lines",
        name="Mandi Arrivals"
    ))
    fig3.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

with b:
    st.subheader("🌦️ Weather Indicator")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=df.tail(30)["date"],
        y=df.tail(30)["rainfall"],
        name="Rainfall"
    ))
    fig4.update_layout(
        height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        yaxis_title="Rainfall (mm)"
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "SIH Prototype • ML model shown here is a demonstration. "
    "For production, validate against official historical mandi prices, "
    "arrivals, weather and transportation data."
)
