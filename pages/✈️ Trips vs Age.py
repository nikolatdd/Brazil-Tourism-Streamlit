from Home import df
import plotly.express as px
import streamlit as st
import pandas as pd
import os

st.divider()
st.title("✈️Trips vs Age")

st.caption("This chart shows how the number of trips varies by tourist age. "
           "It helps identify travel patterns among different age groups.")

# Optional: filter out unrealistic ages (if dataset has noise)
df_chart = df[df["Age"].between(10, 90)]

# Create the chart
st.scatter_chart(
    df_chart,
    x="Age",
    y="Trips",
)

# Optional: Add a little interpretation summary
avg_trips_young = df_chart[df_chart["Age"] < 30]["Trips"].mean()
avg_trips_senior = df_chart[df_chart["Age"] > 60]["Trips"].mean()

st.write(f"🧭 *Average trips for tourists under 30:* **{avg_trips_young:.1f}**")
st.write(f"🏖️ *Average trips for tourists over 60:* **{avg_trips_senior:.1f}**")