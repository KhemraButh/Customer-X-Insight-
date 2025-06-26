import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load the simulated GPS pat
df = pd.read_csv("salesperson_path.csv")

st.set_page_config(page_title="Salesperson Movement Path", layout="wide")
st.title("📍 Salesperson Movement Tracker")

# Show data table
with st.expander("📋 Show Raw Data", expanded=False):
    st.dataframe(df)

# Create a map centered at the mean coordinates
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Draw the path line
folium.PolyLine(df[['latitude', 'longitude']].values, color="blue", weight=4, opacity=0.7).add_to(m)

# Add markers with timestamps every 10th point
for i, row in df.iterrows():
    if i % 10 == 0 or i == len(df) - 1:
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"{row['timestamp']}",
            tooltip=f"Point {i}"
        ).add_to(m)

# Show map
st_folium(m, width=1800, height=700)


