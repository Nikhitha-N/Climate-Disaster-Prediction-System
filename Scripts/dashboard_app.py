import streamlit as st
from visualizations import plot_global_temperature_trend, plot_correlation_heatmap, plot_temp_co2_trend, plot_temperature_distribution, plot_global_temperature_map
import pandas as pd

# Sidebar for navigation
st.sidebar.title("📊 Dashboard Navigation")
selected_option = st.sidebar.selectbox(
    "Choose a section:",
    ["Overview", "Disaster Frequencies", "Correlation Heatmap", "Temperature & CO₂ Trends"]
)
df = pd.read_csv("../Data/merged_df.csv")
df1 = pd.read_csv("../Data/df1.csv")
# Conditional rendering based on selection
if selected_option == "Disaster Frequencies":
    plot_global_temperature_trend(df1)
    plot_temperature_distribution(df1)
    plot_global_temperature_map(df1,2000)
elif selected_option == "Correlation Heatmap":
    plot_correlation_heatmap(df)
elif selected_option == "Temperature & CO₂ Trends":
    plot_temp_co2_trend(df)
else:
    st.write("Welcome to the Climate & Disaster Impact Dashboard!")
