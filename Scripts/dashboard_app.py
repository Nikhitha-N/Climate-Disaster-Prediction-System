import streamlit as st
from visualizations import plot_global_temperature_trend, plot_correlation_heatmap, plot_temp_co2_trend, plot_temperature_distribution, plot_global_temperature_map, plot_top_10_hottest_countries,plot_us_temperature_trend,plot_numerical_feature_distributions
import pandas as pd
from visualizations import plot_disaster_type_vs_total_affected, plot_disaster_event_heatmap,plot_co2_vs_temperature
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
    plot_top_10_hottest_countries(df1)
    plot_us_temperature_trend(df1)
elif selected_option == "Correlation Heatmap":
    plot_correlation_heatmap(df)
    plot_numerical_feature_distributions(df)
    plot_disaster_type_vs_total_affected(df)
    plot_disaster_event_heatmap(df)
    plot_co2_vs_temperature(df)
elif selected_option == "Temperature & CO₂ Trends":
    plot_temp_co2_trend(df)
else:
    st.write("Welcome to the Climate & Disaster Impact Dashboard!")
