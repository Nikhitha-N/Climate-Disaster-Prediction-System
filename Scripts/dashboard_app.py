import streamlit as st
from visualizations import plot_global_temperature_trend, plot_correlation_heatmap, plot_temp_co2_trend, plot_temperature_distribution, plot_global_temperature_map, plot_top_10_hottest_countries,plot_us_temperature_trend,plot_numerical_feature_distributions
import pandas as pd
from visualizations import plot_disaster_type_vs_total_affected, plot_disaster_event_heatmap,plot_co2_vs_temperature, plot_disaster_damage_trend,plot_top_affected_countries
from visualizations import  plot_global_precipitation_trend, plot_top_precip_countries

# Sidebar for navigation
st.sidebar.title("📊 Dashboard Navigation")
selected_option = st.sidebar.selectbox(
    "Choose a section:",
    ["Overview", "Temperature Trends", "Correlations and Distributions", "Disasters, Precipitation & CO₂ Trends"]
)
df = pd.read_csv("../Data/merged_df.csv")
df1 = pd.read_csv("../Data/df1.csv")
df2=pd.read_csv('../Data/new_df5.csv')
df3=pd.read_csv('../Data/df3.csv')
# Conditional rendering based on selection
if selected_option == "Temperature Trends":
    plot_global_temperature_trend(df1)
    plot_temperature_distribution(df1)
    plot_global_temperature_map(df1,2000)
    plot_top_10_hottest_countries(df1)
    plot_us_temperature_trend(df1)
elif selected_option == "Correlations and Distributions":
    plot_correlation_heatmap(df)
    plot_numerical_feature_distributions(df)
    plot_disaster_type_vs_total_affected(df)
    plot_disaster_event_heatmap(df)
    plot_co2_vs_temperature(df)
elif selected_option == "Disasters, Precipitation & CO₂ Trends":
    plot_temp_co2_trend(df)
    plot_disaster_damage_trend(df2)
    plot_top_affected_countries(df2)
    plot_global_precipitation_trend(df3)
    plot_top_precip_countries(df3)
else:
    st.write("Welcome to the Climate & Disaster Impact Dashboard!")





