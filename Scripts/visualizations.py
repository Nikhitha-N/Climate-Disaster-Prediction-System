import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from mpl_toolkits.basemap import Basemap

def plot_global_temperature_trend(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # Convert dt column to datetime and extract year
    df["Year"] = pd.to_datetime(df["dt"]).dt.year
    yearly_avg = df.groupby("Year")["AverageTemperature"].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=yearly_avg, x="Year", y="AverageTemperature")
    plt.title("Global Average Temperature Over Time")
    plt.xlabel("Year")
    plt.ylabel("Avg Temperature (°C)")
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_temperature_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["AverageTemperature"].dropna(), bins=30, kde=True, color="blue")
    plt.title("Distribution of Average Temperatures")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_global_temperature_map(df, year):
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    # Ensure datetime format
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df_year = df[df['dt'].dt.year == year]

    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='merc',
                llcrnrlat=-60, urcrnrlat=85,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c', ax=ax)

    m.drawcoastlines()
    m.drawcountries()

    # Convert coordinates to map projection
    x, y = m(df_year["Longitude"].values, df_year["Latitude"].values)

    # Plot temperature points
    sc = ax.scatter(x, y, c=df_year["AverageTemperature"], cmap="coolwarm", alpha=0.7)

    plt.colorbar(sc, label="Avg Temperature (°C)")
    plt.title(f"Global Temperature Distribution - {year}")
    plt.tight_layout()
    plt.show()




def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    corr = df.select_dtypes(include=["float64", "int64"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

def plot_temp_co2_trend(df):
    yearly_avg = df.groupby("Year")[['AvgTemp_Year', 'Annual CO₂ emissions (per capita)']].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='AvgTemp_Year', data=yearly_avg, label='Avg Temperature')
    sns.lineplot(x='Year', y='Annual CO₂ emissions (per capita)', data=yearly_avg, label='CO₂ Emissions')
    plt.title("Temperature and CO₂ Trends Over Time")
    plt.ylabel("Value")
    plt.grid(True)
    st.pyplot(plt)

