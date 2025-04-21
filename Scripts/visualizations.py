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

import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

def plot_global_temperature_map(df, year):
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df_year = df[df['dt'].dt.year == year]

    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='merc',
                llcrnrlat=-60, urcrnrlat=85,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c', ax=ax)

    m.drawcoastlines()
    m.drawcountries()

    x, y = m(df_year["Longitude"].values, df_year["Latitude"].values)

    sc = ax.scatter(x, y, c=df_year["AverageTemperature"], cmap="coolwarm", alpha=0.7)
    plt.colorbar(sc, label="Avg Temperature (°C)")
    plt.title(f"Global Temperature Distribution - {year}")
    plt.tight_layout()

    #  Show in Streamlit
    st.pyplot(fig)

import seaborn as sns
import matplotlib.pyplot as plt



def plot_top_10_hottest_countries(df):
    # Compute the highest temperature recorded for each country
    df_country_max = df.groupby('Country', as_index=False)['AverageTemperature'].max()

    # Get the top 10 countries
    top_10 = df_country_max.nlargest(10, 'AverageTemperature')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_10, x='Country', y='AverageTemperature', palette='Reds_r', ax=ax)

    # Annotate each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    ax.set_xlabel("Country", fontsize=12, fontweight='bold')
    ax.set_ylabel("Highest Recorded Temperature (°C)", fontsize=12, fontweight='bold')
    ax.set_title("Top 10 Countries with Highest Recorded Temperatures (All Time)", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

def plot_us_temperature_trend(df):
    # Filter for United States
    df_us = df[df['Country'] == 'United States'].copy()

    # Ensure datetime format and extract year
    df_us['dt'] = pd.to_datetime(df_us['dt'], errors='coerce')
    df_us['Year'] = df_us['dt'].dt.year

    # Group and compute yearly average temperature
    df_us_mean = df_us.groupby('Year', as_index=False)['AverageTemperature'].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_us_mean['Year'], df_us_mean['AverageTemperature'], marker='o', linestyle='-', color='b')

    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Temperature (°C)", fontsize=12, fontweight='bold')
    ax.set_title("Temperature Trend in the United States Over Time", fontsize=14, fontweight='bold')
    ax.grid(True)

    st.pyplot(fig)


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

def plot_numerical_feature_distributions(df):
    features = [
        "AvgTemp_Year", "Annual CO₂ emissions (per capita)", "Annual precipitation",
        "Total Events", "Total Affected", "Total Deaths",
        "Total Damage (USD, adjusted)", "CPI"
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(features):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        else:
            axes[i].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_disaster_type_vs_total_affected(df):
    if "Disaster Type" in df.columns and "Total Affected" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Disaster Type", y="Total Affected")
        plt.xticks(rotation=45)
        plt.title("Disaster Type vs Total Affected (log scale)")
        plt.yscale("log")
        plt.tight_layout()
        st.pyplot(plt.gcf())  # Display the current figure in Streamlit
    else:
        st.warning("Required columns 'Disaster Type' or 'Total Affected' not found in the dataset.")

def plot_disaster_event_heatmap(df):
    if all(col in df.columns for col in ["Country", "Disaster Type", "Total Events"]):
        pivot = df.pivot_table(index="Country", columns="Disaster Type",
                               values="Total Events", aggfunc="sum", fill_value=0)

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
        plt.title("Disaster Event Count per Country")
        plt.tight_layout()
        st.pyplot(plt.gcf())  # Display the current figure in Streamlit
    else:
        st.warning("Required columns 'Country', 'Disaster Type', or 'Total Events' are missing in the DataFrame.")

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_co2_vs_temperature(df):
    required_cols = ["Annual CO₂ emissions (per capita)", "AvgTemp_Year", "Country"]
    if all(col in df.columns for col in required_cols):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df,
                        x="Annual CO₂ emissions (per capita)",
                        y="AvgTemp_Year",
                        hue="Country",
                        alpha=0.7,
                        legend=False)
        plt.title("CO₂ Emissions vs Avg Temperature")
        plt.xlabel("CO₂ Emissions (per capita)")
        plt.ylabel("Avg Temperature (°C)")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt.gcf())
    else:
        st.warning("Required columns missing: 'Annual CO₂ emissions (per capita)', 'AvgTemp_Year', or 'Country'.")

# Disaster visualizations
def plot_disaster_damage_trend(df):
    if "Year" in df.columns and "Total Damage (USD, adjusted)" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="Year", y="Total Damage (USD, adjusted)", marker="o", color="r")

        plt.xlabel("Year", fontsize=12, fontweight='bold')
        plt.ylabel("Total Damage (USD, Adjusted)", fontsize=12, fontweight='bold')
        plt.title("Total Disaster Damage Over Time", fontsize=14, fontweight='bold')
        plt.grid(True)
        st.pyplot(plt.gcf())
    else:
        st.warning("Required columns 'Year' and 'Total Damage (USD, adjusted)' not found in the dataset.")

def plot_top_affected_countries(df):
    if "Country" in df.columns and "Total Affected" in df.columns:
        top_affected = df.groupby("Country")["Total Affected"].sum().nlargest(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_affected.values, y=top_affected.index, palette="Reds_r")

        plt.xlabel("Total People Affected", fontsize=12, fontweight='bold')
        plt.ylabel("Country", fontsize=12, fontweight='bold')
        plt.title("Top 10 Countries with Most Affected People", fontsize=14, fontweight='bold')

        # Annotate values
        for index, value in enumerate(top_affected.values):
            plt.text(value, index, f"{int(value):,}", ha="left", va="center", fontsize=10, fontweight="bold")

        st.pyplot(plt.gcf())
    else:
        st.warning("Required columns 'Country' and 'Total Affected' not found in the dataset.")

## co2 emission
def plot_global_precipitation_trend(df):
    if "Year" in df.columns and "Annual precipitation" in df.columns:
        # Aggregate
        global_precip = df.groupby("Year", as_index=False)["Annual precipitation"].mean()

        # Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=global_precip, x="Year", y="Annual precipitation", marker="o", color="b")
        plt.xlabel("Year", fontsize=12, fontweight='bold')
        plt.ylabel("Global Avg Precipitation (mm)", fontsize=12, fontweight='bold')
        plt.title("Global Precipitation Over Time", fontsize=14, fontweight='bold')
        plt.grid(True)

        # Display with Streamlit
        st.pyplot(plt.gcf())
    else:
        st.warning("Required columns 'Year' and 'Annual precipitation' not found in the dataset.")

def plot_top_precip_countries(df):
    if "Entity" in df.columns and "Annual precipitation" in df.columns:
        # Compute mean precipitation
        top_countries = df.groupby("Entity", as_index=False)["Annual precipitation"].mean()
        top_10 = top_countries.nlargest(10, "Annual precipitation")

        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=top_10, x="Annual precipitation", y="Entity", palette="Blues_r")

        # Annotate values
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.2f}", 
                        (p.get_width(), p.get_y() + p.get_height()/2),
                        ha='left', va='center', fontsize=12, fontweight='bold', color='black')

        # Labels
        plt.xlabel("Avg Annual Precipitation (mm)", fontsize=12, fontweight='bold')
        plt.ylabel("Country", fontsize=12, fontweight='bold')
        plt.title("Top 10 Countries with Highest Precipitation", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Show in Streamlit
        st.pyplot(plt.gcf())
    else:
        st.warning("Required columns 'Entity' and 'Annual precipitation' not found in the dataset.")






