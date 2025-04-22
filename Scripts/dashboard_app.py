import streamlit as st
from visualizations import plot_global_temperature_trend, plot_correlation_heatmap, plot_temp_co2_trend, plot_temperature_distribution, plot_global_temperature_map, plot_top_10_hottest_countries,plot_us_temperature_trend,plot_numerical_feature_distributions
import pandas as pd
from visualizations import plot_disaster_type_vs_total_affected, plot_disaster_event_heatmap,plot_co2_vs_temperature, plot_disaster_damage_trend,plot_top_affected_countries
from visualizations import  plot_global_precipitation_trend, plot_top_precip_countries
import plotly.express as px

# Sidebar for navigation
st.sidebar.title("📊 Dashboard Navigation")
selected_option = st.sidebar.selectbox(
    "Choose a section:",
    ["Overview", "Temperature Trends", "Correlations and Distributions", "Disasters, Precipitation & CO₂ Trends",
     "Clustering Results","Model Results"]
)
df = pd.read_csv("../Data/merged_df.csv")
df1 = pd.read_csv("../Data/df1.csv")
df2=pd.read_csv('../Data/new_df5.csv')
df3=pd.read_csv('../Data/df3.csv')
cluster_df = pd.read_csv("../Data/clustering_results.csv")
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
elif selected_option == "Clustering Results":
    # Load the clustering results
    df = pd.read_csv("../Data/clustering_results.csv")

    # Dropdown to choose clustering algorithm
    algorithm = st.selectbox("Choose Clustering Algorithm", ["KMeans", "Agglomerative", "GMM"])

    if algorithm == "KMeans":
        cluster_col = "KMeans_Cluster"
        label_col = "KMeans_Label"
    elif algorithm == "Agglomerative":
        cluster_col = "Agglo_Cluster"
        label_col = "Cluster_Label"
    else:  # GMM
        cluster_col = "GMM_Cluster"
        label_col = "Cluster_Label"

    # Title
    st.subheader(f"📊 {algorithm} Clustering Results")

    # Bar chart: Number of countries per cluster
    cluster_counts = df[label_col].value_counts().reset_index()
    cluster_counts.columns = ["Cluster Label", "Number of Countries"]

    fig_bar = px.bar(
        cluster_counts,
        x="Cluster Label",
        y="Number of Countries",
        color="Cluster Label",
        title="Number of Countries per Cluster",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # PCA Scatter plot
    fig_scatter = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color=df[label_col],
        hover_name="Country",
        title=f"{algorithm} Clustering - PCA Projection",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Choropleth Map
    fig_map = px.choropleth(
        df,
        locations="Country",
        locationmode="country names",
        color=df[label_col],
        title=f"{algorithm} Clustering - Country Map",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_name="Country"
    )
    fig_map.update_layout(legend_title_text="Cluster Label")
    st.plotly_chart(fig_map, use_container_width=True)

elif selected_option == "Model Results":
    st.title("📈 Model Performance Comparison")

    # Load results
    reg_df = pd.read_csv("../Data/results/Regression_results.csv")
    cls_df = pd.read_csv("../Data/results/classification_results.csv")

    ######################
    # 🔢 Regression Block
    ######################
    st.subheader("🔢 Regression Model Comparison")

    # Select target
    reg_target = st.selectbox("Select Target Variable", reg_df["Target"].unique(), key="reg_target")

    # Filter based on target
    reg_filtered = reg_df[reg_df["Target"] == reg_target]

    # Choose two models
    reg_models = reg_filtered["Model"].unique().tolist()
    reg_model1 = st.selectbox("Select First Regression Model", reg_models, key="reg_model1")
    reg_model2 = st.selectbox("Select Second Regression Model", [m for m in reg_models if m != reg_model1], key="reg_model2")
    

    reg_comp = reg_filtered[reg_filtered["Model"].isin([reg_model1, reg_model2])]

    # Plot
    st.markdown("#### 📊 Comparison of Metrics")
    reg_melted = reg_comp.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R2 Score"], var_name="Metric", value_name="Value")
    fig_reg = px.bar(reg_melted, x="Metric", y="Value", color="Model", barmode="group", title=f"{reg_target} - Model Comparison")
    st.plotly_chart(fig_reg, use_container_width=True)

    # Show best
    best_model = reg_comp.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
    st.success(f"✅ Best Performing Regression Model for **{reg_target}**: **{best_model}**")

    ######################
    # 🧮 Classification Block
    ######################
    st.subheader("🧮 Classification Model Comparison")

    # Choose two classification models
    cls_models = cls_df["Model"].unique().tolist()
    cls_model1 = st.selectbox("Select First Classification Model", cls_models, key="cls_model1")
    cls_model2 = st.selectbox("Select Second Classification Model", [m for m in cls_models if m != cls_model1], key="cls_model2")

    cls_comp = cls_df[cls_df["Model"].isin([cls_model1, cls_model2])]

    # Plot
    st.markdown("#### 📊 Comparison of Metrics")
    cls_melted = cls_comp.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Value")
    fig_cls = px.bar(cls_melted, x="Metric", y="Value", color="Model", barmode="group", title="Classification Model Comparison")
    st.plotly_chart(fig_cls, use_container_width=True)

    # Show best
    best_cls_model = cls_comp.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    st.success(f"✅ Best Performing Classification Model: **{best_cls_model}**")

else:
    st.write("Welcome to the Climate & Disaster Impact Dashboard!")







