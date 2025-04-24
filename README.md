# cap5771sp25-project
Climate and Disaster Prediction Recommender System

📌 **Project Overview**

This project aims to develop an advanced recommender system that predicts climate temperature trends and assesses potential natural disasters based on the user-specified year and location (country). By leveraging historical data and predictive modeling, the system will provide valuable insights into environmental risks, enabling users to make informed decisions. This tool will be a crucial resource for climate analysis, disaster preparedness, and risk mitigation strategies.

---

**Demonstration Video**
```
video link
```
---

🚀 **Features**

✔ Historical Temperature Trends – Retrieves and analyzes past temperature data.

✔ Natural Disaster Predictions – Assesses the likelihood of disasters based on past trends.

✔ CO₂ Emission Insights – Evaluates environmental impact and emissions correlation.

✔ Interactive Visualizations – Provides graphs and maps for easy interpretation.

✔ User-Specific Recommendations – Personalized predictions based on input year and location.

---

📊 **Data Sources**

The system uses multiple datasets collected from reputable sources:

Global Temperature Records (1850-2013) – Kaggle

Per Capita CO₂ Emissions – OurWorldInData

Annual Precipitation Data (1940-2024) – OurWorldInData

Deforestation & Forest Loss Data – OurWorldInData

Natural Disaster & Emergency Events Database – Omdena

---

🛠️ **Technology Stack**

Programming Language:

Python 🐍

Libraries & Frameworks:

Data Handling – Pandas, NumPy

Visualization – Matplotlib, Seaborn

Machine Learning – Scikit-Learn, TensorFlow

Geospatial Analysis – Geopandas, Folium

Web Interface - Streamlit

---

🏗️ Project Structure
```
|-- cap5771sp25-project
    |-- Data/                 # Raw datasets
    |-- Scripts/            # Jupyter Notebooks for analysis
    |-- Reports/              # Milestone reports
    |-- README.md             # Project documentation
```
---

 ## Key Insights
 - Rising Global Temperatures – Consistent warming with extreme fluctuations.
 - CO₂-Climate Link – High emissions drive temperature increases.
 - More Frequent Disasters – Climate-related disasters are rising.
 - High-Risk Regions – Coastal and developing areas are most vulnerable.

---

## Models Implemented
**Classification Task**
Objective: predicting the severity of disaster impact for a given country-year-disaster combination.

Models Implemented

- Random Forest
- MLP - Neural Network
- XGBoost 

**Regression Task**
Objective: Forecasting future climate indicators like temperature, co2 emissions and precipitation based on country and year.

Models Implemented

- Random Forest Regression
- LSTM
- XGBoost

**Clustering**

Objective: Clustering the countries based on the similar disaster profiles

Models Implemented

- KMeans
- Agglomerative Clustering
- Gaussian Mixture Model (GMM)

---

## Accessing the GitHub Repository
1. Go to the **GitHub repository**:
   ```
   https://github.com/Nikhitha-N/cap5771sp25-project
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/Nikhitha-N/cap5771sp25-project.git
   ```
3. Navigate to the project directory
4. Run the ipynb Notebook for analysis in jupyter notebook and vs code

--- 

## Accessing the Streamlit Interface
1. Clone the **GitHub repository** as per instructions given above:
2. Execute the following commands
   ```bash
   Cd Scripts
   ```
   ```bash
   streamlit run dashboard_app.py
   ```
---
## Contributors
- **Nikhitha Nagalla**

For any inquiries, feel free to open an issue on GitHub!

