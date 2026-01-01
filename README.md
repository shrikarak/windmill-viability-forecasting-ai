# AI-Powered Viability Analysis for Windmill Installation Sites

Copyright (c) 2026 Shrikara Kaudambady. All rights reserved.

## 1. Introduction

Choosing the right location for a wind farm is a complex decision that goes beyond simply finding a windy spot. The financial viability of a wind energy project depends on the **consistency, reliability, and predictability** of its power generation over time.

This project introduces a sophisticated Jupyter Notebook that uses **AI-powered time-series forecasting** to analyze and compare the potential viability of multiple candidate sites for windmill installations. By training a machine learning model on historical wind patterns, we can forecast future energy output and calculate key performance metrics for each site.

## 2. The Solution Explained

This solution moves beyond static mapping and implements a dynamic analysis. We simulate hourly wind speed data for several candidate sites and then build a predictive model to forecast their future energy generation potential.

### 2.1. Data Simulation and Power Curve

1.  **Time-Series Wind Data:** The notebook generates several years of synthetic hourly wind speed data for five distinct candidate sites. Each site is given a unique profile, with different seasonal, daily, and random volatility characteristics, simulating real-world conditions.
2.  **Turbine Power Curve:** A realistic power curve function is modeled. This function translates wind speed (m/s) into electrical power output (kW), accounting for the turbine's operational limits:
    *   **Cut-in Speed:** The minimum wind speed required to start generating power.
    *   **Rated Speed:** The wind speed at which the turbine reaches its maximum power output.
    *   **Cut-out Speed:** The maximum safe operational wind speed, beyond which the turbine shuts down to prevent damage.

### 2.2. Methodology: Time-Series Forecasting

The core of the analysis is a machine learning model trained to predict energy output.

1.  **Feature Engineering:** To help the model understand temporal patterns, we enrich the dataset by creating features from the timestamp, such as:
    *   Hour of the day
    *   Day of the week
    *   Month of the year
    *   Lag features (wind speed from previous hours)

2.  **Model Training:** A `GradientBoostingRegressor` model from `scikit-learn` is trained for each site. The model learns the complex, non-linear relationships between the time-based features and the resulting power output.

3.  **Forecasting and Viability Analysis:** The trained models are used to forecast the hourly energy output for each site over the next year. From this forecast, we calculate two critical industry metrics:
    *   **Annual Energy Production (AEP):** The total predicted energy (in GWh) that a site will generate over one year. This is the primary indicator of revenue.
    *   **Capacity Factor (%):** The ratio of the actual AEP to the theoretical maximum energy that could have been produced if the turbine ran at full capacity 24/7. This is a crucial measure of a site's efficiency and reliability.

4.  **Comparative Analysis:** The notebook concludes by generating a summary table and visualizations that rank the candidate sites based on their forecasted AEP and Capacity Factor, providing a clear, data-driven recommendation for the most promising investment.

## 3. How to Use the Notebook

### 3.1. Prerequisites

You will need Python 3 and Jupyter Notebook/JupyterLab, along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3.2. Running the Notebook

1.  Clone this repository:
    ```bash
    git clone https://github.com/shrikarak/windmill-viability-forecasting-ai.git
    cd windmill-viability-forecasting-ai
    ```
2.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
3.  Open `windmill_viability_analysis.ipynb` and run the cells in order.

## 4. Deployment and Customization

This notebook is designed as a template and can be easily adapted for your own projects.

1.  **Using Your Own Data:**
    *   Replace the synthetic data generation cell with code to load your own historical time-series wind data. The data should be in a pandas DataFrame with a `datetime` index and columns for the wind speed at each of your candidate sites.

2.  **Customizing the Turbine:**
    *   Modify the parameters within the `calculate_power` function (`cut_in_speed`, `rated_speed`, `cut_out_speed`, `rated_power_kw`) to match the exact specifications of the wind turbine model you plan to use.

3.  **Adjusting the Forecast:**
    *   You can change the `forecast_horizon_days` variable to extend or shorten the prediction period for your analysis.
