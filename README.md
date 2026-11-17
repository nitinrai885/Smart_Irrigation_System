# 🌱 Smart Irrigation System Using Apache Spark and Machine Learning

## 📌 Project Overview

The **Smart Irrigation System** is a machine-learning-based application designed to predict evapotranspiration (ET) for tomato crops using agricultural and environmental data.

The system analyses parameters such as temperature, humidity, soil moisture, solar radiation, wind speed, soil nutrients, and crop growth stages to estimate evapotranspiration.

The main objective is to support smarter, data-driven irrigation planning and help reduce unnecessary water usage.

## 🎯 Project Objectives

- Predict evapotranspiration using machine learning.
- Process agricultural datasets using Apache Spark and PySpark.
- Analyse environmental and agricultural parameters.
- Train and compare multiple regression algorithms.
- Evaluate models using RMSE and R².
- Develop an interactive dashboard using Streamlit.
- Support better agricultural water-management decisions.

## 🌾 What is Evapotranspiration?

Evapotranspiration is the combined loss of water through two processes:

1. **Evaporation:** Water changing into vapour from the soil surface.
2. **Transpiration:** Water vapour released by plants through their leaves.

**Evapotranspiration = Evaporation + Transpiration**

Predicting ET helps understand water loss from crops and soil.

## 📊 Dataset Information

- **Dataset:** Tomato Irrigation Dataset
- **Source:** Mendeley Data
- **Records:** 3,000
- **Original Attributes:** 14
- **Target Variable:** Evapotranspiration

Dataset: https://data.mendeley.com/datasets/33cngpcrmx/3

### Input Features

1. Temperature
2. Humidity
3. Soil Moisture
4. Reference Evapotranspiration
5. Crop Coefficient
6. Crop Growth Stage
7. Nitrogen
8. Phosphorus
9. Potassium
10. Solar Radiation
11. Wind Speed
12. Days Planted
13. Soil pH

**Target:** Evapotranspiration (ET)

### Crop Growth Stages

The dataset is divided into four crop growth stages:

- Initial Stage
- Development Stage
- Mid Stage
- Last Stage

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Apache Spark | Large-scale data processing |
| PySpark | Spark operations using Python |
| Spark MLlib | Machine learning |
| Pandas | Data analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Streamlit | Interactive dashboard |

## 🤖 Machine Learning Algorithms

We implemented three regression algorithms:

### 1. Linear Regression
Predicts continuous numerical values by learning a linear relationship between input features and the target variable.

### 2. Random Forest Regression
Uses multiple decision trees and combines their predictions to produce an output.

### 3. Gradient-Boosted Tree Regression (GBT)
Builds decision trees sequentially, with each new tree attempting to improve previous predictions.

### Model Training

We trained separate models for four crop growth stages.

- 4 Crop Growth Stages
- 3 Machine Learning Algorithms
- 12 Model Combinations

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Loading using PySpark
3. Data Cleaning and Preprocessing
4. Missing Value Handling using Median Imputation
5. Categorical Encoding using StringIndexer
6. Feature Combination using VectorAssembler
7. Dividing Data into Training and Testing Sets (80:20)
8. Stage-Wise Model Training
9. Model Evaluation
10. Feature Correlation and Importance Analysis
11. Model Saving
12. ET Prediction through Streamlit Dashboard

## 📈 Model Evaluation

We used two evaluation metrics:

**RMSE (Root Mean Squared Error)**

Measures the difference between actual and predicted values.

Lower RMSE indicates better prediction accuracy.

**R² (R-Squared)**

Measures how much variation in the target variable is explained by the model.

Higher R² generally indicates better model performance.

We also compared the trained models against a mean-based baseline model.

## 📂 Project Structure

```text
Smart_Irrigation_System/
│
├── data/
│   ├── tomato_irrigation_dataset.csv
│   └── processed_irrigation.csv
│
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── stage_wise_training.py
│   └── stage_model_comparison.py
│
├── models/
│   └── Saved ML Models
│
├── analysis/
│   ├── stage_feature_correlations.csv
│   ├── stage_feature_importance.csv
│   └── charts/
│
├── dashboard/
│   ├── app.py
│   ├── style.css
│   └── header.html
│
├── requirements.txt
└── README.md
```

## 🖥️ Dashboard Features

The project includes an interactive Streamlit dashboard with five main sections:

1. **Home:** Project introduction and overview.
2. **Data Analysis:** Dataset exploration and visualisation.
3. **Model Performance:** Comparison of regression algorithms.
4. **ET Prediction:** Predict evapotranspiration using agricultural input parameters.
5. **About Project:** Project information and technologies.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/nitinrai885/Smart_Irrigation_System.git
```

### Step 2: Navigate to the Project

```bash
cd Smart_Irrigation_System
```

### Step 3: Create Virtual Environment

```bash
py -3.13 -m venv .venv
```

### Step 4: Activate Virtual Environment

For Windows:

```cmd
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

**Note:** Apache Spark and its required Java and Windows configurations must also be installed correctly. The prediction dashboard requires the trained model files at the locations expected by the application.

## 🔍 Key Results

The project evaluates three regression algorithms across four crop growth stages.

The reported best-performing models based on RMSE are:

| Crop Growth Stage | Best Algorithm |
|---|---|
| Initial | Gradient-Boosted Trees |
| Development | Random Forest |
| Mid | Linear Regression |
| Last | Random Forest |

Model performance varies across stages. Some models did not outperform their baseline, indicating opportunities for further improvement.

## ⚠️ Project Limitations

- Uses an existing agricultural dataset rather than live sensor readings.
- Unequal distribution of records across crop growth stages.
- Some models require further performance improvement.
- Initial Stage results require additional validation.
- Does not directly control irrigation pumps or hardware.

## 🔮 Future Scope

- Integration with IoT-based soil moisture sensors.
- Real-time weather data integration.
- Automatic irrigation recommendations.
- Improved machine learning models.
- Mobile application development.
- Integration with automated irrigation systems.

## 👨‍💻 Author

**Nitin Rai**

B.Tech Computer Science and Engineering

GitHub: https://github.com/nitinrai885

## 📜 Project Note

This project demonstrates the application of Big Data processing and Machine Learning in agriculture.

It aims to support data-driven irrigation planning through evapotranspiration prediction.
