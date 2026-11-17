from PIL import Image, ImageOps
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_irrigation.csv"
RESULT_PATH = BASE_DIR / "models" / "stage_wise_results.csv"
CHART_DIR = BASE_DIR / "analysis" / "charts"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 Smart Irrigation")
st.sidebar.write("Machine Learning Based ET Analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Data Analysis",
        "Model Performance",
        "ET Prediction",
        "About Project"
    ]
)

# ---------------- HOME ----------------
if page == "Home":

    st.title("🌱 Smart Irrigation System Prediction")
    st.write(
        "A data-driven system that analyzes environmental and soil "
        "conditions to understand and predict evapotranspiration (ET) "
        "for irrigation planning."
    )

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(df.columns))
    c3.metric(
        "Average Evapotranspiration",
        f"{df['evapotranspiration'].mean():.2f}"
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Project Objectives")
    st.markdown("""
    - Analyze soil and environmental conditions.
    - Study evapotranspiration at different crop stages.
    - Apply machine learning algorithms using Apache Spark.
    - Compare model performance using RMSE and R².
    - Support better irrigation planning through data analysis.
    """)

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.subheader("Evapotranspiration Distribution")

    fig, ax = plt.subplots()
    ax.hist(
        df["evapotranspiration"].dropna(),
        bins=30,
        edgecolor="black"
    )
    ax.set_xlabel("Evapotranspiration")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Evapotranspiration")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Records by Crop Stage")

    stage_counts = df["crop_coefficient_stage"].value_counts()

    fig, ax = plt.subplots()
    stage_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Crop Stage")
    ax.set_ylabel("Number of Records")
    ax.set_title("Records in Each Crop Stage")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average ET by Crop Stage")

    stage_avg = (
        df.groupby("crop_coefficient_stage")["evapotranspiration"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(stage_avg)

    st.subheader("Numerical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.title("🤖 Model Performance Comparison")

    if RESULT_PATH.exists():

        results = pd.read_csv(RESULT_PATH)

        st.subheader("Stage-wise Model Results")
        st.dataframe(results, use_container_width=True)

        st.info(
            "RMSE measures prediction error (lower is better). "
            "R² measures how well the model explains variation "
            "in the target (higher is better)."
        )

    else:
        st.warning("Model results file was not found.")

        st.subheader("Generated Analysis Charts")

    chart_files = {
        "Initial Stage": "Initial_Stage",
        "Development Stage": "Development_Stage",
        "Mid Stage": "Mid_stage",
        "Last Stage": "Last_stage"
    }

    for stage, filename in chart_files.items():

        st.markdown(f"### {stage}")

        actual_chart = CHART_DIR / f"{filename}_actual_vs_predicted.png"
        importance_chart = CHART_DIR / f"{filename}_feature_importance.png"

        col1, col2 = st.columns(2)

        with col1:
            if actual_chart.exists():
                img = Image.open(actual_chart).convert("RGB")
                img = ImageOps.fit(img, (900, 500))
                st.image(img, caption="Actual vs Predicted", use_container_width=True)
            else:
                st.caption("Actual vs Predicted chart not found.")

        with col2:
            if importance_chart.exists():
                img = Image.open(importance_chart).convert("RGB")
                img = ImageOps.fit(img, (900, 500))
                st.image(img, caption="Feature Importance", use_container_width=True)
            else:
                st.caption("Feature importance chart not found.")

# ---------------- ET PREDICTION ----------------
elif page == "ET Prediction":

    st.title("💧 Evapotranspiration Prediction")

    st.write(
        "Enter environmental conditions to prepare a prediction. "
        "A trained prediction model must be connected before this "
        "form can return a real prediction."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.number_input(
                "Temperature (°C)", value=25.0
            )
            humidity = st.number_input(
                "Humidity (%)", value=60.0
            )
            soil_moisture = st.number_input(
                "Soil Moisture", value=150.0
            )
            reference_et = st.number_input(
                "Reference Evapotranspiration", value=50.0
            )
            crop_coefficient = st.number_input(
                "Crop Coefficient", value=0.8
            )

        with col2:
            nitrogen = st.number_input("Nitrogen", value=40.0)
            phosphorus = st.number_input("Phosphorus", value=30.0)
            potassium = st.number_input("Potassium", value=40.0)
            solar = st.number_input("Solar Radiation", value=20.0)
            wind = st.number_input("Wind Speed", value=2.0)

        submitted = st.form_submit_button("Prepare Prediction")

    if submitted:
        st.warning(
            "The input has been collected, but a trained model "
            "has not yet been connected. No prediction has been generated."
        )

# ---------------- ABOUT PROJECT ----------------
elif page == "About Project":

    st.title("📘 About the Project")

    st.subheader("Project Description")

    st.write("""
    Smart Irrigation System Prediction uses environmental and soil
    information to analyze evapotranspiration across different crop
    growth stages.

    The project uses Apache Spark for data processing and machine
    learning, and Streamlit for the interactive dashboard.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Apache Spark (PySpark)
    - Spark MLlib
    - Pandas and NumPy
    - Matplotlib
    - Streamlit
    """)

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    - Linear Regression
    - Random Forest Regression
    - Gradient-Boosted Tree Regression
    """)

    st.subheader("Project Workflow")

    st.markdown("""
    1. Dataset collection
    2. Data preprocessing
    3. Exploratory data analysis
    4. Stage-wise model training
    5. Model evaluation
    6. Dashboard development
    """)

    st.warning(
        "Model performance is not equally strong across all crop stages. "
        "The dashboard reports experimental results and should not be "
        "treated as a production irrigation controller."
    )

st.sidebar.divider()
st.sidebar.caption("Smart Irrigation System | By Adarsh Yadav")
