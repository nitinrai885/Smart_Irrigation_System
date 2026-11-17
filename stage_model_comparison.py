
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import matplotlib.pyplot as plt
import os


# 1. Start Spark
spark = SparkSession.builder \
    .appName("StageModelAnalysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark started successfully!")


# 2. Load processed dataset
df = spark.read.csv(
    "data/processed_irrigation.csv",
    header=True,
    inferSchema=True
)

print("Dataset loaded:", df.count(), "records")


# 3. Features and target
features = [
    "temperature_c_clean",
    "humidity_pct_clean",
    "soil_moisture_clean",
    "reference_evapotranspiration_clean",
    "crop_coefficient_clean",
    "nitrogen_clean",
    "phosphorus_clean",
    "potassium_clean",
    "solar_radiation_clean",
    "wind_speed_clean",
    "days_planted_clean",
    "ph_clean",
    "crop_stage_index"
]

target = "evapotranspiration"
stage_column = "crop_coefficient_stage"

stages = [
    "Initial Stage",
    "Development Stage",
    "Mid stage",
    "Last stage"
]


# 4. Output folders
os.makedirs("analysis", exist_ok=True)
os.makedirs("analysis/charts", exist_ok=True)


# 5. Evaluation metric
evaluator = RegressionEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="rmse"
)

correlation_results = []
importance_results = []


# 6. Analyze each crop stage
for stage in stages:

    print("\n====================================")
    print("ANALYZING:", stage)
    print("====================================")

    stage_df = df.filter(
        col(stage_column) == stage
    ).filter(
        col(target).isNotNull()
    ).cache()

    total_records = stage_df.count()

    print("Total records:", total_records)

    if total_records < 30:
        print("Not enough records. Skipping.")
        stage_df.unpersist()
        continue

    # --------------------------------
    # A. Feature correlation
    # --------------------------------

    print("\nFeature Correlations:")

    for feature in features:

        correlation = stage_df.stat.corr(
            feature,
            target
        )

        if correlation is not None:
            correlation_results.append({
                "Crop Stage": stage,
                "Feature": feature,
                "Correlation": correlation
            })

            print(feature, ":", round(correlation, 4))

    # --------------------------------
    # B. Train Random Forest
    # --------------------------------

    train_df, test_df = stage_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    if train_df.count() == 0 or test_df.count() == 0:
        print("Insufficient train/test records.")
        stage_df.unpersist()
        continue

    imputer = Imputer(
        inputCols=features,
        outputCols=[f"{c}_imputed" for c in features],
        strategy="median"
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_imputed" for c in features],
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=target,
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[
        imputer,
        assembler,
        rf
    ])

    fitted_model = pipeline.fit(train_df)

    predictions = fitted_model.transform(test_df)

    rmse = evaluator.evaluate(predictions)

    print("\nRandom Forest RMSE:", round(rmse, 4))

    # --------------------------------
    # C. Feature Importance
    # --------------------------------

    rf_model = fitted_model.stages[-1]

    importances = rf_model.featureImportances.toArray()

    importance_list = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 5 Important Features:")

    for feature, importance in importance_list[:5]:

        print(feature, ":", round(float(importance), 4))

        importance_results.append({
            "Crop Stage": stage,
            "Feature": feature,
            "Importance": float(importance)
        })

    # --------------------------------
    # D. Actual vs Predicted Chart
    # --------------------------------

    prediction_pdf = predictions.select(
        target,
        "prediction"
    ).toPandas()

    plt.figure(figsize=(7, 5))

    plt.scatter(
        prediction_pdf[target],
        prediction_pdf["prediction"],
        alpha=0.6
    )

    min_value = min(
        prediction_pdf[target].min(),
        prediction_pdf["prediction"].min()
    )

    max_value = max(
        prediction_pdf[target].max(),
        prediction_pdf["prediction"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Evapotranspiration")
    plt.ylabel("Predicted Evapotranspiration")
    plt.title(f"Actual vs Predicted - {stage}")
    plt.tight_layout()

    filename = stage.lower().replace(" ", "_")

    plt.savefig(
        f"analysis/charts/{filename}_actual_vs_predicted.png"
    )

    plt.close()

    stage_df.unpersist()


# 7. Save feature correlations
if correlation_results:

    correlation_df = pd.DataFrame(correlation_results)

    correlation_df.to_csv(
        "analysis/stage_feature_correlations.csv",
        index=False
    )


# 8. Save feature importance
if importance_results:

    importance_df = pd.DataFrame(importance_results)

    importance_df.to_csv(
        "analysis/stage_feature_importance.csv",
        index=False
    )


# 9. Feature importance charts
if importance_results:

    for stage in stages:

        stage_data = importance_df[
            importance_df["Crop Stage"] == stage
        ].sort_values(
            "Importance",
            ascending=True
        )

        if stage_data.empty:
            continue

        plt.figure(figsize=(8, 5))

        plt.barh(
            stage_data["Feature"],
            stage_data["Importance"]
        )

        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importance - {stage}")
        plt.tight_layout()

        filename = stage.lower().replace(" ", "_")

        plt.savefig(
            f"analysis/charts/{filename}_feature_importance.png"
        )

        plt.close()


spark.stop()

print("\n====================================")
print("STAGE MODEL ANALYSIS COMPLETED!")
print("Results saved in analysis folder.")
print("====================================")
