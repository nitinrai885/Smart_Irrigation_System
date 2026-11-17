


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator
import os


# 1. Start Spark
spark = SparkSession.builder \
    .appName("SmartIrrigationStageWiseTraining") \
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

print("Dataset loaded successfully!")
print("Total records:", df.count())


# 3. Define features and target
feature_cols = [
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

target_col = "evapotranspiration"


# 4. Prepare assembler
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)


# 5. Define models
models = {
    "LinearRegression": LinearRegression(
        featuresCol="features",
        labelCol=target_col,
        maxIter=100
    ),

    "RandomForest": RandomForestRegressor(
        featuresCol="features",
        labelCol=target_col,
        numTrees=100,
        maxDepth=10,
        seed=42
    ),

    "GBT": GBTRegressor(
        featuresCol="features",
        labelCol=target_col,
        maxIter=100,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )
}


# 6. Create evaluators
rmse_evaluator = RegressionEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="r2"
)


# 7. Create model folder
os.makedirs("models", exist_ok=True)


# 8. Train models separately for each stage
stages = [
    "Initial Stage",
    "Development Stage",
    "Mid stage",
    "Last stage"
]

results = []

for stage in stages:

    print("\n====================================")
    print("CROP STAGE:", stage)
    print("====================================")

    stage_df = df.filter(
        col("crop_coefficient_stage") == stage
    )

    total_records = stage_df.count()

    print("Total records:", total_records)

    if total_records < 10:
        print("Not enough records. Skipping this stage.")
        continue

    # Split data
    train_data, test_data = stage_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_count = train_data.count()
    test_count = test_data.count()

    print("Training records:", train_count)
    print("Testing records:", test_count)

    if train_count == 0 or test_count < 2:
        print("Not enough training/testing records.")
        continue

    # Calculate baseline RMSE using training mean
    training_mean = train_data.agg(
        {"evapotranspiration": "avg"}
    ).first()[0]

    baseline_predictions = test_data.withColumn(
        "prediction",
        lit(training_mean)
    )

    baseline_rmse = rmse_evaluator.evaluate(
        baseline_predictions
    )

    print("Baseline RMSE:", round(baseline_rmse, 4))

    # Train each algorithm
    for model_name, estimator in models.items():

        print("\nTraining:", model_name)

        pipeline = Pipeline(
            stages=[assembler, estimator]
        )

        model = pipeline.fit(train_data)

        predictions = model.transform(test_data)

        rmse = rmse_evaluator.evaluate(predictions)
        r2 = r2_evaluator.evaluate(predictions)

        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 4))

        # Save trained model
        print("Model evaluation completed:", model_name)

        results.append({
            "stage": stage,
            "model": model_name,
            "rmse": rmse,
            "r2": r2,
            "baseline_rmse": baseline_rmse
        })


# 9. Final comparison
print("\n\n========== FINAL MODEL COMPARISON ==========")

for stage in stages:

    stage_results = [
        r for r in results if r["stage"] == stage
    ]

    if not stage_results:
        continue

    print("\nStage:", stage)

    for result in sorted(
        stage_results,
        key=lambda x: x["rmse"]
    ):
        print(
            result["model"],
            "| RMSE:", round(result["rmse"], 4),
            "| R2:", round(result["r2"], 4)
        )

    baseline = stage_results[0]["baseline_rmse"]

    best = min(stage_results, key=lambda x: x["rmse"])

    print("Baseline RMSE:", round(baseline, 4))
    print("Lowest model RMSE:", round(best["rmse"], 4))

    if best["rmse"] < baseline:
        print("Result: Best model beats the baseline.")
    else:
        print("Result: Baseline performs better than all models.")


print("\nModel training completed successfully!")
print("All trained models are saved in the models folder.")

spark.stop()


