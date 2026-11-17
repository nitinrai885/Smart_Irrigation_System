
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import os


# 1. Start Spark
spark = SparkSession.builder \
    .appName("StageWiseIrrigationModel") \
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


# 4. Crop stages
stages = [
    "Initial Stage",
    "Development Stage",
    "Mid stage",
    "Last stage"
]


# 5. Evaluation metrics
rmse_evaluator = RegressionEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="rmse"
)

r2_evaluator = RegressionEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="r2"
)


# 6. Store results
all_results = []


# 7. Train models stage by stage
for stage in stages:

    print("\n====================================")
    print("CROP STAGE:", stage)
    print("====================================")

    stage_df = df.filter(
        col(stage_column) == stage
    ).select(*(features + [target])).cache()

    # Remove records with missing target
    stage_df = stage_df.filter(
        col(target).isNotNull()
    ).cache()

    total_records = stage_df.count()

    print("Total records:", total_records)

    if total_records < 30:
        print("Not enough records. Skipping.")
        stage_df.unpersist()
        continue

    # 8. Train-test split
    train_df, test_df = stage_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print("Training records:", train_count)
    print("Testing records:", test_count)

    if train_count == 0 or test_count == 0:
        print("Insufficient train/test records. Skipping.")
        stage_df.unpersist()
        continue

    # 9. Baseline prediction
    average_et = train_df.selectExpr(
        f"avg({target})"
    ).first()[0]

    baseline_predictions = test_df.withColumn(
        "prediction",
        lit(float(average_et))
    )

    baseline_rmse = rmse_evaluator.evaluate(
        baseline_predictions
    )

    baseline_r2 = r2_evaluator.evaluate(
        baseline_predictions
    )

    print("\nBaseline RMSE:", round(baseline_rmse, 4))
    print("Baseline R2:", round(baseline_r2, 4))

    # 10. Fresh model estimators for each stage
    models = {
        "Linear Regression": LinearRegression(
            featuresCol="features",
            labelCol=target,
            maxIter=100
        ),

        "Random Forest": RandomForestRegressor(
            featuresCol="features",
            labelCol=target,
            numTrees=100,
            maxDepth=10,
            seed=42
        ),

        "Gradient Boosted Trees": GBTRegressor(
            featuresCol="features",
            labelCol=target,
            maxIter=100,
            maxDepth=5,
            seed=42
        )
    }

    # 11. Preprocessing
    imputer = Imputer(
        inputCols=features,
        outputCols=[f"{c}_imputed" for c in features],
        strategy="median"
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_imputed" for c in features],
        outputCol="features"
    )

    # 12. Train and evaluate models
    stage_results = []

    for model_name, estimator in models.items():

        print("\nTraining:", model_name)

        pipeline = Pipeline(stages=[
            imputer,
            assembler,
            estimator
        ])

        fitted_model = pipeline.fit(train_df)

        predictions = fitted_model.transform(test_df)

        rmse = rmse_evaluator.evaluate(predictions)
        r2 = r2_evaluator.evaluate(predictions)

        beats_baseline = rmse < baseline_rmse

        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))
        print("Beats baseline:", beats_baseline)

        result = {
            "Crop Stage": stage,
            "Model": model_name,
            "Total Records": total_records,
            "Training Records": train_count,
            "Testing Records": test_count,
            "Baseline RMSE": baseline_rmse,
            "RMSE": rmse,
            "R2": r2,
            "Beats Baseline": beats_baseline
        }

        stage_results.append(result)
        all_results.append(result)

    # 13. Show best model for this stage
    best_result = min(
        stage_results,
        key=lambda x: x["RMSE"]
    )

    print("\nBest model:", best_result["Model"])
    print("Best RMSE:", round(best_result["RMSE"], 4))

    if best_result["Beats Baseline"]:
        print("Result: Best model beats baseline.")
    else:
        print("Result: Baseline performs better than all models.")

    stage_df.unpersist()


# 14. Save experiment results as CSV
if all_results:

    os.makedirs("models", exist_ok=True)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "models/stage_wise_results.csv",
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(
        results_df[
            ["Crop Stage", "Model", "RMSE", "R2", "Beats Baseline"]
        ].to_string(index=False)
    )

    print("\nResults saved to models/stage_wise_results.csv")

else:
    print("No model results were generated.")

spark.stop()

print("\nStage-wise model training completed!")
