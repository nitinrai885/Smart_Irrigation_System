import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer

# 1. Start Spark
spark = SparkSession.builder \
    .appName("SmartIrrigationPreprocessing") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark started successfully!")

# 2. Load dataset
df = spark.read.csv(
    "data/tomato irrigation dataset.csv",
    header=True,
    inferSchema=True
)

# 3. Rename columns
clean_names = [
    "temperature_c",
    "humidity_pct",
    "soil_moisture",
    "reference_evapotranspiration",
    "evapotranspiration",
    "crop_coefficient",
    "crop_coefficient_stage",
    "nitrogen",
    "phosphorus",
    "potassium",
    "solar_radiation",
    "wind_speed",
    "days_planted",
    "ph"
]

for old, new in zip(df.columns, clean_names):
    df = df.withColumnRenamed(old, new)

# 4. Remove rows with missing target values
df = df.filter(col("evapotranspiration").isNotNull())

# 5. Define numerical columns
numeric_cols = [
    "temperature_c",
    "humidity_pct",
    "soil_moisture",
    "reference_evapotranspiration",
    "crop_coefficient",
    "nitrogen",
    "phosphorus",
    "potassium",
    "solar_radiation",
    "wind_speed",
    "days_planted",
    "ph"
]

# 6. Handle missing numerical values using median
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[name + "_clean" for name in numeric_cols]
).setStrategy("median")

# 7. Convert crop growth stage into a number
indexer = StringIndexer(
    inputCol="crop_coefficient_stage",
    outputCol="crop_stage_index",
    handleInvalid="keep"
)

# 8. Create preprocessing pipeline
pipeline = Pipeline(stages=[imputer, indexer])

model = pipeline.fit(df)
processed_df = model.transform(df)

# 9. Display processed data
print("\nProcessed Dataset:")
processed_df.select(
    "temperature_c_clean",
    "humidity_pct_clean",
    "soil_moisture_clean",
    "crop_coefficient_stage",
    "crop_stage_index",
    "evapotranspiration"
).show(10, truncate=False)

# 10. Check final record count
print("Total Processed Records:", processed_df.count())

# 11. Save processed dataset

os.makedirs("data", exist_ok=True)

output_path = "data/processed_irrigation.csv"

processed_df.toPandas().to_csv(output_path, index=False)

print("\nProcessed dataset saved successfully!")
print("Saved location:", output_path)

# 12. Display final results
print("Total Processed Records:", processed_df.count())

print("\nPreprocessing completed successfully!")
print("Dataset is ready for model training!")

