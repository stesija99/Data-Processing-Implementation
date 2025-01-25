from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, udf, min, max
from typing import List
import os

def create_spark_session():
    """Create and return a SparkSession."""
    return SparkSession.builder.appName("Feature Engineering").getOrCreate()

def calculate_stress_index(df: DataFrame) -> DataFrame:
    """Calculate the Stress Index as a weighted average."""
    return df.withColumn(
        "Stress Index",
        (col("Academic Pressure") * 0.33 +
         col("Work Pressure") * 0.33 +
         col("Financial Stress") * 0.33)
    )

def categorize_sleep(df: DataFrame) -> DataFrame:
    """Create Sleep Categories: Low, Normal, High."""
    return df.withColumn(
        "Sleep Category",
        when(col("Sleep Duration") < 6, "Low")
        .when((col("Sleep Duration") >= 6) 
        & (col("Sleep Duration") <= 8), "Normal")
        .otherwise("High")
    )

def create_age_groups(df: DataFrame) -> DataFrame:
    """Create Age Groups: 18-21, 22-25, 26-30, >30."""
    return df.withColumn(
        "Age Group",
        when((col("Age") >= 18) & (col("Age") <= 21), "18-21")
        .when((col("Age") >= 22) & (col("Age") <= 25), "22-25")
        .when((col("Age") >= 26) & (col("Age") <= 30), "26-30")
        .otherwise(">30")
    )

def normalize_features(df: DataFrame, numeric_cols: List[str]) -> DataFrame:
    """Normalize numerical features to a range of 0-1."""
    # Collect min and max for all numeric columns
    min_max = df.select(
        *[min(c).alias(f"{c}_min") for c in numeric_cols],
        *[max(c).alias(f"{c}_max") for c in numeric_cols]
    ).collect()[0]

    for col_name in numeric_cols:
        # Convert min and max to numeric types
        min_val = float(min_max[f"{col_name}_min"])
        max_val = float(min_max[f"{col_name}_max"])

        # Add normalized column
        df = df.withColumn(
            f"{col_name}_normalized",
            (col(col_name).cast("double") - min_val) / (max_val - min_val)
        )

    return df

def create_dummy_variables(df: DataFrame, categorical_cols: List[str]) -> DataFrame:
    """Generate dummy variables for categorical columns."""
    for col_name in categorical_cols:
        categories = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
        for category in categories:
            df = df.withColumn(
                f"{col_name}_{category}",
                when(col(col_name) == category, 1).otherwise(0)
            )
    return df

def save_featured_data(df: DataFrame, output_path: str)-> None:
    """Save the dataset with engineered features as a Parquet file."""
    df.write.mode("overwrite").parquet(output_path, compression="snappy")

if __name__ == "__main__":
    # Initialize Spark session
    spark = create_spark_session()

    # Define input and output paths
    input_path = os.path.join("outputs", "processed_data")
    output_path = os.path.join("outputs", "feature_engineered_data")

    # Load the cleaned data
    try:
        df = spark.read.parquet(input_path)
    except Exception as e:
        print(f"Error reading Parquet file at {input_path}: {e}")
        raise
    
    # Add new features
    df = calculate_stress_index(df)
    df = categorize_sleep(df)
    df = create_age_groups(df)

    # Normalize numerical features
    numeric_cols = ["Academic Pressure", "Work Pressure", "Financial Stress", "Sleep Duration", "Depression", "CGPA"]
    df = normalize_features(df, numeric_cols)

    # Create dummy variables for categorical columns
    categorical_cols = ["Gender", "City"]
    df = create_dummy_variables(df, categorical_cols)

    # Save the feature-engineered dataset
    save_featured_data(df, output_path)

    # Stop Spark session
    spark.stop()
