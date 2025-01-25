from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, corr, desc, count, avg, stddev, when
from typing import List, Union
import os

def create_spark_session():
    """Create and return a SparkSession."""
    return SparkSession.builder.appName("Analysis Outputs").getOrCreate()

def compute_distributions(df: DataFrame) -> DataFrame:
    """Compute distribution statistics."""
    depression_by_age_profession = df.groupBy("Age Group", "Profession").agg(
        avg("Depression").alias("Avg Depression"),
        stddev("Depression").alias("StdDev Depression")
    )

    cgpa_by_sleep_category = df.groupBy("Sleep Category").agg(
        avg("CGPA").alias("Avg CGPA"),
        stddev("CGPA").alias("StdDev CGPA")
    )

    return depression_by_age_profession, cgpa_by_sleep_category

def compute_correlations(df, numeric_cols):
    """Compute correlation matrix and top factors correlated with Depression."""
    # Ensure all numeric columns are cast to double
    for col_name in numeric_cols:
        df = df.withColumn(col_name, col(col_name).cast("double"))
    
    # Compute correlation matrix
    correlation_matrix = [
        (col1, col2, df.stat.corr(col1, col2))
        for i, col1 in enumerate(numeric_cols)
        for col2 in numeric_cols[i:]
    ]

    # Calculate correlations with Depression
    depression_correlations = [
        (col_name, df.stat.corr("Depression", col_name)) for col_name in numeric_cols if col_name != "Depression"]

    # Sort and pick the top 5 correlated features
    top_depression_correlations = sorted(depression_correlations, key=lambda x: abs(x[1]), reverse=True)[:5]
    return correlation_matrix, top_depression_correlations


def compute_aggregations(df: DataFrame) -> DataFrame:
    """Compute aggregated results."""
    depression_by_city_degree = df.groupBy("City", "Degree").agg(
        avg("Depression").alias("Avg Depression"),
        stddev("Depression").alias("StdDev Depression")
    )

    stress_by_age_gender = df.groupBy("Age Group", "Gender").agg(
        avg("Stress Index").alias("Avg Stress Index"),
        stddev("Stress Index").alias("StdDev Stress Index")
    )

    performance_by_sleep_category = df.groupBy("Sleep Category").agg(
        avg("Academic Pressure").alias("Avg Academic Pressure"),
        avg("Work Pressure").alias("Avg Work Pressure"),
        avg("CGPA").alias("Avg CGPA")
    )

    return depression_by_city_degree, stress_by_age_gender, performance_by_sleep_category

def compute_risk_analysis(df: DataFrame) -> DataFrame:
    """Identify high-risk students based on specific criteria."""
    high_risk_students = df.filter(
        (col("Stress Index") > 0.7) &
        (col("Sleep Duration") < 6) &
        (col("Study Satisfaction") < 3) &
        (col("Financial Stress") > 7)
    )

    return high_risk_students

def save_outputs(dfs: List[Union[DataFrame, List[tuple]]], output_paths: List[str], spark: SparkSession) -> None:
    """Save DataFrames and other outputs to specified paths."""
    for df, path in zip(dfs, output_paths):
        if isinstance(df, list):
            # Check if the list contains tuples of length 3 or 2
            if all(len(row) == 3 for row in df):
                schema = ["Column1", "Column2", "Correlation"]
            elif all(len(row) == 2 for row in df):
                schema = ["Column1", "Correlation"]
            else:
                raise ValueError(f"Unsupported tuple length in list for path: {path}")
            
            # Convert the list to a DataFrame
            df = spark.createDataFrame([Row(*row) for row in df], schema=schema)
        
        # Save the DataFrame as Parquet
        df.write.mode("overwrite").parquet(path, compression="snappy")

if __name__ == "__main__":
    # Initialize Spark session
    spark = create_spark_session()

    # Define input and output paths
    input_path = os.path.join("outputs", "feature_engineered_data")
    output_base_path = "outputs"

    # Load the feature-engineered data
    df = spark.read.parquet(input_path)

    # Compute distributions
    depression_by_age_profession, cgpa_by_sleep_category = compute_distributions(df)

    # Compute correlations
    numeric_cols = [
        "Academic Pressure", "Work Pressure", "Financial Stress", "Sleep Duration", "Depression", "CGPA"]
    correlation_matrix, top_depression_correlations = compute_correlations(df, numeric_cols)

    # Compute aggregations
    depression_by_city_degree, stress_by_age_gender, performance_by_sleep_category = compute_aggregations(df)

    # Compute risk analysis
    high_risk_students = compute_risk_analysis(df)

    # Save all outputs
    save_outputs(
        [
            depression_by_age_profession, 
            cgpa_by_sleep_category,
            correlation_matrix, 
            top_depression_correlations,
            depression_by_city_degree, 
            stress_by_age_gender, 
            performance_by_sleep_category,
            high_risk_students
        ],
        [
            os.path.join(output_base_path, "distributions", "depression_by_demographics.parquet"),
            os.path.join(output_base_path, "distributions", "academic_performance.parquet"),
            os.path.join(output_base_path, "correlations", "correlation_matrix.parquet"),
            os.path.join(output_base_path, "correlations", "depression_correlations.parquet"),
            os.path.join(output_base_path, "aggregations", "city_degree_stats.parquet"),
            os.path.join(output_base_path, "aggregations", "demographic_stress.parquet"),
            os.path.join(output_base_path, "aggregations", "sleep_performance.parquet"),
            os.path.join(output_base_path, "risk_analysis", "high_risk_students.parquet")
        ],spark=spark
    )

    # Stop Spark session
    spark.stop()
