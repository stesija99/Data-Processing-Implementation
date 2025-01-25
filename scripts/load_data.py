from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, when, isnan, mean, stddev
import os

def create_spark_session():
    """Create and return a SparkSession."""
    return SparkSession.builder.appName("Student Depression Data Processing").getOrCreate()

def load_data(spark: SparkSession, input_path: str) -> DataFrame:
    """Load the CSV file into a PySpark DataFrame."""
    try:
        return spark.read.csv(input_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error reading CSV file at {input_path}: {e}")
        raise

def clean_data(df: DataFrame) -> DataFrame:
    """Clean and preprocess the DataFrame."""
    # Handle missing values by dropping rows with critical missing fields
    # df = df.dropna(how='any', subset=['Age', 'Gender', 'Sleep Duration', 'Depression'])
    df = df.na.drop()
    
    # Convert 'Sleep Duration' to numeric (if not already inferred)
    df = df.withColumn("Sleep Duration", col("Sleep Duration").cast("double"))

    # Remove rows with inconsistent or invalid values
    df = df.filter((col("Sleep Duration") > 0) & (col("Sleep Duration") <= 24))

    return df

def profile_data(df: DataFrame) -> DataFrame:
    """Generate data quality metrics."""
    # Null counts per column
    null_counts = df.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns
    ])

    # Value counts for categorical columns
    value_counts = {
        col_name: df.groupBy(col_name).count().orderBy('count', ascending=False).limit(5)
        for col_name, dtype in df.dtypes if dtype == 'string'
    }

    # Descriptive stats for numerical columns
    numeric_cols = [c for c, dtype in df.dtypes if dtype in ('int', 'double')]
    stats = df.select(*[mean(c).alias(f"{c}_mean") for c in numeric_cols],
                      *[stddev(c).alias(f"{c}_stddev") for c in numeric_cols])

    return null_counts, value_counts, stats

def save_processed_data(df: DataFrame, output_path: str)-> None:
    """Save cleaned data as a Parquet file with Snappy compression."""
    try:
        df.write.mode("overwrite").parquet(output_path, compression="snappy")
    except Exception as e:
        print(f"Error saving output to {output_path}: {e}")
        raise

if __name__ == "__main__":
    # Initialize Spark session
    spark = create_spark_session()

    # Define input and output paths
    input_path = os.path.join("data", "Student_Depression_Dataset.csv")
    output_path = os.path.join("outputs", "processed_data")

    # Load the dataset
    raw_data = load_data(spark, input_path)

    # Clean the dataset
    processed_data = clean_data(raw_data)

    # Profile the dataset
    null_counts, value_counts, stats = profile_data(processed_data)

    # Display profiling results
    print("\n=== Null Counts ===")
    null_counts.show()

    print("\n=== Value Counts (Top 5) ===")
    for col_name, counts in value_counts.items():
        print(f"\n{col_name}:")
        counts.show()

    print("\n=== Descriptive Stats ===")
    stats.show()

    # Save the processed dataset
    save_processed_data(processed_data, output_path)

    # Stop Spark session
    spark.stop()
