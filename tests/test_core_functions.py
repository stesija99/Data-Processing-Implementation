import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col
from scripts.feature_engineering import calculate_stress_index
from scripts.feature_engineering import categorize_sleep
from scripts.feature_engineering import create_age_groups
from scripts.feature_engineering import normalize_features
from scripts.feature_engineering import create_dummy_variables
from scripts.analysis_outputs import compute_aggregations
from scripts.analysis_outputs import save_outputs

# Fixtures for SparkSession
@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.master("local[*]").appName("PySpark Unit Tests").getOrCreate()

# Test Data
@pytest.fixture
def test_data(spark):
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("Depression", IntegerType(), True),
        StructField("Academic Pressure", DoubleType(), True),
        StructField("Work Pressure", DoubleType(), True),
        StructField("Financial Stress", DoubleType(), True),
        StructField("Sleep Duration", DoubleType(), True),
        StructField("Age", IntegerType(), True),
        StructField("CGPA", DoubleType(), True),
        StructField("Profession", StringType(), True),
        StructField("City", StringType(), True),
        StructField("Degree", StringType(), True)
    ])
    data = [
        (1, 0, 8.0, 5.0, 7.0, 6.0, 20, 3.5, "Student", "Mumbai","B.Pharm"),
        (2, 0, 6.0, 3.0, 8.0, 4.0, 23, 3.8, "Engineer","Nagpur","BSc"),
        (3, 1, None, 4.0, 5.0, 9.0, 26, 3.2, "Doctor", "Bhopal","BA")
    ]
    return spark.createDataFrame(data, schema)

# Test: Data Loading
def test_load_data(test_data):
    assert test_data.count() == 3
    assert len(test_data.columns) == 11

# Test: Stress Index Calculation
def test_calculate_stress_index(spark, test_data):
    result = calculate_stress_index(test_data)
    assert "Stress Index" in result.columns
    assert result.filter(col("Stress Index").isNull()).count() == 1  # Check null handling
    assert result.count() == test_data.count()

# Test: Sleep Categories
def test_create_sleep_categories(spark, test_data):
    result = categorize_sleep(test_data)
    assert "Sleep Category" in result.columns
    sleep_categories = [row["Sleep Category"] for row in result.select("Sleep Category").distinct().collect()]
    assert set(sleep_categories) == {"Low", "Normal", "High"}

# Test: Age Groups
def test_categorize_age_groups(spark, test_data):
    result = create_age_groups(test_data)
    assert "Age Group" in result.columns
    age_groups = [row["Age Group"] for row in result.select("Age Group").distinct().collect()]
    assert set(age_groups) == {"18-21", "22-25", "26-30"}

# Test: Normalization
def test_normalize_features(test_data, numeric_cols=["Academic Pressure", "Work Pressure", "Financial Stress", "Sleep Duration", "CGPA"]):
    result = normalize_features(test_data,numeric_cols)
    assert any(col.endswith("_normalized") for col in result.columns)
    for col_name in numeric_cols:
        normalized_col = f"{col_name}_normalized"
        if normalized_col in result.columns:
            assert result.filter(result[normalized_col] > 1).count() == 0
            assert result.filter(result[normalized_col] < 0).count() == 0

# Test: Dummy Variables
def test_create_dummy_variables(spark, test_data):
    result = create_dummy_variables(test_data, categorical_cols=["Profession"])
    assert "Profession_Student" in result.columns
    assert "Profession_Engineer" in result.columns

# Test: Save Outputs
def test_save_outputs(spark, test_data, tmp_path):
    # Prepare DataFrames and paths
    dfs = [test_data]
    output_paths = [str(tmp_path / "test_output.parquet")]

    save_outputs(dfs, output_paths, spark)
    loaded_df = spark.read.parquet(output_paths[0])
    assert loaded_df.count() == test_data.count()
    assert set(loaded_df.columns) == set(test_data.columns)
