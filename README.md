
# Project Title

Data Processing Implementation - Student mental health

## Project Overview
A student depression dataset typically contains data aimed at analyzing, understanding, and predicting depression levels among students. It may include features such as demographic information (age, gender), academic performance (grades, attendance), lifestyle habits (sleep patterns, exercise, social activities), mental health history, and responses to standardized depression scales.
This project involves processing, analyzing, and engineering features from a student mental health dataset using PySpark. The pipeline is designed to:

- Load and clean data.
- Engineer relevant features.
- Perform data analysis and generate insights.
- Save processed outputs as Parquet files.

Dataset

The dataset used is from Kaggle 
[Student Mental Health Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset/data)

## Project Structure
```
project/ 
├── scripts/ 
│ ├── init.py 
│ ├── data_loading.py # Code for loading and cleaning the dataset 
│ ├── feature_engineering.py # Code for engineering features 
│ ├── analysis_outputs.py # Code for performing analysis and saving outputs 
├── test/ 
│ ├── init.py 
│ ├── test_data_loading.py # Unit tests for data_loading.py 
│ ├── test_feature_engineering.py # Unit tests for feature_engineering.py 
│ ├── test_analysis_outputs.py # Unit tests for analysis_outputs.py 
├── outputs/ 
│ ├── processed_data/ # Cleaned and transformed base dataset 
│ ├── feature_engineered_data/ # Dataset with derived features 
│ ├── distributions/ # Distribution statistics outputs 
│ ├── correlations/ # Correlation results 
│ ├── aggregations/ # Aggregated analysis results 
│ ├── risk_analysis/ # List of high-risk students 
├── requirements.txt # Python dependencies 
├── README.md # Project instructions and usage
```

## Setup Instructions

### Prerequisites

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **PySpark 3.0+**: [PySpark Installation Guide](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)
- **Java (for PySpark)**: Ensure Java 8 or later is installed ([Download Java](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html))
- **Virtual Environment (recommended)**: Use `venv` or `virtualenv` to isolate dependencies.

### Installation

Clone the repository:
```
git clone <https://github.com/stesija99/Data-Processing-Implementation.git>

cd project
```
Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Running the Pipeline
1. Set up environment variables (if necessary):
```
export PYTHONPATH=$(pwd)
```

2. Run individual scripts as needed in this order
- For data loading:
```
python scripts/data_loading.py
```
- For feature engineering:
```
python scripts/feature_engineering.py
```
- For analysis outputs:
```
python scripts/analysis_outputs.py
```
## Scripts Overview

### data_loading.py
- Responsibilities:

    - Load the dataset.
    - Handle missing values.
    - Convert data types where necessary.
    - Save cleaned data as processed_data.parquet.

### feature_engineering.py
- Responsibilities:

    - Create derived features:
    - Stress Index (weighted average).
    - Sleep Categories (Low, Normal, High).
    - Age Groups (18-21, 22-25, 26-30, >30).
    - Normalized numerical features.
    - Dummy variables for categorical columns.
    - Save engineered features as feature_engineered_data.parquet.

### analysis_outputs.py
- Responsibilities:

    - Generate distribution statistics.
    - Calculate correlations.
    - Perform aggregated analyses.
    - Identify high-risk students.
    - Save results as partitioned Parquet files.

## Testing
To execute unit tests:
```
pytest tests/
```

## Outputs
- Processed Data:
    - outputs/processed_data/processed_data.parquet
- Feature Engineered Data:
    - outputs/feature_engineered_data/
- Analysis Results:
    - Distribution statistics: outputs/distributions/
    - Correlation results: outputs/correlations/
    - Aggregated analyses: outputs/aggregations/
    - Risk analysis: outputs/risk_analysis/
## Notes
- Use Snappy compression for all Parquet files.
- Ensure partitioning for large datasets (e.g., by city or age group).
- Utilize caching for intermediate transformations in PySpark.

## License
MIT