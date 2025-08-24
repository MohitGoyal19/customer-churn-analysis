# End-to-End MLOps Pipeline for Customer Churn Prediction

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.12-orange)
![DVC](https://img.shields.io/badge/DVC-3.49-blueviolet)

This repository contains the source code for a complete, end-to-end MLOps pipeline designed to predict customer churn. The project demonstrates best practices in data management and machine learning, from ingesting data from multiple sources to training and comparing multiple models in a reproducible and automated workflow.

## Key Features

* **Multi-Source Data Ingestion**: Ingests customer data from both a MongoDB database and local CSV files.
* **Automated Data Processing**: Includes automated steps for data validation, cleaning, and standardization.
* **Multi-Model Training & Comparison**: Trains, evaluates, and compares multiple models (Random Forest, Logistic Regression, Gradient Boosting) in a single run.
* **Experiment Tracking**: Uses **MLflow** to log all experiment parameters, metrics, and model artifacts for easy comparison and versioning.
* **Data Version Control**: Uses **DVC** to version large data files and feature sets, keeping the Git repository lightweight.
* **Centralized Logging & Configuration**: All pipeline parameters and credentials are centrally managed for easy configuration and debugging.
* **End-to-End Orchestration**: A master script (`run_pipeline.py`) automates the entire workflow from start to finish.

## Pipeline Workflow

The pipeline follows a modular, six-step workflow that ensures a clear separation of concerns and maintainability.



## Technology Stack

* **Language**: Python
* **Data Manipulation**: Pandas
* **Databases**: MongoDB
* **ML Framework**: Scikit-learn
* **Experiment Tracking**: MLflow
* **Data Versioning**: DVC
* **Code Versioning**: Git

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd customer-churn-pipeline
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up MongoDB**
    The easiest way to run a local MongoDB instance is with Docker:
    ```bash
    docker run --name churn-mongo -p 27017:27017 -d mongo
    ```
    Load your personal information data into a collection in this database.

5.  **Configure Credentials**
    Create a `.env` file in the project root by copying the example. Then, fill in your MongoDB details.
    ```
    MONGO_URI="mongodb://localhost:27017/"
    MONGO_DATABASE="your_db_name"
    MONGO_COLLECTION="your_collection_name"
    ```

6.  **Place Source Data**
    * Place your customer account data CSV file inside the `user_data/` directory and ensure it's named `account_data.csv`.

7.  **Initialize DVC**
    ```bash
    dvc init
    ```

## How to Run the Pipeline

**1. Configure the Target Column**
Before running, open `src/config.py` and ensure the `RAW_TARGET_COLUMN` variable is set to the exact name of the churn/attrition column in your `account_data.csv` file.

**2. Execute the Pipeline**
Run the master script from the root of the project directory:
```bash
python run_pipeline.py
