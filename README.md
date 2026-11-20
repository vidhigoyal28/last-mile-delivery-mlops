# 🚚 An MLOps-Based Last-Mile Delivery Duration Prediction System

### An End-to-End Machine Learning and MLOps Project Using XGBoost, MLflow, FastAPI, Docker, GitHub Actions and AWS ECR

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-orange" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue" alt="MLflow"/>
  <img src="https://img.shields.io/badge/API-FastAPI-green?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Containerization-Docker-blue?logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success?logo=githubactions" alt="GitHub Actions"/>
  <img src="https://img.shields.io/badge/Cloud-AWS%20ECR-orange?logo=amazonaws" alt="AWS ECR"/>
</p>

<p align="center">
  <a href="https://github.com/vidhigoyal28/last-mile-delivery-mlops/actions">View GitHub Actions</a> |
  <a href="https://github.com/vidhigoyal28/last-mile-delivery-mlops">GitHub Repository</a>
</p>

---

## 📌 1. Project Overview

Last-mile delivery is the final stage of the logistics process, where packages are transported from a distribution center or local hub to the customer's destination.

Accurately predicting delivery duration is important for improving logistics planning, estimating delivery times, and increasing operational efficiency.

This project develops a Machine Learning-based system that predicts delivery duration using historical courier delivery information, geographical features, delivery regions, and temporal patterns.

Beyond traditional model development, the project implements an MLOps workflow to manage the complete model lifecycle, including data validation, model training, experiment tracking, evaluation, retraining, and controlled model promotion.

The project also integrates Docker, GitHub Actions, and Amazon Elastic Container Registry (ECR) to automate testing, container image building, and publishing.

### Key Highlights

- Large-scale delivery data processing.
- XGBoost-based delivery duration prediction.
- Reusable Scikit-learn preprocessing pipeline.
- MLflow experiment tracking.
- Incoming data validation.
- Champion and candidate model evaluation.
- Automated candidate retraining.
- Quality-gate-based model promotion.
- FastAPI inference service.
- Docker containerization.
- GitHub Actions CI/CD automation.
- AWS ECR integration for Docker image publishing.

## 🎯 2. Project Objectives

- Predict last-mile delivery duration using historical delivery records.
- Develop a reliable regression model using XGBoost.
- Implement reusable preprocessing and training pipelines.
- Track experiments and training using MLflow.
- Validate incoming delivery data before retraining.
- Evaluate the existing champion model on incoming labelled data.
- Retrain candidate models using additional training records.
- Compare champion and candidate model performance.
- Implement a quality gate for controlled model promotion.
- Containerize the application using Docker.
- Automate testing and image publishing using GitHub Actions.
- Integrate AWS ECR for container image management.

## 📊 3. Dataset

**Dataset:** LaDe-D (Last-mile Delivery Dataset)

The project uses approximately 4.5 million courier delivery records collected across five cities in China.

The dataset contains geographical, temporal, and delivery-related information that helps identify patterns affecting delivery duration.

### Input Features

| Feature | Description |
|---|---|
| `city` | City where the delivery takes place |
| `region_id` | Delivery region identifier |
| `aoi_type` | Area of interest type |
| `accept_hour` | Hour at which the delivery is accepted |
| `accept_dayofweek` | Day of the week |
| `accept_month` | Month of acceptance |
| `distance_km` | Delivery distance in kilometres |
| `lat` | Latitude |
| `lng` | Longitude |

**Target Variable:** `target_duration`

The model predicts a continuous numerical delivery duration. Therefore, this is a **supervised regression problem**.

### Dataset Splitting

Chronological splitting is used to simulate predictions on future delivery records.

| Dataset | Period | Purpose |
|---|---|---|
| Training | May–August 2022 | Model training |
| Validation | September 2022 | Model validation |
| Testing | October 2022 | Model evaluation |

Chronological splitting helps simulate real-world prediction scenarios in which models trained on historical data are evaluated on later records.

## ⚙️ 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Data Preprocessing | ColumnTransformer, SimpleImputer, OneHotEncoder |
| Model Serialization | Joblib |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Tracking Database | SQLite |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Container Registry | AWS ECR |
| Data Storage | Parquet |
| Version Control | Git, GitHub |

## 🧠 5. Machine Learning Pipeline

The project uses a Scikit-learn Pipeline that combines preprocessing transformations with an XGBoost regression model.

### Data Preprocessing

**Categorical Features**

- Missing values are handled using `SimpleImputer(strategy="most_frequent")`.
- Categorical variables are encoded using `OneHotEncoder(handle_unknown="ignore")`.

**Numerical Features**

- Missing values are handled using `SimpleImputer(strategy="median")`.

**ColumnTransformer**

`ColumnTransformer` applies appropriate preprocessing operations to numerical and categorical feature groups.

This ensures that the same transformations are applied consistently during training and inference.

### Model: XGBoost Regressor

The tuned XGBoost model uses the following hyperparameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.03 |
| `min_child_weight` | 3 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |

The complete preprocessing and model pipeline is serialized using Joblib.

**Champion model artifact:** `models/tuned_xgboost_model.joblib`

## 📈 6. Model Evaluation

The model is evaluated using three regression metrics.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted delivery durations.

**Root Mean Squared Error (RMSE)**

Penalizes larger prediction errors more heavily.

**R-squared (R²)**

Measures the proportion of variance in the target variable explained by the model.

### Original Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 129.970 | 397.283 | -0.0421 |
| Random Forest | 125.315 | 350.647 | 0.1882 |
| XGBoost | 124.713 | 350.025 | 0.1911 |
| Tuned XGBoost | 125.234 | 349.393 | 0.1940 |

The tuned XGBoost model was retained as the champion model based on the overall model-selection process.

*Note: These are historical experiment results. The October test data was also used in earlier experimentation, so these results should not be interpreted as a completely untouched final benchmark.*

## 🔄 7. End-to-End MLOps Workflow

The project implements an incoming-data evaluation and retraining workflow.

### Task A: Incoming Data Validation

- Load incoming labelled delivery data.
- Validate required columns.
- Check missing values.
- Identify duplicate records.
- Validate target values.
- Generate a JSON validation report.

Missing feature values are allowed because the preprocessing pipeline includes imputation. Missing target values are rejected.

### Task B: Champion Model Evaluation

- Load the existing champion model.
- Generate predictions on incoming holdout records.
- Calculate MAE, RMSE, and R².
- Save evaluation results in JSON format.

### Task C: Candidate Model Retraining

- Load the original training dataset.
- Combine original training records with incoming training records.
- Clone the existing champion preprocessing and model pipeline.
- Retrain the candidate model.
- Save the candidate separately.
- Track the training experiment using MLflow.

### Task D: Automated Quality Gate

The candidate model is compared against the existing champion using predefined project-level acceptance criteria.

| Quality Check | Acceptance Criteria |
|---|---|
| Evaluation records | Same number of records |
| MAE | At least 1% improvement |
| RMSE | Must not worsen by more than 2% |
| R² | Must not decrease by more than 0.01 |

The quality gate generates a promotion decision in JSON format.

- If all conditions pass, the candidate becomes eligible for promotion.
- If any condition fails, the existing champion is retained.
- The champion is never automatically overwritten.

These thresholds are project-defined acceptance criteria, not universal ML standards.

## 🏆 8. Champion vs Candidate Model Results

For the practical assessment, a simulated incoming batch of 100,000 records was created from historical October test data.

The batch was divided into:

| Dataset | Records |
|---|---:|
| Incoming training data | 80,000 |
| Incoming holdout data | 20,000 |
| Total incoming batch | 100,000 |

The candidate model was trained using the original training data combined with the 80,000 incoming training records.

Both champion and candidate were evaluated on the same 20,000-record holdout dataset.

### Performance Comparison

| Metric | Champion | Candidate |
|---|---:|---:|
| MAE | 126.406 | **124.480** |
| RMSE | 371.388 | **366.935** |
| R² | 0.1733 | **0.1930** |

### Quality Gate Results

| Quality Check | Result |
|---|---|
| Same evaluation records | PASS |
| MAE improvement ≥ 1% | PASS |
| RMSE degradation ≤ 2% | PASS |
| R² difference ≥ -0.01 | PASS |

### Final Decision

**CANDIDATE ELIGIBLE FOR PROMOTION**

- MAE improvement: 1.52%.
- RMSE change: -1.20%.
- R² improvement: +0.0197.

The candidate outperformed the existing champion across all three evaluation metrics.

The quality gate marked the candidate as eligible for promotion. The existing champion was not overwritten.

## 📊 9. MLflow Experiment Tracking

MLflow is integrated to track candidate model training experiments.

It helps maintain:

- Experiment information.
- Training parameters.
- Model training runs.
- Run history.
- Logged metrics where configured.

To start the MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## 🌐 10. FastAPI Model Inference

The project includes a FastAPI-based inference service.

The API loads the saved champion model pipeline and provides an interface for generating delivery duration predictions from input features.

The complete preprocessing and prediction pipeline is reused during inference, ensuring consistency between model training and prediction.

**API implementation:** `api/main.py`

## 🐳 11. Docker Containerization

Docker is used to package the application and its dependencies into a container image.

The project includes a `Dockerfile` that defines the application environment and container build process.

### Benefits

- Consistent application environment.
- Dependency isolation.
- Portable application packaging.
- Simplified deployment preparation.
- Integration with CI/CD pipelines.

Build the Docker image from the project root:

```bash
docker build -t last-mile-delivery-mlops .
```

Run the container using the port configured by the application:

```bash
docker run -p 8000:8000 last-mile-delivery-mlops
```

## ☁️ 12. AWS ECR Integration

The project integrates **Amazon Elastic Container Registry (AWS ECR)** with GitHub Actions to automate Docker image publishing.

ECR provides a managed registry for storing and managing Docker images.

### Automated Workflow

1. Code changes are pushed to GitHub.
2. GitHub Actions triggers the CI/CD workflow.
3. The test job executes.
4. Docker builds the application image.
5. GitHub Actions authenticates with AWS.
6. The Docker image is tagged with the ECR repository URI.
7. The image is pushed to Amazon ECR.

### Verified CI/CD Execution

The GitHub Actions workflow includes two successful jobs:

| Job | Status |
|---|---|
| Test | SUCCESS |
| Build and Publish to Amazon ECR | SUCCESS |

The latest assessment-pipeline commit, `83dadd1`, also completed its workflow successfully.

**AWS ECR integration benefits:**

- Managed Docker image storage.
- Centralized container image management.
- Image tagging and version management.
- Integration with AWS container services.
- Foundation for subsequent cloud deployment.

**Important:** ECR stores container images; it does not itself execute the application. Running the container requires a compute service such as Amazon EC2 or Amazon ECS.

## 🔁 13. CI/CD with GitHub Actions

GitHub Actions is integrated to automate the project's CI/CD workflow.

The repository includes a GitHub Actions workflow that runs project tests and builds and publishes Docker images to AWS ECR.

### CI/CD Workflow

1. Developer pushes code to GitHub.
2. GitHub Actions detects the repository update.
3. The configured test job executes.
4. The Docker image is built.
5. AWS authentication is performed by the workflow.
6. The image is published to Amazon ECR.
7. Workflow execution status is displayed in GitHub Actions.

### Benefits

- Automated testing.
- Consistent Docker builds.
- Automated image publishing.
- Faster feedback on code changes.
- Reduced manual effort.
- Integration with AWS cloud infrastructure.

**View workflow history:**

[GitHub Actions — Last Mile Delivery MLOps](https://github.com/vidhigoyal28/last-mile-delivery-mlops/actions)

## 🏗️ 14. Project Architecture

### Machine Learning and MLOps Architecture

```text
                 ┌──────────────────────────┐
                 │      LaDe-D Dataset      │
                 │  Historical Delivery     │
                 │         Records          │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Data Preprocessing   │
                 │ Cleaning & Engineering  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   ML Training Pipeline   │
                 │  Scikit-learn + XGBoost  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     MLflow Tracking     │
                 │   Experiment Tracking   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  Champion Model Saved   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  Incoming Data Workflow │
                 │ Validation & Evaluation│
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ Candidate Model Training│
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Candidate Evaluation  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Quality Gate       │
                 │ Champion vs Candidate   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  Controlled Promotion   │
                 └──────────────────────────┘
```

### CI/CD and AWS Architecture

```text
             ┌──────────────────────────┐
             │     GitHub Repository    │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │     GitHub Actions       │
             │       CI/CD Pipeline     │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │     Automated Tests      │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │       Docker Build       │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │        AWS ECR           │
             │   Docker Image Registry  │
             └──────────────────────────┘
```

## 📁 15. Project Structure

```text
last-mile-delivery-mlops/
│
├── api/
│   └── main.py
│
├── artifacts/
│   └── quality_monitoring/
│       ├── incoming_validation.json
│       ├── champion_evaluation.json
│       ├── candidate_training.json
│       └── candidate_evaluation.json
│
├── data/
│   ├── raw/
│   │   └── lade-d/
│   ├── interim/
│   │   └── model_dataset.parquet
│   ├── processed/
│   │   ├── features_*.parquet
│   │   └── splits/
│   │       ├── train.parquet
│   │       ├── validation.parquet
│   │       └── test.parquet
│   └── incoming/
│       ├── incoming_batch.parquet
│       ├── incoming_train.parquet
│       └── incoming_holdout.parquet
│
├── models/
│   ├── baseline_model.joblib
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   ├── tuned_xgboost_model.joblib
│   └── candidates/
│       └── candidate_xgboost_model.joblib
│
├── reports/
│   └── quality_gate/
│       └── promotion_decision.json
│
├── src/
│   └── delivery_mlops/
│       └── pipelines/
│           ├── retrain.py
│           ├── training_pipeline.py
│           ├── validate_incoming_data.py
│           ├── prepare_incoming_split.py
│           ├── evaluate_champion.py
│           ├── train_candidate.py
│           ├── evaluate_candidate.py
│           └── quality_gate.py
│
├── create_incoming_batch.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Dataset files and serialized model artifacts are maintained locally and excluded from Git where configured. This structure describes the project workspace; not every listed artifact is committed to GitHub.*

## 🚀 16. Getting Started

### Prerequisites

- Python 3.x.
- Git.
- Docker.
- AWS account and configured AWS CLI for ECR operations.
- Access to the LaDe-D dataset.

### Step 1: Clone the Repository

```bash
git clone https://github.com/vidhigoyal28/last-mile-delivery-mlops.git

cd last-mile-delivery-mlops
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place the original LaDe-D dataset in:

```text
data/raw/lade-d/data/
```

Run the existing data preprocessing and feature-engineering workflow to generate the intermediate dataset and chronological splits.

The dataset and trained model artifacts are not included in the Git repository.

### Step 5: Run the MLOps Assessment Workflow

Run the following commands from the project root.

**1. Generate incoming batch**

```bash
python create_incoming_batch.py
```

**2. Validate incoming data**

```bash
python src/delivery_mlops/pipelines/validate_incoming_data.py
```

**3. Prepare incoming training and holdout splits**

```bash
python src/delivery_mlops/pipelines/prepare_incoming_split.py
```

**4. Evaluate champion model**

```bash
python src/delivery_mlops/pipelines/evaluate_champion.py
```

**5. Train candidate model**

```bash
python src/delivery_mlops/pipelines/train_candidate.py
```

**6. Evaluate candidate model**

```bash
python src/delivery_mlops/pipelines/evaluate_candidate.py
```

**7. Run quality gate**

```bash
python src/delivery_mlops/pipelines/quality_gate.py
```

The evaluation reports and promotion decision are saved as JSON files.

## 🔮 17. Future Enhancements

- Integrate real-time delivery data ingestion.
- Implement automated data drift and model performance monitoring.
- Introduce scheduled or event-triggered retraining.
- Add a model registry and controlled model promotion.
- Deploy the inference service to Amazon ECS or EC2.
- Add automated integration and API testing.
- Introduce DVC for dataset and model versioning.
- Evaluate additional algorithms and feature-engineering techniques.
- Establish an untouched chronological benchmark for final model evaluation.

## ⚠️ 18. Current Limitations

- The assessment's incoming batch is simulated using historical October records rather than newly collected production data.
- The October test period was also used in earlier experimentation.
- The candidate holdout was excluded from candidate training, but the broader historical data has prior experimental exposure.
- The quality gate marks candidates as eligible for promotion but does not automatically replace the champion.
- AWS ECR image publishing is implemented; running the application requires a separate compute service.
- Live production monitoring and real-time data ingestion require further integration.



GitHub: [@vidhigoyal28](https://github.com/vidhigoyal28)

**Repository:** [Last-Mile Delivery MLOps](https://github.com/vidhigoyal28/last-mile-delivery-mlops)

---

⭐ If you find this project useful, consider giving the repository a star!

**Project Focus:** Machine Learning | MLOps | XGBoost | MLflow | FastAPI | Docker | GitHub Actions | AWS ECR
