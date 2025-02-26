# Census Income Prediction (Adult Dataset)

This project aims to predict whether an individual's annual income exceeds $50K based on demographic and occupational data from the [UCI Adult (Census Income) dataset](https://archive.ics.uci.edu/dataset/2/adult).

We compare multiple Machine Learning algorithms:
- **AdaBoost**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**

---

## Table of Contents

1. [Dataset Description](#dataset-description)  
2. [Project Structure](#project-structure)  
3. [Installation & Setup](#installation--setup)  
4. [Usage](#usage)  
5. [Implementation Details](#implementation-details)  
6. [Results](#results)  
7. [Key Insights](#key-insights)  
8. [Future Work](#future-work)  
9. [Authors](#authors)  
10. [License](#license)  

---

## 1. Dataset Description

- **Name**: Adult (Census Income) Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)  
- **Size**: ~32,000 rows  
- **Features**:
  - Demographic data (e.g., age, gender, marital status, race)
  - Occupational data (e.g., workclass, occupation, hours-per-week)
  - **Target**: `income` — binary classification:
    - `0` (≤ 50K)
    - `1` (> 50K)
- **Missing Values**: Some categorical features contain "?" which we drop or encode accordingly.

---

## 2. Project Structure

```
.
├── src
│   ├── adaboost.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── results
│   └── model_comparison.csv (created after evaluation)
├── README.md (this file)
├── requirements.txt (optional: Python dependencies)
└── .gitignore
```

- **`preprocess.py`**: Fetches and preprocesses the dataset (handling missing values, encoding, train/test split).  
- **`adaboost.py`, `svm.py`, `decision_tree.py`, `knn.py`**: Each file contains a `train_<model>` function to train the respective model.  
- **`train.py`**: Main entry point for training all models at once.  
- **`evaluate.py`**: Evaluates all models on the test set and outputs results (and saves a CSV).  

---

## 3. Installation & Setup

### Clone this repository:
```bash
git clone https://github.com/username/IncomePredictionML.git
cd IncomePredictionML
```

### Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
# or on Windows:
.venv\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, ensure you have at least:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib` (optional)
- `matplotlib` / `seaborn` (optional, for visualization)

---

## 4. Usage

### Train all models:
```bash
cd src
python train.py
```
This will:
- Load and preprocess the data
- Train each model (AdaBoost, SVM, Decision Tree, KNN)
- Print accuracy scores
- Finish with "Training Complete."

### Evaluate models:
```bash
python evaluate.py
```
This will:
- Load and preprocess the data
- Train each model on the training set
- Compute test accuracies
- Create a `results/model_comparison.csv` with the final accuracies
- Print a dataframe summary of results

---

## 5. Implementation Details

### Preprocessing:
- Dropped rows with missing values (`dropna`).
- Encoded categorical features with `LabelEncoder`.
- Converted the income column (`>50K`, `<=50K`) into binary labels (`1`, `0`).
- Used an 80/20 train-test split with a fixed `random_state=42`.

### Training:
- **AdaBoost** with `n_estimators=50`, `random_state=42`.
- **SVM** with `kernel='rbf'`, `random_state=42`.
- **Decision Tree** with `random_state=42`.
- **KNN** with `n_neighbors=5` as default.

### Metrics:
- Accuracy is computed using `accuracy_score` from `scikit-learn`.
- For each model, we also store `.score()` on the test set.

---

## 6. Results

Below is a sample result (your numbers may vary slightly):

| Model           | Accuracy |
|---------------|----------|
| AdaBoost      | 0.85     |
| SVM           | 0.80     |
| Decision Tree | 0.81     |
| KNN           | 0.78     |

Generally, **AdaBoost** performs the best, closely followed by **Decision Trees** and **SVM**. **KNN** tends to lag behind.

---

## 7. Key Insights

- **Feature Importance**: Age, education level, marital status, capital-gain/capital-loss, and hours-per-week play a crucial role in predicting income.
- **Ensemble Methods**: AdaBoost outperforms a single Decision Tree by combining many "weak learners" into a stronger ensemble.
- **Data Quality**: Handling missing values and encoding categorical columns is crucial for better performance.
- **Potential for Regression**: Although we focused on classification, the same dataset could be used to predict numeric values (e.g., hours-per-week).

---

## 8. Future Work

- **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize model parameters.
- **Feature Engineering**: Create interaction features (e.g., age vs. education) or bin numeric features (e.g., age groups).
- **Advanced Ensemble Methods**: Try **Random Forest**, **XGBoost**, or **LightGBM** for potentially higher accuracy.
- **Regression**: Predict `hours-per-week` as a continuous variable with linear regression or other regressors.

---

## 9. Authors

- **Maor Uriel Berenstein** – ****  
- **Liel Yoash** – ****  

---

## 10. License

This project is licensed under the MIT License.
