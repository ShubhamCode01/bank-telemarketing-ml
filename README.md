# Bank Telemarketing Success Prediction

## Project Overview

This project implements a machine-learning approach to predict the success of bank telemarketing campaigns. It is based on the research paper *A Data-Driven Approach to Predict the Success of Bank Telemarketing* by Sérgio Moro, Paulo Cortez, and Paulo Rita (2014).

The project evaluates multiple machine-learning models and studies their ability to identify customers who are likely to subscribe to a term deposit.

## Objectives

- Predict whether a customer will subscribe to a term deposit.
- Compare the performance of different machine-learning algorithms.
- Evaluate models using ROC-AUC and cumulative lift (ALIFT).
- Implement rolling-window evaluation to study performance across successive data windows.
- Analyze important features influencing predictions.

## Dataset

The project uses bank telemarketing campaign data.

The dataset contains customer information, campaign details, previous contact history, and the target variable `y`.

- **Target:** `y`
- `yes`: Customer subscribed to a term deposit.
- `no`: Customer did not subscribe.

The `duration` feature is excluded from the main prediction model because call duration is not known before a call begins.

**Dataset setup:** Download the dataset separately and place the required CSV file in the project directory. Do not upload large datasets or sensitive customer information to GitHub.

> Note: The code currently expects `bank-additional-full.csv` with a semicolon separator. Ensure the dataset matches the feature columns used by the script.

## Methodology

### 1. Data Preparation
- Load and inspect the dataset.
- Exclude the call-duration feature to avoid information leakage.
- Encode categorical variables and scale numerical variables using a preprocessing pipeline.

### 2. Feature Selection
Features are evaluated in groups representing client characteristics, call context, contact history, and socioeconomic indicators.

### 3. Model Training
The implementation compares the following models:

- Logistic Regression (LR)
- Decision Tree (DT)
- Neural Network (NN)
- Support Vector Machine (SVM)

### 4. Rolling-Window Evaluation

Rolling-window evaluation trains each model on a window of historical observations and tests it on the next block of observations.

The current configuration uses:
- Training window size (`W`): 20,000 records
- Test block size (`K`): 500 records
- Test period: newest 20% of the dataset

The window advances through the test period, allowing model performance to be examined across successive blocks.

### 5. Evaluation Metrics

- **ROC-AUC:** Measures how well a model ranks subscribers above non-subscribers.
- **ALIFT:** Evaluates how effectively the model ranks likely subscribers near the top of a contact list.
- **ROC curve:** Shows the relationship between true-positive and false-positive rates.
- **Cumulative lift curve:** Shows the proportion of subscribers captured as more customers are contacted.

## Visualizations

The script generates visualizations for:
- ROC curves
- Cumulative lift
- ROC-AUC across rolling windows
- Feature importance
- Variable effect curves

Generated figures are saved as image files when the corresponding code runs successfully.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ShubhamCode01/bank-telemarketing-ml.git
   ```

2. Move into the project directory:

   ```bash
   cd bank-telemarketing-ml
   ```

3. Place the required dataset CSV file in the project directory.

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Python script:

   ```bash
   python bank_telemarketing_replication_with_rolling_graph.py
   ```

6. Review the generated graphs, metric tables, and CSV outputs.

## Research Reference

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31.

## Disclaimer

This project is an academic implementation inspired by the referenced research paper. Differences in dataset version, features, model settings, and evaluation procedures may affect direct comparability with the original published results.
