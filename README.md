# Predicting Financial Resilience in Pakistan: A Machine Learning & SHAP Approach

This repository contains the complete Data Mining and Machine Learning pipeline for predicting household financial resilience in Pakistan. The project utilizes ensemble learning (Random Forest) and Game Theory-based explainability (SHAP) to uncover the socio-economic drivers of financial survival during economic shocks.

## Dataset Reference & Source
The data used in this project is the **Pakistan Global Findex Data 2021**. 
* **Source:** [Kaggle - Pakistan Global Findex Data 2021 by Faiza Zain](https://www.kaggle.com/datasets/faizazain/pakistan-global-findexdata2021)
* **Original Publisher:** World Bank Global Findex Database (Demirgüç-Kunt et al., 2022).
* **File used in code:** `micro_pak.csv`

## Project Pipeline 
This project strictly follows a 7-phase data mining methodology:

1. **Problem Definition:** Predicting if a household can access emergency funds (Resilience) based on demographics and digital financial inclusion.
2. **Data Collection:** Importing the Kaggle Findex 2021 dataset (`micro_pak.csv`).
3. **Data Preprocessing:** Cleaning survey error codes (`8` and `9`), mapping survey responses to binary logic (`1`/`0`), handling missing values via median imputation, and engineering custom features.
4. **Data Exploration (EDA):** Visualizing the gender gap, income distributions, and the impact of financial inclusion on resilience.
5. **Modeling:** Training a baseline **Logistic Regression** model and an advanced **Random Forest Classifier**.
6. **Evaluation:** Assessing models using Accuracy metrics and a Confusion Matrix to evaluate True/False Positives.
7. **Interpretation:** Using **SHAP (Shapley Additive Explanations)** to open the "Black Box" of the Random Forest model and mathematically rank feature importance.

##  Key Engineered Features
Instead of relying solely on raw data, this project engineered new knowledge:
* **`Target_Resilient`:** A binary classification variable derived from survey question `fin24`. (1 = Can access emergency funds, 0 = Cannot).
* **`DFI_Score` (Digital Financial Inclusion):** A custom index (0 to 3) calculated by combining whether an individual has a bank account, owns a debit card, and utilizes digital payments.

## Key Results
* The **Random Forest Classifier** achieved an accuracy of ** 66.2%**, outperforming the linear baseline.
* The **SHAP Analysis** proved that Income, Education, and the engineered `DFI_Score` are the strongest predictive drivers of financial resilience in Pakistan.

All results in this project are 100% reproducible. The models utilize `random_state=42` to ensure consistent training splits and decision tree generation.

**Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
