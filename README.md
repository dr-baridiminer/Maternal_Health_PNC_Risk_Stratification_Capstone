# Maternal_Health_PNC_Risk_Stratification_Capstone
DSF-FT16HYB- Group 2 Capstone project
 ## Dataset
  This project uses the Kenya DHS 2022 dataset: KENR8CFL (Births/Pregnancy/Postnatal Care Recode) and KEGE8AFL (GPS cluster file). Per DHS Program terms of use, this data cannot be redistributed. Each team member/user must independently register and request access: 
    1. Register at https://dhsprogram.com/data/new-user-registration.cfm 
    2. Create a new project request for Kenya, 2022 Standard DHS 
    3. Request the NR and GE datasets 
    4. Place downloaded files in data/raw/ (gitignored, not tracked in this repo) Raw data files are intentionally excluded from this repository in compliance with DHS data-sharing restrictions.

  ## Project Overview

Maternal and postnatal care remain important areas of public health in Kenya. Although many women deliver in health facilities, timely postnatal care (PNC) remains a challenge. This project uses machine learning to identify women who are at higher risk of missing a timely postnatal check within 48 hours of childbirth.

The project develops an explainable binary classification model using Kenya Demographic and Health Survey (KDHS) 2022 data. The model is designed as a decision-support tool that can help maternal and child health stakeholders identify higher-risk groups and prioritize follow-up interventions.

The project focuses on prediction, risk stratification, and explainability rather than replacing clinical or public-health decision making.


## Business Problem

Timely postnatal care provides an important opportunity to identify and respond to maternal and newborn health risks shortly after delivery. However, not all women receive a postnatal check within the recommended 48-hour period.

From a program-management perspective, limited resources make it important to identify groups that may be more likely to miss timely PNC. A data-driven risk stratification approach can support targeted follow-up by helping health programs understand which characteristics are associated with increased risk.

This project therefore addresses the following business question:

> Can an explainable machine learning model identify women who are at higher risk of missing a timely postnatal check within 48 hours of childbirth?


## Project Objectives

The main objective is to develop an explainable binary classification model for maternal PNC risk stratification in Kenya.

Specific objectives are to:

- Define a reliable target for missed timely PNC using the KDHS 2022 birth and postnatal care information.
- Identify demographic, socioeconomic, maternal, healthcare-utilization, and delivery-related factors associated with missed timely PNC.
- Develop and compare multiple machine learning classification models.
- Address class imbalance using appropriate modelling and evaluation strategies.
- Use cluster-aware validation to reduce the risk of overly optimistic performance estimates.
- Select a decision threshold that prioritizes recall for identifying women at risk.
- Explain model predictions using permutation importance and SHAP.
- Examine model performance across selected subgroups.
- Develop a Streamlit decision-support prototype for presenting the model output.


## Target Users and Stakeholders

The intended users and beneficiaries include:

- **County Ministry of Health maternal and child health coordinators** — to support identification of higher-risk groups and planning of targeted interventions.
- **Community Health Promoters (CHPs)** — to support prioritization of household follow-up and community-level maternal health activities.
- **Maternal and child health program implementers** — to support data-driven planning and resource allocation.
- **NGOs and development partners** — to provide additional evidence for maternal and postnatal care interventions.
- **Data scientists and researchers** — to demonstrate an explainable machine learning approach to public-health risk stratification.


## Data Understanding

The project uses Kenya DHS 2022 data from the Births/Pregnancy/Postnatal Care Recode and the GPS cluster file.

The initial KDHS dataset contained:

- **13,184 records**
- **919 variables**

The data was subsequently filtered and prepared for modelling based on the project's eligibility criteria.


## Sample Selection and Target Construction

The analysis focuses on eligible live birth and stillbirth records relevant to the postnatal-care outcome.

The data preparation process resulted in:

- 13,184 initial records
- 10,606 eligible live birth/stillbirth records
- 10,570 records after excluding observations with unknown target timing
- 10,505 records after deduplication

The final modelling dataset contains:

- **10,505 observations**
- **13 predictor variables**
- **1 target variable**

The target variable is:

`missed_timely_pnc`

where:

- `0` = Timely PNC
- `1` = Missed timely PNC

Timely PNC was defined as receiving a postnatal check within **48 hours** of childbirth.

For facility/pre-discharge pathways, the target was constructed using the relevant `m62` and `m63` variables. For post-discharge/home-delivery pathways, the corresponding `m66` and `m67` variables were used.

Observations for which the timing of the postnatal check could not be determined were excluded from modelling.


## Target Distribution

The final target distribution was:

| Outcome | Count | Percentage |
|---|---:|---:|
| Timely PNC | 7,814 | 74.4% |
| Missed timely PNC | 2,691 | 25.6% |
| **Total** | **10,505** | **100%** |

The target is therefore imbalanced, with missed timely PNC representing the minority class.

Because the project aims to identify women at risk of missing timely PNC, recall for the positive class was considered particularly important.


## Predictor Variables

The final model uses 13 approved predictors:

| DHS Variable | Description |
|---|---|
| `v012` | Maternal age |
| `v106` | Education level |
| `v501` | Marital status |
| `v714` | Employment status |
| `v190` | Household wealth level |
| `v201` | Number of children ever born |
| `m10` | Pregnancy intention |
| `m14` | Number of ANC visits |
| `m13` | Timing of first ANC visit |
| `m15` | Place of delivery |
| `m17` | Caesarean delivery |
| `v025` | Urban/rural residence |
| `v024` | Region |

GPS coordinates were used for geographical presentation and mapping but were **not used as model predictors**.


## Exploratory Data Analysis

Exploratory analysis was conducted to understand the distribution of the target and relationships between key predictors and missed timely PNC.

Important patterns observed included:

- **Home deliveries** showed the highest rate of missed timely PNC.
- **Rural residence** had a higher missed-PNC rate than urban residence.
- Women with **no education** had a higher missed-PNC rate, with the rate generally decreasing as education level increased.
- The **poorest household wealth quintile** showed the highest missed-PNC rate, while the richest quintile showed the lowest.
- Missing values were concentrated mainly in ANC-related variables, particularly timing of the first ANC visit.

The exploratory analysis helped identify socioeconomic, geographic, healthcare-utilization, and delivery-related patterns that informed the modelling stage.


## Data Preprocessing

The preprocessing workflow included:

- Selecting the approved modelling variables.
- Applying the project's eligibility criteria.
- Removing observations with unknown target timing.
- Removing duplicate records.
- Handling categorical and numerical variables through the modelling pipeline.
- Preparing the final modelling dataset.
- Preserving cluster information for grouped validation.
- Preparing a separate visualization dataset for analysis and presentation.

The final modelling table contained **10,505 rows and 14 columns**, consisting of 13 predictors and the target variable.


## Machine Learning Approach

Three classification approaches were evaluated:

1. **Tuned Logistic Regression**
2. **Random Forest**
3. **Tuned XGBoost**

The models were evaluated using **Average Precision (PR-AUC)** as the primary model-selection metric because the target class is imbalanced and the project prioritizes identification of women who may miss timely PNC.


## Validation Strategy

To reduce the risk of data leakage and overly optimistic performance estimates, the project used:

- Separate training, validation, and test datasets.
- **5-fold Stratified Group K-Fold cross-validation**.
- DHS sampling clusters as grouping variables.
- Model selection based on validation performance.
- Decision-threshold selection using the validation dataset.
- A final evaluation on an untouched test dataset.

The test dataset was not used to select the model or decision threshold.


## Model Comparison

The validation PR-AUC results were:

| Model | Validation PR-AUC |
|---|---:|
| Tuned XGBoost | **0.600** |
| Random Forest | 0.597 |
| Tuned Logistic Regression | 0.594 |

Tuned XGBoost achieved the highest validation PR-AUC and was therefore selected as the final model.


## XGBoost Hyperparameter Tuning

XGBoost was tuned using `RandomizedSearchCV`.

The tuning process evaluated:

- **20 candidate configurations**
- **5 cross-validation folds**
- **100 total model fits**
- Scoring metric: `average_precision`
- `random_state = 42`

The selected XGBoost configuration was:

n_estimators = 400
max_depth = 3
learning_rate = 0.03
min_child_weight = 10
subsample = 0.8
colsample_bytree = 1.0
reg_alpha = 0.0
reg_lambda = 1.0
gamma = 0.0

## Threshold Selection

The decision threshold was selected using the validation dataset, while keeping the test dataset completely untouched.

Because the main objective of the project is to identify as many women at risk of missing timely postnatal care as possible, recall was prioritized over precision. The F2-score was therefore used to select the optimal decision threshold because it gives greater importance to recall.

The selected decision threshold was:

- **Decision threshold:** 0.278
- **Validation Precision:** 0.332
- **Validation Recall:** 0.912
- **Validation F2-score:** 0.676

A predicted risk score of **0.278 or higher** was classified as **missed timely PNC (high risk)**.

The threshold was selected using validation data only to prevent information leakage from the test dataset.

## Final Results

The selected XGBoost model was evaluated on the untouched test dataset using the selected decision threshold of **0.278**.

| Metric | Test Score |
|---|---:|
| PR-AUC | 0.648 |
| ROC-AUC | 0.805 |
| Recall | 0.897 |
| Precision | 0.337 |
| F2-score | 0.674 |
| Balanced Accuracy | 0.647 |
| Accuracy | 0.524 |
| Brier Score | 0.166 |

The final model achieved a **PR-AUC of 0.648** and a **ROC-AUC of 0.805**.

Most importantly, the model achieved a **recall of 89.7%** for women classified as being at risk of missing timely PNC.

The relatively low precision of **33.7%** reflects the project's deliberate emphasis on identifying as many high-risk cases as possible. Therefore, the model is intended primarily as a **screening and prioritization tool**, rather than a standalone diagnostic system.

### Classification Report

| Outcome | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Timely PNC | 0.92 | 0.40 | 0.55 | 1,558 |
| Missed Timely PNC | 0.34 | 0.90 | 0.49 | 534 |
| **Overall Accuracy** | | | **0.52** | **2,092** |

The model demonstrates strong recall for the positive class, which aligns with the project's objective of identifying women who may require additional follow-up.

---

## Diagnostics

### Overfitting Assessment

Model performance was compared across the training, validation, and test datasets.

| Dataset | PR-AUC | ROC-AUC | Recall | Precision | F2-score |
|---|---:|---:|---:|---:|---:|
| Training | 0.695 | 0.841 | 0.932 | 0.359 | 0.706 |
| Validation | 0.600 | 0.780 | 0.912 | 0.332 | 0.676 |
| Test | 0.648 | 0.805 | 0.897 | 0.337 | 0.674 |

The difference between training and test PR-AUC was approximately **0.046**. The relatively similar performance across the validation and test datasets provides little evidence of severe overfitting.

Several safeguards were used to reduce overfitting and data leakage:

- Separate training, validation, and test datasets
- Cluster-aware data splitting
- 5-fold `StratifiedGroupKFold` cross-validation
- Validation-based model selection
- Validation-based threshold selection
- Controlled XGBoost model complexity
- One-time evaluation on the untouched test dataset

## XAI

Explainable AI (XAI) techniques were used to improve the interpretability of the final machine-learning model.

Two main approaches were used:

- **Permutation Importance**
- **SHAP (SHapley Additive Explanations)**

### Permutation Importance

Permutation importance was used to evaluate how much model performance changes when individual predictors are randomly shuffled.

This provides a global assessment of which variables contribute most to the model's predictive performance.

### SHAP Analysis

SHAP was used to provide both **global and individual-level explanations** of model predictions.

The analysis identified **place of delivery** as a dominant factor in the model's predictions. Geographic variables, including county-level information, also contributed to model predictions.

SHAP was also used to examine individual high-risk and low-risk predictions.

For example:

- A high-risk example received a predicted risk score of approximately **0.976** and had an actual missed-timely-PNC outcome.
- A low-risk example received a predicted risk score of approximately **0.052** and had an actual timely-PNC outcome.

These explanations help users understand which characteristics contributed to an individual model prediction.


## Calibration

Calibration analysis was conducted to assess whether the model's predicted risk scores correspond directly to observed probabilities.

The model showed evidence of **systematic probability overestimation**, which was associated in part with the use of class balancing during model development.

Therefore, the model's outputs should **not be interpreted as literal probabilities** that an individual woman will miss timely PNC.

Instead, the outputs should primarily be interpreted as **relative risk scores for screening and prioritization**.

The test Brier score was **0.166**.

Further calibration using techniques such as **Platt scaling or isotonic regression**, followed by external validation, would be required before interpreting the model outputs as reliable absolute probabilities.


## Subgroups

Subgroup analysis was conducted to assess whether model performance was consistent across different population groups.

The analysis considered:

- **Urban vs. rural residence**
- **Education level**

For each subgroup, the following were examined:

- Sample size
- Missed-timely-PNC prevalence
- Recall
- Precision

Subgroup analysis is important because overall model performance can hide differences in performance between population groups.

Results for small subgroups should be interpreted cautiously because smaller sample sizes may produce less stable performance estimates.


## Geospatial

Geospatial analysis was conducted using the DHS GPS cluster information to visualize geographic patterns within the study population.

The geographic analysis provides additional context for understanding potential regional differences in maternal and postnatal care outcomes and may support future maternal-health program planning.

**Important:** GPS coordinates were used for **mapping and visualization only** and were not included as predictors in the machine-learning model.

This distinction helps ensure that geographic coordinates are not directly used by the model to generate individual risk predictions.

## Streamlit

A **Streamlit decision-support prototype** was developed to demonstrate how the trained model could be used through an interactive interface.

The Streamlit application is located in:

```text
app/
