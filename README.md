# HCV Data Analysis and Ordinal Classification

## 1. Project Overview

This project analyzes the **HCV (Hepatitis C Virus) dataset** from the UCI Machine Learning Repository and develops machine-learning models to predict the patient's disease stage.

The main objective was not simply to maximize accuracy. Because the target variable represents an **ordered clinical progression**, the analysis treats the problem as **ordinal classification**:

**Healthy → Suspect Blood Donor → Hepatitis → Fibrosis → Cirrhosis**

This distinction is important because predicting *Fibrosis* for a patient with *Hepatitis* is a smaller clinical error than predicting *Cirrhosis* for a healthy patient. Therefore, the evaluation includes ordinal metrics in addition to ordinary classification accuracy.

The complete workflow was implemented in Python using a Jupyter/Anaconda environment.

---

## 2. Dataset Understanding

The dataset contains **615 observations** and **14 variables**.

The variables include:

| Variable | Type | Description |
|---|---|---|
| ID | Integer | Patient identifier |
| Category | Categorical | Target: disease stage |
| Age | Integer | Patient age |
| Sex | Binary | Male/Female |
| ALB | Float | Albumin |
| ALP | Float | Alkaline phosphatase |
| ALT | Float | Alanine aminotransferase |
| AST | Float | Aspartate aminotransferase |
| BIL | Float | Bilirubin |
| CHE | Float | Cholinesterase |
| CHOL | Float | Cholesterol |
| CREA | Float | Creatinine |
| GGT | Float | Gamma-glutamyl transferase |
| PROT | Float | Total protein |

The target distribution is strongly imbalanced:

| Target class | Number | Percentage |
|---|---:|---:|
| Healthy / Blood Donor | 533 | 86.67% |
| Suspect Blood Donor | 7 | 1.14% |
| Hepatitis | 24 | 3.90% |
| Fibrosis | 21 | 3.41% |
| Cirrhosis | 30 | 4.88% |

This imbalance is one of the most important characteristics of the dataset. A classifier could obtain a high overall accuracy simply by favoring the majority class while performing poorly on the clinically important minority classes.

---

# 3. Data Preparation

## 3.1 Removing the Patient Identifier

The `ID` column was removed because it is an identifier rather than a biological or clinical measurement.

Keeping it could introduce meaningless numerical patterns into the model. The goal is for predictions to depend on patient characteristics and laboratory measurements, not on the patient's position in the dataset.

The `Category` column was separated from the predictors because it is the target that the models are required to predict.

---

## 3.2 Encoding the Sex Variable

The categorical `Sex` variable was converted into a numerical representation so that it could be used by the machine-learning algorithms.

---

## 3.3 Treating the Target as an Ordinal Variable

The five target categories were represented in their natural order:

```text
0 = Healthy / Blood Donor
1 = Suspect Blood Donor
2 = Hepatitis
3 = Fibrosis
4 = Cirrhosis
```

This ordering is not arbitrary. It represents increasing disease severity.

Consequently, the problem is more appropriately viewed as **ordinal classification** rather than ordinary nominal multiclass classification.

---

# 4. Missing Values

Missing values were found in:

| Variable | Missing values |
|---|---:|
| ALB | 1 |
| ALP | 18 |
| ALT | 1 |
| CHOL | 10 |
| PROT | 1 |

The missing values were handled using `SimpleImputer`.

The main workflow used **mean imputation**, while a separate experiment compared mean and median imputation.

Importantly, imputation was performed inside the modelling pipeline rather than before cross-validation. This prevents information from the validation folds from influencing the imputation values and therefore helps prevent **data leakage**.

---

# 5. Outlier Handling

Clinical laboratory variables can contain extreme observations. These values cannot automatically be considered errors because unusually high or low measurements may represent genuine disease states.

For the main preprocessing workflow, an `OutlierCapper` transformation was used. It identifies limits using the **interquartile range (IQR)** and caps observations outside those limits.

The limits are calculated within the training data of each cross-validation fold. This is important because calculating the limits using the complete dataset would allow information from the validation data to influence preprocessing.

A separate experiment was also performed without outlier capping to investigate whether extreme laboratory measurements contain useful disease information.

---

# 6. Feature Scaling

The laboratory variables have very different numerical ranges. For example, some biomarkers have values in relatively small ranges while others can be much larger.

`StandardScaler` was therefore applied to standardize the numerical variables.

Conceptually, the transformation is:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original value,
- \(\mu\) is the training-set mean,
- \(\sigma\) is the training-set standard deviation.

Scaling is particularly important for distance-based methods such as KNN and for regression-based models.

Again, scaling is performed inside the cross-validation pipeline to avoid leakage.

---

# 7. Handling the Severe Class Imbalance

The target variable is highly imbalanced: 533 of the 615 observations belong to the healthy/blood-donor class, while the smallest class contains only 7 observations.

To prevent the models from simply learning the majority class, **SMOTE (Synthetic Minority Over-sampling Technique)** was incorporated into the modelling workflow.

SMOTE creates synthetic minority observations based on existing minority samples rather than simply duplicating the same observations.

Because the smallest class contains only seven observations, the choice of SMOTE parameters is important. Oversampling must be performed **only on the training portion of each cross-validation fold**.

This prevents synthetic observations generated from validation data from leaking into model training.

---

# 8. Cross-Validation Strategy

A single train/test split would be unreliable for this dataset because it contains only 615 observations and some classes are extremely small.

The workflow therefore uses **Stratified K-Fold Cross-Validation**.

Stratification is important because it attempts to preserve the class distribution across folds.

The complete preprocessing sequence is contained in a pipeline so that every fold independently performs:

1. Imputation
2. Outlier processing
3. Scaling
4. Class balancing
5. Model fitting

This is one of the most important methodological decisions in the project because preprocessing the entire dataset before cross-validation can produce overly optimistic results.

---

# 9. Baseline Models

Four ordinal models were evaluated:

- Ordinal Logistic Regression
- Ordinal Random Forest
- Ordinal Decision Tree
- Ordinal KNN

These models were selected to provide different modelling approaches:

- **Ordinal Logistic Regression** provides a relatively simple linear baseline.
- **Ordinal Random Forest** can model nonlinear relationships and interactions.
- **Ordinal Decision Tree** provides an interpretable nonlinear model.
- **Ordinal KNN** makes predictions based on similarity between patients.

Using several model families makes it possible to distinguish whether the observed performance depends on a particular modelling assumption.

---

# 10. Evaluation Metrics

Ordinary accuracy is not sufficient for this problem.

Three main metrics were therefore used.

## Accuracy

Accuracy measures the proportion of predictions that exactly match the true class.

Higher is better.

However, accuracy does not distinguish between a small ordinal mistake and a large one.

---

## Mean Absolute Ordinal Error (MAE)

The ordinal distance error measures how far the predicted class is from the true class.

For example:

```text
True = Hepatitis (2)
Predicted = Fibrosis (3)
```

has an ordinal error of:

```text
|2 - 3| = 1
```

whereas:

```text
True = Healthy (0)
Predicted = Cirrhosis (4)
```

has an error of:

```text
|0 - 4| = 4
```

Therefore, **lower MAE is better**.

This metric has a useful interpretation for an ordered disease-stage problem.

---

## Quadratic Weighted Kappa (QWK)

Quadratic Weighted Kappa evaluates agreement between predictions and the true ordinal labels while assigning greater penalties to predictions that are farther away from the correct class.

Higher values indicate better ordinal agreement.

This makes QWK particularly useful for this project because the five categories have a meaningful order.

---

# 11. Baseline Results

The baseline results were:

| Model | Accuracy | Ordinal MAE | QWK |
|---|---:|---:|---:|
| Ordinal Logistic Regression | 0.61 | 0.50 | 0.70 |
| Ordinal Random Forest | 0.86 | 0.20 | 0.82 |
| Ordinal Decision Tree | 0.91 | 0.18 | 0.80 |
| Ordinal KNN | 0.88 | 0.20 | 0.82 |

These results demonstrate why accuracy should not be considered in isolation.

The Ordinal Decision Tree achieved the highest accuracy (0.91), while Ordinal Random Forest and Ordinal KNN achieved stronger QWK values (0.82).

The Ordinal Logistic Regression model was clearly weaker, with an accuracy of 0.61 and MAE of 0.50.

The results suggest that the relationship between laboratory measurements and disease stage is not adequately represented by a simple linear ordinal model.

---

# 12. Data-Centric Improvement 1: Log Transformation

A log transformation was investigated to reduce skewness and reduce the influence of extremely large laboratory measurements.

The motivation was that several clinical measurements can have strongly right-skewed distributions.

The expectation was:

```text
Skewed distribution
        ↓
Log transformation
        ↓
Less skewness
        ↓
Potentially more stable modelling
```

However, the results showed that this did **not** universally improve performance.

In the recorded experiments, most models experienced a reduction in accuracy after the transformation, while KNN benefited relatively more.

This is an important result: a theoretically reasonable preprocessing technique does not automatically improve predictive performance.

One possible interpretation is that some extreme laboratory values contain genuine information about disease severity. Compressing those values may therefore remove useful signal.

For KNN, reducing skewness can nevertheless improve the geometry of the feature space and make similarity-based predictions more stable.

---

# 13. Data-Centric Improvement 2: Removing Outlier Capping

The opposite experiment was also performed: instead of restricting extreme observations, the original extreme values were retained.

This experiment was useful because it tests whether the extreme values represent noise or clinically meaningful information.

The results generally decreased compared with the strongest baseline configuration, with the effect varying between models.

This suggests that the extreme values may be useful in some contexts but can also distort the decision boundaries of models that are sensitive to feature distributions.

The experiment demonstrates why outliers in medical data should not automatically be deleted. They need to be investigated as possible biological signals.

---

# 14. Data-Centric Improvement 3: Feature Engineering

A new feature was created:

\[
AST/ALT\ Ratio = \frac{AST}{ALT}
\]

The motivation was biological rather than purely statistical: AST and ALT are both liver-related enzymes, and their relative magnitude can potentially provide information that is not captured by considering only one value.

However, the engineered feature did not improve the overall results compared with the baseline.

Several models performed slightly worse.

This is an important finding because feature engineering is not guaranteed to improve a model. In this dataset, AST and ALT individually appear to contain useful information, and replacing their individual information with a ratio may discard information about their absolute levels.

Therefore, the result supports keeping the original measurements rather than assuming that a biologically motivated ratio must necessarily be a better predictor.

---

# 15. Missing-Value Experiment: Mean vs Median

A separate experiment compared mean imputation with median imputation.

The results were:

| Model | Metric | Mean | Median |
|---|---|---:|---:|
| Ordinal Logistic Regression | Accuracy | 0.65 | 0.66 |
| Ordinal Logistic Regression | MAE | 0.43 | 0.42 |
| Ordinal Logistic Regression | QWK | 0.74 | 0.76 |
| Ordinal Random Forest | Accuracy | 0.86 | 0.86 |
| Ordinal Random Forest | MAE | 0.17 | 0.17 |
| Ordinal Random Forest | QWK | 0.89 | 0.89 |
| Ordinal Decision Tree | Accuracy | 0.90 | 0.90 |
| Ordinal Decision Tree | MAE | 0.18 | 0.17 |
| Ordinal Decision Tree | QWK | 0.83 | 0.83 |
| Ordinal KNN | Accuracy | 0.90 | 0.90 |
| Ordinal KNN | MAE | 0.15 | 0.15 |
| Ordinal KNN | QWK | 0.88 | 0.88 |

The differences are very small.

This is consistent with the relatively small difference between the mean and median values of the variables containing missing observations.

For example:

| Variable | Mean | Median | Missing |
|---|---:|---:|---:|
| ALP | 68.284 | 66.200 | 18 |
| CHOL | 5.368 | 5.300 | 10 |
| ALB | 41.620 | 41.950 | 1 |
| ALT | 28.451 | 23.000 | 1 |
| PROT | 72.044 | 72.200 | 1 |

Because the mean and median are generally close, changing the imputation strategy does not substantially change the data presented to the models.

Median imputation produced a small improvement for Ordinal Logistic Regression, which is plausible because the median is less sensitive to skewed distributions and extreme observations.

Overall, however, the experiment indicates that **the imputation strategy is not the dominant factor affecting performance in this dataset**.

---

# 16. Interpretation of the Results

The experiments lead to several important conclusions.

### 1. The problem should be treated as ordinal

The disease categories have a natural progression. Therefore, metrics such as MAE and QWK provide information that ordinary accuracy cannot provide.

### 2. Accuracy alone is misleading

Because the dataset is strongly imbalanced, a model can achieve high accuracy while performing poorly on minority disease stages.

The evaluation therefore considers both exact predictions and the distance between predicted and true disease stages.

### 3. Extreme values may contain clinical information

The experiments with log transformation and outlier handling showed that removing or compressing extreme measurements does not automatically improve prediction.

In medical data, an extreme value may be a true manifestation of disease rather than an error.

### 4. More complex preprocessing is not necessarily better

Neither log transformation nor the AST/ALT engineered feature consistently improved the models.

This illustrates an important data-analysis principle:

> A preprocessing method should be justified by the data and validated experimentally rather than assumed to improve the model.

### 5. Imputation method had limited influence

Mean and median imputation produced very similar results because the corresponding summary statistics were relatively close for most variables with missing observations.

---

# 17. Reproducibility and Leakage Prevention

A major design principle of the implementation is that preprocessing is performed inside the cross-validation workflow.

The model must not see information from the validation fold during:

- Imputation
- Outlier-bound calculation
- Scaling
- SMOTE

Otherwise, the validation score can become artificially optimistic.

The intended workflow is:

```text
Raw data
   │
   ├── Separate target
   │
   ▼
Stratified CV split
   │
   ├── Training fold ──► Imputation
   │                    ├── Outlier processing
   │                    ├── Scaling
   │                    ├── SMOTE
   │                    └── Model
   │
   └── Validation fold ─► Apply transformations learned from training fold
                           │
                           ▼
                       Prediction
                           │
                           ▼
                       Evaluation
```

This structure is essential for obtaining an honest estimate of generalization performance.

---

# 18. Overall Conclusion

The analysis demonstrates a complete data-analysis workflow rather than simply fitting several classifiers.

The most important decisions were driven by the characteristics of the HCV dataset:

- The target was treated as **ordinal** because disease stages have a meaningful order.
- The `ID` variable was removed because it has no predictive clinical meaning.
- Missing laboratory values were handled systematically.
- Numerical measurements were standardized.
- Extreme observations were investigated rather than automatically discarded.
- Severe target imbalance was explicitly addressed.
- Cross-validation was used to obtain more reliable estimates.
- Ordinal-specific metrics were included because the distance between classes matters.
- Multiple data manipulations were tested experimentally rather than assuming that every preprocessing technique improves the model.

The experiments also show that the best data-processing decision is not necessarily the most complicated one. Some transformations reduced performance, while median imputation produced only marginal differences compared with mean imputation.

The overall lesson is that **data preprocessing should be driven by the biological meaning of the variables, statistical properties of the dataset, and experimentally validated model performance.**
