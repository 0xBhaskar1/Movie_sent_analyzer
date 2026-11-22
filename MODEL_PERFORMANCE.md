# IMDb Movie Sentiment Analysis — Model Performance

## Dataset and evaluation setup

- **Dataset:** IMDb movie reviews
- **Task:** Binary sentiment classification
- **Classes:** `0 = Negative`, `1 = Positive`
- **Training samples:** 25,000
- **Testing samples:** 25,000
- **Features:** TF-IDF (up to 30,000 features, unigrams and bigrams)
- **Evaluation:** All metrics below are calculated on the held-out IMDb test split.

## Performance summary

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 86.95% | 87.89% | 85.71% | 86.79% |
| Logistic Regression | **90.09%** | 89.70% | **90.58%** | **90.14%** |
| Linear SVM | 89.44% | **89.56%** | 89.30% | 89.43% |

**Best overall model by accuracy and F1-score: Logistic Regression.**

## Confusion matrices

Rows represent the actual class; columns represent the predicted class.

Column order: **Negative (0), Positive (1)**  
Row order: **Negative (0), Positive (1)**

### 1. Multinomial Naive Bayes

```text
                 Predicted
                 Negative  Positive
Actual Negative    11024      1476
Actual Positive     1786     10714
```

Matrix:

```text
[[11024  1476]
 [ 1786 10714]]
```

- True Negative (TN): 11,024
- False Positive (FP): 1,476
- False Negative (FN): 1,786
- True Positive (TP): 10,714

### 2. Logistic Regression

```text
                 Predicted
                 Negative  Positive
Actual Negative    11200      1300
Actual Positive     1177     11323
```

Matrix:

```text
[[11200  1300]
 [ 1177 11323]]
```

- True Negative (TN): 11,200
- False Positive (FP): 1,300
- False Negative (FN): 1,177
- True Positive (TP): 11,323

### 3. Linear SVM

```text
                 Predicted
                 Negative  Positive
Actual Negative    11199      1301
Actual Positive     1338     11162
```

Matrix:

```text
[[11199  1301]
 [ 1338 11162]]
```

- True Negative (TN): 11,199
- False Positive (FP): 1,301
- False Negative (FN): 1,338
- True Positive (TP): 11,162

## Interpretation

- **True Negative (TN):** A negative review correctly classified as negative.
- **False Positive (FP):** A negative review incorrectly classified as positive.
- **False Negative (FN):** A positive review incorrectly classified as negative.
- **True Positive (TP):** A positive review correctly classified as positive.

### Comparison

- **Naive Bayes** is the weakest of the three on this test split, with 86.95% accuracy.
- **Logistic Regression** has the highest accuracy (90.09%), recall (90.58%), and F1-score (90.14%). It also has the fewest total errors: 2,477.
- **Linear SVM** is close behind, with 89.44% accuracy and 2,639 total errors.
- Logistic Regression is the strongest overall choice among these three models for this experiment.

## Notes

1. Results are specific to this dataset split, preprocessing pipeline, TF-IDF settings, and model hyperparameters. They are not a guarantee of performance on every review or domain.
2. Precision and recall above treat **Positive (class 1)** as the positive class, as in scikit-learn's default binary metric behavior.
3. The confusion matrix totals 25,000 test reviews for each model.
4. Logistic Regression and Naive Bayes provide `predict_proba()` in this implementation. `LinearSVC` provides a decision score rather than calibrated class probabilities by default.

---
Generated from the training output for the IMDb Movie Sentiment Analysis project.
