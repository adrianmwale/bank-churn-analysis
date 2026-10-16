# Bank Customer Churn Analysis

Exploratory analysis and churn prediction on a 10,000-customer bank dataset, using Python, pandas and scikit-learn.

## Question

What is associated with customers leaving a bank, and how well can we predict who will leave?

## Dataset

[Churn Modelling on Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling): 10,000 customers with 14 columns, including credit score, country, gender, age, tenure, balance, number of products, activity status and salary. The target column is `Exited` (1 = customer left). There were no missing values and no duplicate rows. 20.4% of customers left, so the classes are imbalanced.

The data file is not included in this repo. Download it from the link above.

## Method

1. Checked data types, missing values, duplicates and the churn rate.
2. Dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`).
3. Compared churn by age, country and number of products.
4. One-hot encoded `Geography` and `Gender`, then split the data 80/20 (stratified on `Exited`).
5. Trained logistic regression (on scaled features) and a random forest (200 trees), and evaluated both on the held-out 20%.

## Findings

**Age.** Customers who left were older. The median age of leavers was 45 compared with 36 for customers who stayed.

**Country.** Customers in Germany churned at 32.4%, about double the rate in France (16.2%) and Spain (16.7%).

**Number of products.** Churn by number of products was not a straight line. Customers with 2 products churned least (7.6% of 4,590 customers), and customers with 1 product churned at 27.7% (5,084 customers). Churn was far higher with 3 products (82.7% of 266 customers) and 4 products (100% of 60 customers). The 3- and 4-product groups are small, so I treat that result with caution.

**Model results (test set of 2,000 customers, 407 of whom left):**

| Model | Accuracy | Precision (leavers) | Recall (leavers) | F1 (leavers) |
|---|---|---|---|---|
| Logistic regression | 81% | 0.59 | 0.19 | 0.28 |
| Random forest | 86% | 0.78 | 0.46 | 0.58 |

Accuracy is misleading here, because about 80% of customers stay and a model that always predicted "stays" would score about 80%. Logistic regression caught only 76 of the 407 customers who left. The random forest performed better, catching 46% of leavers and being correct 78% of the time when it flagged someone. In the random forest, age was the most important feature.

## Limitations

- Even the better model misses more than half of the customers who leave.
- These are patterns in a public dataset, not proof of what causes customers to leave.
- I used default model settings and did not tune them or try class weighting.

## Next steps

- Try class weighting or a different decision threshold to improve recall on leavers.
- Tune the random forest and compare it with gradient boosting.

## How to run

Open `bank_churn_analysis.ipynb` in Google Colab, upload `Churn_Modelling.csv`, and run all cells.
