# Credit Card Fraud Detection

A simple machine-learning project that explores credit-card transaction data and uses logistic regression to identify potentially fraudulent transactions.

## Project objective

Investigate how effectively a classification model can detect fraud when fraudulent transactions represent a small fraction of the data.

The analysis focuses on precision, recall, and the trade-off between detecting fraud and generating false alerts.

## Dataset

The dataset contains:
- Numerical features produced through principal component analysis (PCA).
- `Time`: elapsed time in seconds since the first recorded transaction.
- `Amount`: transaction amount.
- `Class`: the target label, where 0 indicates legitimate transactions and 1 indicates fraud.

The PCA-transformed features are anonymized and do not have clear business meanings.

Dataset source: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv]


## Tools
- Python
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook


## Approach
1. Loaded the data and checked for missing values and duplicate rows.
2. Explored class imbalance and transaction amounts.
3. Sorted transactions by time and used an approximately 80/20 chronological training/test split.
4. Trained logistic regression with standardized features and balanced class weights.
5. Evaluated the model using precision, recall, a confusion matrix, and a precision–recall curve.


## Results
The test set contained 56,962 transactions, including 75 fraudulent transactions.


| Metric | Result |
|---|---:|
| Fraud prevalence in the test set | 0.132% |
| Fraud cases detected | 67 of 75 |
| Fraud cases missed | 8 |
| Fraud precision | 7.1% |
| Fraud recall | 89.3% |
| Overall accuracy | 98.4% |


The model detected most fraudulent transactions, but generated many false alerts. Approximately 7 out of every 100 transactions flagged as fraud were actually fraudulent.


Accuracy alone does not describe the model's usefulness: a model that classified every transaction as legitimate would achieve approximately 99.87% accuracy on this test set while detecting no fraud.


## Key takeaway
The model achieved high fraud recall but low fraud precision. It demonstrates the trade-off between catching fraudulent transactions and incorrectly flagging legitimate activity.

These results do not justify automatically blocking transactions based on the model's predictions.


## Project files
- `fraud_detection.ipynb`: notebook containing the code, outputs, figures, and explanations.
- `fraud_detection.md`: Markdown export of the completed notebook.
- `fraud_detection_files/`: figures generated during Markdown export.
- `requirements.txt`: Python dependencies.


## How to run
1. Download the dataset from the source linked above.
2. Place `creditcard.csv` in the same folder as the notebook.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `fraud_detection.ipynb` and run all cells from top to bottom.

## Limitations
- PCA-transformed features limit business interpretation.
- Results apply to this dataset and evaluation period and may not generalize to future transactions.
- The test set contains only 75 fraud cases, so a small change in detections noticeably affects recall.
- Financial losses, review costs, and customer inconvenience are not quantified.
- This is a learning project, not a production fraud-detection system.


## Possible improvement
Explore whether changing the classification threshold can reduce false alerts while retaining an acceptable level of fraud detection.
Any threshold selection should use validation data rather than the final test set.