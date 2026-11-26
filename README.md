# Employee Attrition Prediction and Risk Scoring - Palo Alto Networks

Predicts which employees are likely to leave, scores each with an attrition probability (Low <30%, Medium 30-60%, High >60%) and explains the main drivers.

## Run
```
pip install -r requirements.txt
python train.py Palo_Alto_Networks.csv     # optional: rebuilds models, scores and charts
python -m streamlit run app.py
```
## Method
- 1,470 employees, 31 fields (dataset rebuilt from the supplied PDF table). Employee ID = row number.
- Features added: income-to-experience ratio, promotion delay ratio and flag, engagement composite, workload stress flag, companies per year.
- One-hot encoding, scaled numerics, class weights for imbalance, stratified 80/20 split, 5-fold CV.
- Models: Logistic Regression, Random Forest, Gradient Boosting. Metrics: accuracy, precision, recall, F1, ROC-AUC.
- Probabilities are calibrated; dashboard scores are out-of-fold. Reason codes show how much each factor raises or lowers an employee's probability versus a typical employee.
