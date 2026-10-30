# Customer Satisfaction (CSAT) Prediction with a Deep Learning ANN

One month (August 2023) of customer support contacts of the e-commerce platform Shopzilla, used to predict the CSAT score (1 to 5) of a contact with an artificial neural network, flag the customers who are likely to be unhappy, and explain what drives satisfaction. The models run locally in a Streamlit web app.

Everything is in one notebook that follows the provided ML submission template:
**[Ecommerce_CSAT_Prediction_ANN.ipynb](Ecommerce_CSAT_Prediction_ANN.ipynb)**

## Repository layout

```
ecommerce-csat-ann/
├── Ecommerce_CSAT_Prediction_ANN.ipynb   the whole project, run top to bottom
├── csat_features.py                      data preparation and text cleaning (written by the notebook, used by the app)
├── app/
│   └── app.py                            Streamlit app for local deployment
├── data/
│   └── eCommerce_Customer_support_data.csv   85,907 support contacts (the provided file)
├── models/                               created by the notebook
│   ├── csat_ann_realtime.keras           ANN without remarks, for live alerts
│   ├── csat_ann_with_remarks.keras       ANN with the remark text branch
│   └── csat_metadata.json                input lists, alert thresholds, test scores, dropdown values for the app
├── outputs/
│   ├── figures/                          every chart of the notebook as a png
│   ├── risk_by_subcategory.csv           predicted and real low-CSAT share per sub-category
│   ├── risk_by_supervisor.csv            the same per supervisor
│   └── sample_new_contacts.csv           200 test contacts without the score, to try the app
├── requirements.txt
└── README.md
```

## What the notebook covers

| Objective | Where |
|---|---|
| Data cleaning: missing values, duplicates, broken timestamps | sections 1 to 3, 6.1, 6.2 |
| Feature engineering and selection | section 3 and 6.3 to 6.5 |
| Text preprocessing of the customer remarks | section 6.4 |
| ANN design and training | ML Models 2 and 3 |
| Evaluation and validation | every model, 5-fold cross-validation, error analysis, time-based check |
| Insights | 15 charts, 3 hypothesis tests, permutation importance, what-if analysis, section 7.1 |
| Local deployment | section 7.2 and the Streamlit app |

## The models

Two networks, because the customer's remark only exists after the survey:

- **Real-time ANN**: uses only what is known while the contact is handled (channel, issue, response time, agent and team, order and product). It can score a contact before the customer answers the survey.
- **ANN with remarks**: the same network plus a text branch for the remark, for analysis after the survey.

Both use an entity embedding for every categorical column (an agent becomes 24 learned numbers instead of 1,371 one-hot columns), a normalisation layer for the numbers, dense ReLU layers with batch normalisation and dropout, and a softmax over the five scores. They are trained with Adam on mini-batches, with early stopping and a learning rate that halves on plateaus, and tuned with a random search. The category lookups, text vocabulary and scaling are saved inside the model files.

## Results (test set, 12,887 contacts)

| Model | Accuracy | Macro F1 | QWK | Low-CSAT ROC-AUC | Low-CSAT PR-AUC |
|---|---|---|---|---|---|
| Always predict 5 | 0.694 | 0.164 | 0.000 | 0.500 | 0.146 |
| Logistic regression, real time | 0.696 | 0.202 | 0.134 | 0.720 | 0.317 |
| **ANN, real time (tuned)** | 0.697 | 0.200 | 0.124 | **0.725** | **0.331** |
| Logistic regression, with remarks | 0.726 | 0.261 | 0.385 | 0.813 | 0.537 |
| **ANN with remarks (tuned)** | 0.720 | 0.263 | **0.407** | 0.808 | 0.534 |

Low CSAT means a score of 1 or 2 (14.6% of the contacts). The PR-AUC of random guessing is 0.146. The ANN with remarks is stable in 5-fold cross-validation (ROC-AUC 0.810 +- 0.002).

The low-CSAT alert (threshold chosen on the validation rows):

| Model | Contacts flagged | Unhappy customers caught | Alerts that are right |
|---|---|---|---|
| Real time | 29.1% | 57.1% | 28.6% |
| With remarks | 13.1% | 46.7% | 52.0% |

Main findings:
- The response time matters most. 8.9% of the customers answered within 5 minutes give a low score, against 36.1% of those who wait more than a day.
- Process failures cause the most unhappy customers: technician visits, seller cancellations, installations and delayed orders have 20-34% low scores. Email, expensive orders and agents in training are riskier too.
- With the remark the model is much sharper, but the middle scores (2 to 4) can't be told apart from the 5s with this data. The model is a tool to find unhappy customers and rank risk, not to tell a 4 from a 5.

## Running the notebook

On **Google Colab**: open the notebook and run all cells. If the `data/` folder is not next to the notebook, it reads the csv from this GitHub repo. A full run takes about 11 minutes on a laptop CPU.

Locally (Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Ecommerce_CSAT_Prediction_ANN.ipynb
```

The results are reproducible: every model uses a fixed seed and TensorFlow runs in deterministic mode on the CPU.

## Running the app (local deployment)

After the notebook has created the `models/` folder (it is also included in the repo):

```bash
source .venv/bin/activate
streamlit run app/app.py
```

The app opens at http://localhost:8501 with three tabs:
1. **Predict one contact**: enter a contact and get the most likely score, the chance of a low score, the alert and a what-if for a 5-minute response.
2. **Score a file**: upload contacts in the Shopzilla export format (try `outputs/sample_new_contacts.csv`) and download the predictions.
3. **Insights**: the main charts and the risk tables.

## Data notes

- Dates are day/month/year. On 28 Aug 2023 the reply time was saved at 00:00 without the time of day, which gave 3,128 negative response times. These are treated as unknown.
- The handling time is missing in 99.7% of the rows and is not used. City, product category, price and order date exist for about 20% of the contacts, and the remark for 33.5%.
- Unique id and Order_id are random codes and are not used as features.
