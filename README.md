# FinanceCare AI

A chatbot that helps bank staff spot customers who may be heading into financial trouble, so they can reach out early and offer support.

## What it does

- Checks a customer's account and flags warning signs, such as high credit card use, late payments, a low balance or an overdrawn account
- Gives each customer a status: **No Flag**, **Early Warning** or **Financial Distress**
- Explains why a customer was flagged and suggests support options
- Lets staff record a review and close a flag, with supervisor approval for the most serious cases
- Keeps a log of every question asked and every review made

A flag is a reason for a person to take a closer look, not a final decision about the customer.

## How it works

1. The bot reads the bank's customer data.
2. It scores each customer using clear rules, such as "credit card more than 75% used" or "payment made late".
3. More points means a higher status.
4. An optional trained model gives a second opinion, but it never changes the rule-based result.
5. Staff chat with the bot in a desktop window and ask questions in plain English, like "Find customers at risk" or "Check customer 1000".

Age, gender, city and other personal details are never used to score a customer.

## Constraints

- **Small dataset:** the data holds 5,000 customers, so results are a demonstration and not proof that the bot works on a real bank's customer base

## Data

The Comprehensive Banking Database is included in this repository as `Comprehensive_Banking_Database.csv` so the project runs straight after cloning. It comes from Kaggle, and the original author keeps the rights to it:

[Kaggle dataset link](PASTE-YOUR-KAGGLE-LINK-HERE)

## Run it

```
pip install -r requirements.txt
python train_model.py
python financecare_ai.py
```

`train_model.py` is optional. It turns on the model's second opinion.

## Files

| File | Purpose |
|---|---|
| `financecare_ai.py` | The chatbot and desktop window |
| `train_model.py` | Trains the optional model |
| `test_distress.py` | Tests |
| `requirements.txt` | Python packages |
