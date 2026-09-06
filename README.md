# Customer Segmentation & Churn Pattern Analytics

This starter project turns the supplied European banking workbook into an interactive Streamlit dashboard. It covers the requested descriptive analytics first, then compares an explainable Logistic Regression baseline with a Random Forest churn model.

## 1. What you are building

The project answers four business questions:

1. What percentage of customers churned?
2. Which customer segments have the highest churn rate and contribute the most churners?
3. How do geography, age, tenure, engagement, and financial profile relate to churn?
4. How much customer balance is associated with churn among high-value customers?

The required project is mainly **descriptive analytics**, not AI. Rule-based segmentation and KPI calculation should be completed before any machine-learning model. The optional ML section predicts `Exited` and helps rank customers by risk.

## 2. Important facts about the supplied workbook

The workbook contains one sheet with 10,000 customer rows and 14 source columns. It has:

- 2,037 churned customers and an overall churn rate of 20.37%.
- No missing cells in the source fields.
- No duplicate rows or duplicate `CustomerId` values.
- Valid 0/1 values in `HasCrCard`, `IsActiveMember`, and `Exited`.
- A `Year` column containing only 2025. A constant field cannot help an ML model, so it is excluded.

The workbook does not contain source documentation proving that it is European Central Bank data. Present it as an educational European banking churn dataset unless you receive authoritative provenance.

## 3. Segment definitions used in this starter

The project brief names several bands but does not define every boundary. This starter uses explicit, non-overlapping assumptions:

| Dimension | Definition |
|---|---|
| Age | `<30`, `30–45`, `46–59`, `60+` |
| Credit score | Low `<580`, Medium `580–669`, High `670+` |
| Tenure | New `0–2`, Mid-term `3–6`, Long-term `7–10` |
| Balance | Zero `=0`; Low `>0` through the median positive balance; High above that median |
| High value | Balance at or above the portfolio's 75th percentile |

Why change the brief's `46–60` and `60+` labels? They overlap at age 60. The definitions above put each customer in exactly one group.

Credit-score bands are project assumptions, not universal European banking standards. Put them in the methodology section of your report and allow stakeholders to change them if they have business-approved definitions.

## 4. KPI formulas

Use counts in the numerator and denominator. Do not average already-calculated percentages.

- **Overall churn rate** = churned customers / all customers.
- **Segment churn rate** = churned customers in segment / all customers in segment.
- **Churn contribution** = churned customers in segment / all churned customers.
- **High-value churn ratio** = churned high-value customers / all high-value customers.
- **Geographic risk index** = geography churn rate / overall churn rate. A value of 1.25 means 25% above the portfolio average.
- **Engagement risk ratio** = inactive churn rate / active churn rate.
- **Balance at risk** = sum of balances held by churned customers in the chosen high-value group.

`Balance at risk` is not revenue loss. Estimating revenue risk requires revenue or margin, fees, cost-to-serve, and ideally customer-lifetime-value data.

## 5. Recommended implementation order

### Phase A — Set up and validate

1. Create a Python virtual environment.
2. Install the packages in `requirements.txt`.
3. Load the workbook with `pandas.read_excel`.
4. Check required columns, missing values, duplicates, category values, numeric ranges, and 0/1 fields.
5. Record any data-quality decisions instead of silently deleting rows.

### Phase B — Prepare segments

1. Do not use `Surname` for analysis or ML.
2. Keep `CustomerId` only as an identifier.
3. Exclude constant `Year` from ML.
4. Create `AgeGroup`, `CreditBand`, `TenureGroup`, `BalanceSegment`, and `HighValue`.
5. Check that segment counts reconcile to the total number of customers.

### Phase C — Explore and calculate KPIs

Start with these tables:

- Overall customers, churners, retained customers, and churn rate.
- Geography: customer count, churners, churn rate, contribution, risk index.
- Age, gender, tenure, products, activity, credit band, and balance segment.
- Geography × age churn-rate matrix.
- Average financial profile for churned versus retained customers.
- High-value churn by geography and engagement.

Every chart should display both the **rate** and the **sample size**. A 100% churn rate based on 60 customers is not equivalent to a 30% rate based on thousands.

### Phase D — Build the Streamlit dashboard

The supplied `app.py` contains:

- Excel file upload and validation.
- Geography, gender, and age filters.
- Overall KPI cards.
- Segment selector and downloadable summary.
- Geography × age heatmap.
- Age and tenure comparison.
- High-value explorer with an adjustable percentile.
- Logistic Regression versus Random Forest comparison and decision-threshold slider.

### Phase E — Add ML only after the dashboard works

`Exited` is a binary target, so start with Logistic Regression as an explainable baseline. Use Random Forest as the main model because it can learn nonlinear relationships and interactions.

The model workflow is:

1. Remove identifiers and non-predictive fields: `CustomerId`, `Surname`, and constant `Year`.
2. Split data into 80% train and 20% test sets with `stratify=y`.
3. Impute numeric values with the median and standardize them.
4. Impute categorical values and one-hot encode them.
5. Put preprocessing and the classifier in one scikit-learn `Pipeline` to avoid train/test leakage.
6. Use `class_weight="balanced"` because churners are the minority class.
7. Train Logistic Regression and Random Forest with the same split.
8. Measure ROC-AUC, PR-AUC, recall, precision, F1, and the confusion matrix.
9. Select the Random Forest probability threshold based on campaign capacity and the relative cost of missed churners versus unnecessary outreach.

Do not use accuracy alone. Predicting every customer as retained would already be 79.63% accurate in this dataset and would identify no churners.

## 6. Run the project

### macOS or Linux

```bash
cd european_bank_churn_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell

```powershell
cd european_bank_churn_starter
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser will open the app. Upload `European_Bank (5).xlsx` when prompted.

To test the analysis functions without the web app:

```bash
python smoke_test.py "/full/path/to/European_Bank (5).xlsx"
```

## 7. How to interpret the supplied data

Using the documented bands, the main descriptive findings are:

- Germany: 32.44% churn, compared with 16.15% in France and 16.67% in Spain. Germany's risk index is about 1.59.
- Inactive customers: 26.85% churn versus 14.27% for active customers, a risk ratio of about 1.88.
- Ages 46–59: 51.10% churn, the highest age band.
- Germany × ages 46–59: 67.28% churn, the highest large geography-age intersection.
- Female customers: 25.07% churn versus 16.46% for male customers. Treat this as an association and review fairness before using it operationally.
- Three products: 82.71% churn across 266 customers. Four products: 100% churn across only 60 customers. These extreme small groups deserve a data-quality and business-definition check.
- High value at the 75th balance percentile (`127,644.24`): 2,500 customers, 592 churners, and a 23.68% churn rate.
- Churned balance in that high-value group: about 88.65 million, or 47.77% of the balance held by all churned customers.

These are patterns, not causes. For example, the data cannot prove that location or age caused churn.

## 8. Suggested report structure

1. **Executive summary** — three to five findings and retention actions.
2. **Business problem** — why churn and segment differences matter.
3. **Dataset and limitations** — rows, columns, period, target, missing data, and unknown provenance.
4. **Methodology** — validation, cleaning, segment boundaries, formulas, and filters.
5. **Descriptive results** — overall, geography, demographics, engagement, products, and financial profile.
6. **High-value analysis** — threshold, churn ratio, balance at risk, and limitations.
7. **ML comparison** — split, preprocessing, Logistic Regression baseline, Random Forest, metrics, threshold, and fairness checks.
8. **Recommendations** — targeted outreach experiments, measurement plan, and monitoring.
9. **Limitations** — one-year snapshot, no event date, no interaction history, no reason for exit, no revenue field, and no causal inference.

## 9. Practical retention recommendations

- Prioritize diagnostic research for Germany and ages 46–59 before launching a broad campaign.
- Create a re-engagement experiment for inactive customers and compare treatment versus control groups.
- Review why customers with three or four products show extreme churn; confirm that product counts and churn labels are correctly defined.
- Use high-value churn flags to prioritize human retention review, not to deny services or make adverse decisions.
- Track campaign lift, retained customers, incremental margin, contact cost, and complaints—not only model accuracy.

## 10. Learning sources

### Official documentation and courses

- pandas Excel import: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
- pandas grouping and aggregation: https://pandas.pydata.org/docs/reference/groupby.html
- pandas `cut` for bands: https://pandas.pydata.org/docs/reference/api/pandas.cut.html
- Streamlit beginner guide: https://docs.streamlit.io/get-started
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit deployment: https://docs.streamlit.io/deploy
- Free scikit-learn MOOC: https://inria.github.io/scikit-learn-mooc/
- Mixed numeric/categorical preprocessing: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- Classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

### Beginner-friendly videos

- Build 12 Data Science Apps with Python and Streamlit: https://www.youtube.com/watch?v=JwSS70SZdyM
- Logistic Regression, Clearly Explained (StatQuest): https://www.youtube.com/watch?v=yIYKR4sgzI8
- ROC and AUC, Clearly Explained (StatQuest): https://www.youtube.com/watch?v=4jRBRDbJemM
- K-means clustering, Clearly Explained (optional extension): https://www.youtube.com/watch?v=4b5d3muPQmA

K-means is optional. The project's named segments are rule-based business segments; do not replace them with clustering unless your mentor specifically asks for an unsupervised-learning extension.

## 11. Files

- `app.py`: Streamlit interface and charts.
- `churn_analysis.py`: validation, preparation, KPI, segmentation, and ML functions.
- `requirements.txt`: Python dependencies.
- `smoke_test.py`: quick command-line validation.

## 12. Share the project safely

The `.gitignore` file keeps the virtual environment, PyCharm settings, caches, secrets, and Excel/CSV data out of GitHub. This is important because real customer-level banking data should not be published.

Create an empty GitHub repository, then run these commands in the PyCharm terminal:

```bash
git init -b main
git add .
git commit -m "Build European bank churn analytics dashboard"
git remote add origin https://github.com/YOUR-USERNAME/european-bank-churn-analytics.git
git push -u origin main
```

To share a working web application:

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose the GitHub repository and the `main` branch.
4. Set the entry-point file to `app.py`.
5. Select a supported Python version, preferably Python 3.11 or 3.12.
6. Deploy the app and share the resulting `streamlit.app` link.

The deployed application will ask each visitor to upload an `.xlsx` workbook. Do not bundle or publish the supplied customer-level workbook unless you have permission to distribute it.
