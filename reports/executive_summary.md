# Executive summary for public-sector and banking stakeholders

## Purpose

This project demonstrates how customer-level banking data can be converted into transparent churn
indicators, reusable customer segments, and a reviewed risk-ranking model. The dashboard is intended
for educational decision support and policy discussion, not direct customer eligibility decisions.

## Portfolio-level evidence

The supplied snapshot contains 10,000 customers, including 2,037 recorded exits. Overall churn is
20.37%. Churn is unevenly distributed:

- Germany records 32.44% churn, compared with 16.15% in France and 16.67% in Spain.
- Inactive customers record 26.85% churn versus 14.27% for active customers, a ratio of 1.88.
- Customers aged 46–59 record the highest age-band churn rate at 51.10%.
- Germany × age 46–59 records 67.28% churn in the supplied snapshot.
- Female customers record 25.07% churn versus 16.46% for male customers. This is an association and
  requires fairness review; it must not be interpreted as a causal explanation.
- At the 75th balance percentile, 2,500 customers qualify as high value and 592 exited, producing
  23.68% churn. Their churned balances total approximately 88.65 million in the dataset's unspecified
  currency.

## Recommended decisions

1. Investigate service, product, pricing, and channel differences in Germany before starting a broad
   retention campaign.
2. Run a randomized re-engagement experiment for inactive customers, with clear contact limits and
   an untreated control group.
3. Review the meaning and quality of `NumOfProducts`; the unusually high churn among customers with
   three or four products may indicate a genuine issue or a data-definition problem.
4. Prioritize high-balance churn cases for human review, but do not describe balance exposure as
   revenue loss without margin, fees, cost-to-serve, and lifetime-value information.
5. Monitor campaign lift, incremental retained customers, cost, complaints, and fairness—not only
   model accuracy.

## Governance requirements

- Confirm the dataset owner, collection purpose, lawful basis, retention period, and geographic
  scope before operational use.
- Validate whether the source is authorized; the workbook itself does not prove European Central
  Bank provenance.
- Remove direct identifiers and restrict access to customer-level financial information.
- Review error rates and outcomes across demographic and regional groups.
- Require a human decision owner and documented override process for any model-assisted action.

## Limits

The analysis uses one cross-sectional snapshot. It has no event timeline, contact history, stated
reason for exit, treatment outcome, revenue field, or causal design. The observed patterns are useful
for prioritizing investigation, but they do not establish why customers left or what intervention
will retain them.
