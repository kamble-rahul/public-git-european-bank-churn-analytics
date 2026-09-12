# Security and data disclosure policy

## Supported version

Only the current `main` branch is maintained.

## Reporting a vulnerability

Do not open a public issue containing credentials, customer data, private repository links, or an
exploitable security detail. Contact the repository owner privately through the contact method on
their GitHub profile and include only the minimum information required to reproduce the problem.

## Sensitive-data incidents

If customer-level banking data is committed or exposed:

1. Stop sharing the affected link.
2. Notify the data owner and repository owner using an approved private channel.
3. Revoke any exposed credential immediately.
4. Follow the organization's incident-response and legal notification process.
5. Do not assume that deleting the latest commit removes data from Git history or existing clones.

## Deployment posture

The public Streamlit app does not accept file uploads. It loads only the bundled standardized
dataset, whose original customer IDs and surnames have been replaced before publication. The
private source workbook must never be committed or deployed. This remains an educational demo,
not an audited banking system; do not publish regulated or production customer information without
formal authorization, privacy review, security controls, and an approved processing agreement.
