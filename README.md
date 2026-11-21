# SupplyLens Compliance Monitor

**Automated supply chain compliance monitoring: check supplier, shipment and product records against rules you
configure, raise findings with evidence, and track each one until it is resolved.**

> **Decision support, not legal advice.** This tool tests records against rules as *you* configure them. It does not
> know what any law, regulation or contract requires of you. Every deadline, grace period, age limit and retention
> period is a setting with a neutral placeholder default: set it to the requirement that applies to you, and confirm
> it with your customs broker or counsel. A record that passes every rule is not thereby compliant, and a screening
> result is only ever a potential match for a person to review.

Each day the monitor does three things:

1. **Checks** suppliers, certificates, import entries and products against the rules that are switched on.
2. **Raises a finding** for every record that breaks a rule, with the values the rule looked at and a suggested next step.
3. **Tracks** each finding: status, owner, due date, a full audit trail, and automatic closing when the record is fixed.

The included data is simulated, with fictional suppliers, placeholder countries ("Country A") and an invented
tariff schedule. Results on it show that the method works; they are not findings about real companies or trade.

## What it does

| Question | Where | Function |
|---|---|---|
| What is open today, how serious, and who owns it? | **Open findings** | `daily_check`, `FindingStore.findings`, `headline` |
| What did the rule see, and what has been done about it? | **Open findings**, finding detail | `FindingStore.update`, `audit_trail` |
| How does one supplier stand? | **Supplier profile** | `supplier_scores`, `build_supplier_profile` |
| Which supplier names resemble an entry on my screening list? | **Screening review** | `screen_names`, `name_similarity` |
| Which rules run, and with what deadlines and periods? | **Rules** | `default_rules`, `RuleSet` |
| How often are the checks right, and what do they miss? | **Detection performance** | `evaluate.evaluate_sample`, `outlier_methods`, `screening_benchmark` |
| Is the backlog shrinking? | **Trends** | `FindingStore.runs`, `ageing_table`, `build_audit_report` |

Fifteen rules in six categories: document completeness, expiry, data consistency, timeliness, record retention
and screening. Outputs: findings as CSV, a single-file HTML compliance profile per supplier, an HTML audit
summary, and a command line for all of it.

## Results on the simulated data

60 suppliers, 145 certificates, 5,000 import entries over 540 days and 120 products, with 595 violations of
15 kinds injected and written down. The default rules, run with every violation still in place:

| | Findings raised | Injected violations | Precision | Recall | False positives per 1,000 records |
|---|---|---|---|---|---|
| **All 15 rules** | 657 | 595 | **87.1%** | **96.1%** | 2.4 |
| 11 rules that test a recorded date, flag or format | 362 | 362 | 100% | 100% | 0 |
| Origin differs from supplier's countries | 147 | 75 | 51.0% | 100% | 14.4 |
| Declared unit value out of line | 74 | 75 | 89.2% | 88.0% | 1.6 |
| Description does not fit tariff chapter | 64 | 75 | 96.9% | 82.7% | 0.4 |
| Name resembles a screening-list entry | 10 | 8 | 70.0% | 87.5% | 50.0 (3 of 60 suppliers) |

- The eleven rules that test a recorded fact recover exactly what was injected. That shows the plumbing works and
  nothing more; the informative rows are the four checks that can be wrong.
- The origin rule finds every injected mismatch, but half of what it raises are shipments from three suppliers'
  second sites that are honest and simply missing from the supplier table.
- The unit-value check trades precision for recall with its threshold: at a robust score of 3.5 it finds 97.3% of
  the injected values at 70.2% precision, at 6.0 it finds 84.0% at 95.5%. An IsolationForest set to flag 2% of
  each tariff code finds 96.0% at 64.3% precision.
- The description classifier names the declared chapter for 98.5% of untampered entries across the 8 simulated
  chapters (cross-fitted).
- Name screening raises the 3 simulated look-alike companies along with 7 of the 8 altered names, and misses the
  one written with an abbreviated word. On 60 generated variants of each kind, added legal forms, changed word
  order, one or two typos, and punctuation and accents are all raised; 21.7% of names with a word dropped are
  raised and none with a word abbreviated. None of 400 unrelated invented names is raised.
- Replayed week by week over 13 checks from 7 April to 30 June 2026 with a scripted reviewer, 502 findings are
  raised (363 of them the backlog found by the first check), 46 close automatically when the record is fixed, 86
  are repeats for the same rule and supplier, and open findings fall from 363 to 66 while the average supplier
  score rises from 65.8 to 91.0. The audit trail holds 1,832 entries.

These numbers come from simulated data with injected violations, which are cleaner than real ones. Expect more
false positives and more misses on real records, and review a sample of findings before relying on any rule.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-compliance-monitor.git
cd supplylens-compliance-monitor
pip install -e ".[app,dev]"
streamlit run app.py
```

On first start the dashboard replays twelve weeks of the simulated records into a findings database, which takes
about half a minute. The database is `compliance_monitor.db` next to `app.py`; set the environment variable
`COMPLIANCE_MONITOR_DB` to keep it elsewhere.

Command line:

```bash
compliance-monitor sample --out data
compliance-monitor check --data data --as-of 2026-06-30 --out findings.csv
compliance-monitor findings --status open --severity critical high
compliance-monitor update --id 12 --actor "Analyst A" --status "in review" --owner "Analyst A"
compliance-monitor profile --data data --supplier SUP-034 --out profile.html
compliance-monitor report --data data --out audit_summary.html
compliance-monitor evaluate
compliance-monitor replay --weeks 12
```

`check` takes `--rules rules.json` and `--screening-list your_list.csv`. Run it once a day on a fresh extract of
your records; findings carry over between runs.

Python:

```python
import compliance_monitor as cm

records = cm.load_records("data")                      # or cm.generate_sample().records()
screening = cm.load_screening_list("data/screening_list.csv")

rules = cm.default_rules()
rules.set_params("TIM-01", deadline_days=10)           # your requirement; the default is a placeholder
rules.set_enabled("EXP-05", False)
rules.save("rules.json")

store = cm.FindingStore("compliance_monitor.db")
result = cm.daily_check(records, store, rules, as_of="2026-06-30", screening_list=screening)
print(result["new"], "new,", result["auto_closed"], "closed because the record was fixed")

open_findings = store.findings("2026-06-30", status=["open", "in review"])
first = open_findings.iloc[0]
print(first["summary"], first["evidence"])             # what the rule saw
store.update(first["finding_id"], actor="Analyst A", status="in review", owner="Analyst A")
print(store.audit_trail(first["finding_id"]))          # who changed what, and when

print(cm.supplier_scores(store.findings("2026-06-30"), records.suppliers).head())
cm.write_html(cm.build_audit_report(store, records, rules, "2026-06-30"), "audit_summary.html")
```

## Input data

Four CSV files in one folder. Only `suppliers.csv` is required; a table without a file is treated as empty.

| File | Required columns | Optional columns |
|---|---|---|
| `suppliers.csv` | `supplier_id`, `supplier_name`, `country` | `other_origins` (separated by semicolons), `onboarded_date`, `insurance_expiry`, `code_of_conduct_ack_date`, `last_audit_date` |
| `certificates.csv` | `certificate_id`, `supplier_id`, `certificate_type`, `expiry_date` | `issue_date` |
| `shipments.csv` | `entry_id`, `supplier_id`, `entry_date`, `tariff_code`, `goods_description`, `declared_value_usd`, `quantity`, `country_of_origin` | `product_id`, `unit`, `filed_at`, `records_purged_on`, `has_commercial_invoice`, `has_packing_list`, `has_bill_of_lading`, `has_certificate_of_origin` |
| `products.csv` | `product_id`, `product_name`, `category` | `supplier_id`, `tariff_code`, `material_declaration_date` |

`has_...` columns are yes/no. A blank in an optional column means "not on file" or "has not happened", and the
rule for that column will raise it, so switch off rules you have no data for. Files are validated on load;
problems are reported with row counts, and nothing is dropped silently.

**Screening list.** A CSV with a `name` column, and optionally `list_id` and `source`. The package ships only a
twelve-entry *fictional* list with made-up names so that the demo runs. For real use, obtain the list or lists
that apply to you from their publishers and load them with `--screening-list` or on the Screening review view.

**Rules.** `rules.json` holds the rule set. `compliance-monitor sample` writes the defaults; see
`examples/sample_records/rules.json`.

## How it works

**Rules are data.** A rule has an id, title, category, severity, the kind of record it applies to, parameters,
remediation text, and the name of the check that implements it. Rules can be switched off, re-parameterised, and
saved to or loaded from JSON; a file may use the same check twice with different parameters. The defaults:

| Rule | What it tests | Parameters (placeholder defaults) |
|---|---|---|
| DOC-01 Shipment documents on file | Each document in the list is marked as on file | `required_documents` (invoice, packing list, bill of lading, certificate of origin), `grace_days` 0 |
| DOC-02 Code of conduct acknowledged | The supplier has an acknowledgment date | `grace_days` 30 after onboarding |
| DOC-03 Material declaration on file | The product has a declaration date | none |
| EXP-01 Certificate expired | Expiry date plus grace is before the check date | `grace_days` 0 |
| EXP-02 Certificate expiring soon | Not yet expired, expires within the window | `days_ahead` 30 |
| EXP-03 Insurance expired | Expiry date plus grace is before the check date, or missing | `grace_days` 0 |
| EXP-04 Supplier audit overdue | Last audit older than the limit, or missing | `max_age_months` 12 |
| EXP-05 Material declaration out of date | Declaration older than the limit | `max_age_months` 24 |
| CON-01 Tariff code format | Code fully matches a regular expression | `pattern` `\d{4}\.\d{2}\.\d{4}` |
| CON-02 Origin differs from supplier's countries | Origin is the supplier's country or one of its `other_origins` | none |
| CON-03 Declared unit value out of line | Unit value is not an outlier within its tariff code | `threshold` 4.5, `min_group` 10, `method` |
| CON-04 Description does not fit tariff chapter | Declared chapter is not unlikely for the description | `max_declared_probability` 0.05, `min_predicted_probability` 0.5, `chapter_digits` 2 |
| TIM-01 Filed after deadline | Filed by the end of the day N days after entry | `deadline_days` 5 |
| RET-01 Records purged before retention period | Records not deleted within N years of entry | `retention_years` 5 |
| SCR-01 Name resembles a screening-list entry | Name similarity to every list entry is below the threshold | `threshold` 0.85 |

The parameter defaults are neutral starting points chosen for the simulated data. They are not requirements.

**Name screening.** Names are reduced to a comparable form: accents removed, lower case, punctuation to spaces,
legal-form words such as "ltd" or "inc" dropped. Two names are then compared in two ways: by word (each word is
paired with its most similar word in the other name, ignoring order) and by character (after sorting the words).
The score is the mean of the two, from 0 to 1. A supplier scoring at or above the threshold against any entry is
raised as "Potential match: needs human review". The score decides nothing by itself.

**Declared unit value.** Declared value divided by quantity, compared with other entries under the same tariff
code. The score is the distance of the log unit value from the group median in robust standard deviations
(1.4826 times the median absolute deviation); entries beyond the threshold are raised. Codes with fewer than
`min_group` entries are not scored. An IsolationForest per tariff code is available as an alternative method.

**Description against tariff chapter.** A text classifier (TF-IDF on the words of the description, then logistic
regression) is trained on your own entries with the declared chapter as the label, so it learns how each chapter
is usually described in your data and knows nothing about any tariff schedule. Predictions are cross-fitted: no
entry is judged by a model that saw its own label. An entry is raised when the model gives its declared chapter a
probability below 5% and another chapter at least 50%.

**Findings workflow.** Findings live in SQLite. Each has a status (open, in review, resolved, or waived with a
required reason), an owner, a due date set from its severity (2, 7, 30 or 60 days by default), and the evidence
seen when it was raised. Every change is appended to an audit trail with the actor, the field, the old and new
value and the time. A new finding is marked as a repeat when the same supplier has had a finding under the same
rule before.

**Re-check.** At each run, an open finding whose record no longer breaks its rule is closed automatically. A
finding that a person resolved or waived stays closed while the violation persists, so a late filing that has
been reviewed is not raised again every day. If the record is fixed and later breaks the rule again, a new
finding opens. Findings of a rule that is switched off are left as they are.

**Scores and trends.** A supplier's score is 100 minus points for each finding that is open or in review
(critical 25, high 10, medium 4, low 1), counting at most three findings per rule, never below 0. The totals of
every run are stored, which gives the trend of open findings by severity, overdue findings and the average score.

**Measured against ground truth.** The simulator builds clean records, injects violations and writes each one
down, and also plants honest records that look wrong (unrecorded second sites, extreme but true unit values,
vague descriptions, look-alike company names). `evaluate` scores every rule against that table.

## Limits

- This is decision support, not legal advice. The tool does not know your legal obligations, and its defaults are
  placeholders, not requirements.
- The sample data is simulated. Its countries, companies, tariff codes, chapters and screening list are invented,
  and its violations are cleaner than real ones.
- Rules test what is recorded. A document marked as on file is not opened or read, and a certificate date is not
  verified with its issuer.
- A record that passes every rule is not thereby compliant; the rules cover only what they are configured to cover.
- Name matching misses names that are abbreviated, translated, transliterated or have words dropped, and raises
  unrelated companies with similar names. It compares names only, not addresses, owners or identifiers, and it
  compares every name with every entry, so very large lists are slow.
- The unit-value and description checks compare a record with its peers. They need enough entries per tariff code
  or chapter, they cannot tell which of two disagreeing records is right, and a problem shared by most peers looks
  normal to them.
- The origin rule depends on the supplier table listing every legitimate origin.
- The supplier score summarises open findings in this tool. It is not a rating of the supplier.
- Automatic closing trusts the data: a record that disappears from an extract closes its findings just as a fixed
  record does. Runs are expected in date order.
- The dashboard keeps uploaded records in memory for the session only; the findings database persists.

## Tests

```bash
pytest
```

Twenty-two tests: every rule against small hand-made records with the boundary cases worked out by hand, name
normalisation and similarity by hand, the unit-value score by hand, the description classifier on a small example,
the findings workflow and audit trail, automatic closing on re-check and repeat detection, the supplier score by
hand, detection measured against the injected violations, the replay, the reports, and the command line end to end.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): supply chain disruption analytics on purchase orders
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses
- [SupplyLens Network Simulator](https://github.com/MDABDURRAUF/supplylens-network-simulator): disruption scenarios for service and goods networks

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
