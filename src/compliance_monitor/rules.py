"""The rules engine: declarative rules, the checks behind them, and one function that runs them all.

A rule is data: an id, a title, a category, a severity, the kind of record it
applies to, parameters, and remediation text. The logic lives in a small set
of named checks; a rule says which check to run and with which parameters.
Rules can be switched off, re-parameterised, saved to JSON and loaded back,
and a JSON file may contain several rules that use the same check with
different parameters.

Every deadline, grace period, age limit and retention period below is a
parameter with a neutral placeholder default. None of them is a statement of
what any law, regulation or contract requires. Set each one to the requirement
that applies to you, and confirm it with your customs broker or counsel.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DOCUMENTS, Records
from .learning import chapter_fit, unit_value_outliers
from .screening import POTENTIAL_MATCH, screen_names

SEVERITIES = ["critical", "high", "medium", "low"]
CATEGORIES = ["document completeness", "expiry", "data consistency", "timeliness", "record retention", "screening"]
RECORD_TABLE = {"supplier": "suppliers", "certificate": "certificates", "shipment": "shipments", "product": "products"}
CONFIRM = "Set this to the requirement that applies to you; confirm with your customs broker or counsel."
FINDING_COLUMNS = ["finding_key", "rule_id", "title", "category", "severity", "record_type", "record_id", "supplier_id",
                   "supplier_name", "summary", "evidence", "score", "remediation"]


@dataclass
class Rule:
    """One compliance rule.

    ``check`` names the logic to run (a key of ``CHECKS``); ``applies_to`` is
    the kind of record a finding points at; ``params`` are the numbers and
    lists the check reads; ``description`` says in plain words what is tested.
    """

    id: str
    title: str
    category: str
    severity: str
    applies_to: str
    check: str
    description: str
    remediation: str
    params: dict = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"Rule {self.id}: severity must be one of {SEVERITIES}")
        if self.category not in CATEGORIES:
            raise ValueError(f"Rule {self.id}: category must be one of {CATEGORIES}")
        if self.applies_to not in RECORD_TABLE:
            raise ValueError(f"Rule {self.id}: applies_to must be one of {list(RECORD_TABLE)}")
        if self.check not in CHECKS:
            raise ValueError(f"Rule {self.id}: unknown check '{self.check}'. Known checks: {', '.join(sorted(CHECKS))}")


class RuleSet:
    """An ordered collection of rules with unique ids."""

    def __init__(self, rules: list[Rule]):
        ids = [r.id for r in rules]
        if len(set(ids)) != len(ids):
            raise ValueError("Rule ids must be unique")
        self.rules = list(rules)

    def __iter__(self):
        return iter(self.rules)

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(f"No rule {rule_id}")

    def enabled(self) -> list[Rule]:
        return [r for r in self.rules if r.enabled]

    def set_enabled(self, rule_id: str, enabled: bool) -> None:
        self[rule_id].enabled = bool(enabled)

    def set_params(self, rule_id: str, **params) -> None:
        """Change parameters of a rule. Only parameters the rule already has can be set."""
        rule = self[rule_id]
        unknown = set(params) - set(rule.params)
        if unknown:
            raise ValueError(f"Rule {rule_id} has no parameter(s): {', '.join(sorted(unknown))}")
        rule.params.update(params)

    def to_table(self) -> pd.DataFrame:
        return pd.DataFrame([{**asdict(r), "params": json.dumps(r.params)} for r in self.rules])

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.rules], indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RuleSet":
        try:
            raw = json.loads(text)
            return cls([Rule(**item) for item in raw])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Not a valid rules file: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RuleSet":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# -------------------------------------------------------------------- checks
def _plain(value):
    """Make a value safe to store as JSON evidence."""
    if value is None or (not isinstance(value, (list, tuple, dict, str)) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return f"{ts:%Y-%m-%d}" if ts == ts.normalize() else f"{ts:%Y-%m-%d %H:%M}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _finding(rule: Rule, record_id, supplier_id, summary: str, evidence: dict, score=None) -> dict:
    return {"finding_key": f"{rule.id}|{rule.applies_to}|{record_id}", "rule_id": rule.id, "title": rule.title, "category": rule.category,
            "severity": rule.severity, "record_type": rule.applies_to, "record_id": str(record_id), "supplier_id": str(supplier_id),
            "summary": summary, "evidence": {k: _plain(v) for k, v in evidence.items()},
            "score": None if score is None else round(float(score), 4), "remediation": rule.remediation}


def _day(ts) -> str:
    return f"{pd.Timestamp(ts):%d %b %Y}"


def check_documents(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when any document in ``required_documents`` is not marked as on file.

    Shipments entered within the last ``grace_days`` days are not checked yet.
    """
    sh, as_of = ctx["records"].shipments, ctx["as_of"]
    required = list(rule.params["required_documents"])
    unknown = [d for d in required if f"has_{d}" not in sh.columns]
    if unknown:
        raise ValueError(f"Rule {rule.id}: no column for document(s) {', '.join(unknown)} (expected has_<document>)")
    due = sh["entry_date"] + pd.Timedelta(days=int(rule.params["grace_days"])) <= as_of
    absent = ~sh[[f"has_{d}" for d in required]].to_numpy(dtype=bool)
    out = []
    for i in np.flatnonzero(due.to_numpy() & absent.any(axis=1)):
        row = sh.iloc[i]
        gone = [d.replace("_", " ") for d, a in zip(required, absent[i]) if a]
        out.append(_finding(rule, row["entry_id"], row["supplier_id"], f"Not on file: {', '.join(gone)}",
                            {"entry_date": row["entry_date"], "missing_documents": gone, "required_documents": required,
                             "grace_days": rule.params["grace_days"]}))
    return out


def check_acknowledgment(rule: Rule, ctx: dict) -> list[dict]:
    """A supplier is raised when it has no code-of-conduct acknowledgment date and was onboarded more than ``grace_days`` days ago.

    A supplier with no onboarding date is treated as past the grace period.
    """
    sup, as_of = ctx["records"].suppliers, ctx["as_of"]
    past_grace = sup["onboarded_date"].isna() | (sup["onboarded_date"] + pd.Timedelta(days=int(rule.params["grace_days"])) < as_of)
    out = []
    for _, row in sup[sup["code_of_conduct_ack_date"].isna() & past_grace].iterrows():
        out.append(_finding(rule, row["supplier_id"], row["supplier_id"], "No code-of-conduct acknowledgment on file",
                            {"onboarded_date": row["onboarded_date"], "code_of_conduct_ack_date": None, "grace_days": rule.params["grace_days"]}))
    return out


def check_declaration_present(rule: Rule, ctx: dict) -> list[dict]:
    """A product is raised when it has no material declaration date."""
    pr = ctx["records"].products
    return [_finding(rule, row["product_id"], row["supplier_id"], f"No material declaration on file for {row['product_name']}",
                     {"product_name": row["product_name"], "category": row["category"], "material_declaration_date": None})
            for _, row in pr[pr["material_declaration_date"].isna()].iterrows()]


def check_declaration_age(rule: Rule, ctx: dict) -> list[dict]:
    """A product is raised when its material declaration is dated more than ``max_age_months`` months before the check date."""
    pr, as_of = ctx["records"].products, ctx["as_of"]
    limit = as_of - pd.DateOffset(months=int(rule.params["max_age_months"]))
    return [_finding(rule, row["product_id"], row["supplier_id"],
                     f"Material declaration for {row['product_name']} is dated {_day(row['material_declaration_date'])}",
                     {"product_name": row["product_name"], "material_declaration_date": row["material_declaration_date"],
                      "max_age_months": rule.params["max_age_months"], "oldest_accepted_date": limit})
            for _, row in pr[pr["material_declaration_date"] < limit].iterrows()]


def check_certificate_expired(rule: Rule, ctx: dict) -> list[dict]:
    """A certificate is raised when its expiry date plus ``grace_days`` is before the check date."""
    ce, as_of = ctx["records"].certificates, ctx["as_of"]
    late = ce["expiry_date"] + pd.Timedelta(days=int(rule.params["grace_days"])) < as_of
    return [_finding(rule, row["certificate_id"], row["supplier_id"], f"{row['certificate_type']} expired on {_day(row['expiry_date'])}",
                     {"certificate_type": row["certificate_type"], "expiry_date": row["expiry_date"],
                      "days_past_expiry": (as_of - row["expiry_date"]).days, "grace_days": rule.params["grace_days"]})
            for _, row in ce[late].iterrows()]


def check_certificate_expiring(rule: Rule, ctx: dict) -> list[dict]:
    """A certificate is raised when it has not expired yet and expires within ``days_ahead`` days of the check date."""
    ce, as_of = ctx["records"].certificates, ctx["as_of"]
    soon = (ce["expiry_date"] >= as_of) & (ce["expiry_date"] <= as_of + pd.Timedelta(days=int(rule.params["days_ahead"])))
    return [_finding(rule, row["certificate_id"], row["supplier_id"],
                     f"{row['certificate_type']} expires on {_day(row['expiry_date'])} (in {(row['expiry_date'] - as_of).days} days)",
                     {"certificate_type": row["certificate_type"], "expiry_date": row["expiry_date"],
                      "days_to_expiry": (row["expiry_date"] - as_of).days, "days_ahead": rule.params["days_ahead"]})
            for _, row in ce[soon].iterrows()]


def check_insurance_expired(rule: Rule, ctx: dict) -> list[dict]:
    """A supplier is raised when its insurance expiry date plus ``grace_days`` is before the check date, or when no date is recorded."""
    sup, as_of = ctx["records"].suppliers, ctx["as_of"]
    late = sup["insurance_expiry"].isna() | (sup["insurance_expiry"] + pd.Timedelta(days=int(rule.params["grace_days"])) < as_of)
    out = []
    for _, row in sup[late].iterrows():
        text = "No insurance expiry date on file" if pd.isna(row["insurance_expiry"]) else f"Insurance expired on {_day(row['insurance_expiry'])}"
        out.append(_finding(rule, row["supplier_id"], row["supplier_id"], text,
                            {"insurance_expiry": row["insurance_expiry"], "grace_days": rule.params["grace_days"]}))
    return out


def check_audit_age(rule: Rule, ctx: dict) -> list[dict]:
    """A supplier is raised when its last audit is dated more than ``max_age_months`` months before the check date, or is not recorded."""
    sup, as_of = ctx["records"].suppliers, ctx["as_of"]
    limit = as_of - pd.DateOffset(months=int(rule.params["max_age_months"]))
    out = []
    for _, row in sup[sup["last_audit_date"].isna() | (sup["last_audit_date"] < limit)].iterrows():
        text = "No audit date on file" if pd.isna(row["last_audit_date"]) else f"Last audit was on {_day(row['last_audit_date'])}"
        out.append(_finding(rule, row["supplier_id"], row["supplier_id"], text,
                            {"last_audit_date": row["last_audit_date"], "max_age_months": rule.params["max_age_months"],
                             "oldest_accepted_date": limit}))
    return out


def check_code_format(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when its tariff code, with surrounding spaces removed, does not fully match the regular expression ``pattern``."""
    sh = ctx["records"].shipments
    try:
        pattern = re.compile(rule.params["pattern"])
    except re.error as exc:
        raise ValueError(f"Rule {rule.id}: pattern is not a valid regular expression ({exc})") from exc
    ok = sh["tariff_code"].map(lambda code: bool(pattern.fullmatch(code)))
    return [_finding(rule, row["entry_id"], row["supplier_id"],
                     "Tariff code is blank" if row["tariff_code"] == "" else f"Tariff code '{row['tariff_code']}' does not match the expected format",
                     {"tariff_code": row["tariff_code"], "expected_pattern": rule.params["pattern"], "entry_date": row["entry_date"]})
            for _, row in sh[~ok.to_numpy(dtype=bool)].iterrows()]


def check_origin(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when its country of origin is neither the supplier's country nor one of the supplier's ``other_origins``.

    ``other_origins`` is a list separated by semicolons in the suppliers
    table. Comparison ignores case and surrounding spaces. A blank origin is
    raised too.
    """
    rec = ctx["records"]
    allowed = {}
    for _, s in rec.suppliers.iterrows():
        allowed[s["supplier_id"]] = {s["country"].strip().lower()} | {o.strip().lower() for o in s["other_origins"].split(";") if o.strip()}
    sh = rec.shipments
    fine = np.array([o.strip().lower() in allowed.get(s, ()) for o, s in zip(sh["country_of_origin"], sh["supplier_id"])], dtype=bool)
    out = []
    for i in np.flatnonzero(~fine):
        row = sh.iloc[i]
        ok = allowed.get(row["supplier_id"], set())
        text = "Country of origin is blank" if not row["country_of_origin"] else \
            f"Origin {row['country_of_origin']} is not a country on record for this supplier"
        out.append(_finding(rule, row["entry_id"], row["supplier_id"], text,
                            {"country_of_origin": row["country_of_origin"], "countries_on_record": sorted(ok), "entry_date": row["entry_date"]}))
    return out


def check_unit_value(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when its declared unit value is an outlier among shipments with the same tariff code.

    See ``learning.unit_value_outliers`` for the exact calculation.
    """
    sh, p = ctx["records"].shipments, rule.params
    scores = unit_value_outliers(sh, threshold=float(p["threshold"]), min_group=int(p["min_group"]), method=p["method"],
                                 contamination=float(p["contamination"]))
    out = []
    for i in np.flatnonzero(scores["outlier"].to_numpy()):
        row, s = sh.iloc[i], scores.iloc[i]
        side = "above" if s["score"] > 0 else "below"
        out.append(_finding(rule, row["entry_id"], row["supplier_id"],
                            f"Declared unit value {s['unit_value']:,.2f} is far {side} the usual {s['group_median']:,.2f} for code {row['tariff_code']}",
                            {"tariff_code": row["tariff_code"], "declared_value_usd": row["declared_value_usd"], "quantity": row["quantity"],
                             "unit_value": s["unit_value"], "median_unit_value_for_code": s["group_median"],
                             "shipments_with_this_code": int(s["group_size"]), "robust_score": s["score"], "method": p["method"]},
                            score=abs(s["score"])))
    return out


def check_description(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when its goods description looks unlikely for the chapter of its declared tariff code.

    See ``learning.chapter_fit`` for the exact calculation.
    """
    sh, p = ctx["records"].shipments, rule.params
    fit = chapter_fit(sh, chapter_digits=int(p["chapter_digits"]), max_declared_probability=float(p["max_declared_probability"]),
                      min_predicted_probability=float(p["min_predicted_probability"]))
    out = []
    for i in np.flatnonzero(fit["unlikely"].to_numpy()):
        row, f = sh.iloc[i], fit.iloc[i]
        out.append(_finding(rule, row["entry_id"], row["supplier_id"],
                            f"Description '{row['goods_description']}' reads like chapter {f['predicted_chapter']}, declared chapter is {f['declared_chapter']}",
                            {"goods_description": row["goods_description"], "tariff_code": row["tariff_code"],
                             "declared_chapter": f["declared_chapter"], "probability_of_declared_chapter": f["p_declared"],
                             "most_likely_chapter": f["predicted_chapter"], "probability_of_most_likely_chapter": f["p_predicted"]},
                            score=1 - f["p_declared"]))
    return out


def check_filing(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when it was filed after its deadline, or has not been filed and the deadline has passed.

    The deadline is the end of the day ``deadline_days`` days after the entry
    date. A negative number puts the deadline before entry.
    """
    sh, as_of = ctx["records"].shipments, ctx["as_of"]
    deadline = sh["entry_date"] + pd.Timedelta(days=int(rule.params["deadline_days"]))
    filed_day = sh["filed_at"].dt.normalize()
    late = (filed_day > deadline) | (sh["filed_at"].isna() & (as_of > deadline))
    out = []
    for i in np.flatnonzero(late.to_numpy()):
        row = sh.iloc[i]
        if pd.isna(row["filed_at"]):
            text, days = f"Not filed; the deadline was {_day(deadline.iloc[i])}", (as_of - deadline.iloc[i]).days
        else:
            days = (filed_day.iloc[i] - deadline.iloc[i]).days
            text = f"Filed {days} day(s) after the deadline of {_day(deadline.iloc[i])}"
        out.append(_finding(rule, row["entry_id"], row["supplier_id"], text,
                            {"entry_date": row["entry_date"], "filed_at": row["filed_at"], "deadline": deadline.iloc[i],
                             "days_late": days, "deadline_days": rule.params["deadline_days"]}))
    return out


def check_retention(rule: Rule, ctx: dict) -> list[dict]:
    """A shipment is raised when its records were purged less than ``retention_years`` years after the entry date."""
    sh = ctx["records"].shipments
    keep_until = sh["entry_date"] + pd.DateOffset(years=int(rule.params["retention_years"]))
    early = sh["records_purged_on"].notna() & (sh["records_purged_on"] < keep_until)
    return [_finding(rule, sh.iloc[i]["entry_id"], sh.iloc[i]["supplier_id"],
                     f"Records purged on {_day(sh.iloc[i]['records_purged_on'])}, before the retention period ended on {_day(keep_until.iloc[i])}",
                     {"entry_date": sh.iloc[i]["entry_date"], "records_purged_on": sh.iloc[i]["records_purged_on"],
                      "keep_until": keep_until.iloc[i], "retention_years": rule.params["retention_years"]})
            for i in np.flatnonzero(early.to_numpy())]


def check_screening(rule: Rule, ctx: dict) -> list[dict]:
    """A supplier is raised for human review when its name scores at least ``threshold`` against an entry on the screening list.

    One finding per supplier, quoting the best-scoring entry and listing the
    others. See ``screening.name_similarity`` for the score. This is a
    potential match, never a determination.
    """
    matches = screen_names(ctx["records"].suppliers, ctx.get("screening_list"), threshold=float(rule.params["threshold"]))
    out = []
    for supplier_id, group in matches.groupby("supplier_id", sort=False):
        best = group.iloc[0]
        out.append(_finding(rule, supplier_id, supplier_id,
                            f"{POTENTIAL_MATCH}. '{best['supplier_name']}' resembles list entry '{best['list_name']}' (similarity {best['score']:.2f})",
                            {"supplier_name": best["supplier_name"], "list_entry": best["list_name"], "list_id": best["list_id"],
                             "list_source": best["source"], "similarity": best["score"], "token_similarity": best["token_similarity"],
                             "character_similarity": best["character_similarity"], "threshold": rule.params["threshold"],
                             "other_entries_above_threshold": group["list_name"].tolist()[1:], "status": POTENTIAL_MATCH},
                            score=best["score"]))
    return out


CHECKS = {
    "documents_on_file": check_documents, "acknowledgment_on_file": check_acknowledgment,
    "declaration_on_file": check_declaration_present, "declaration_age": check_declaration_age,
    "certificate_expired": check_certificate_expired, "certificate_expiring": check_certificate_expiring,
    "insurance_expired": check_insurance_expired, "audit_age": check_audit_age, "code_format": check_code_format,
    "origin_on_record": check_origin, "unit_value_outlier": check_unit_value, "description_fits_chapter": check_description,
    "filed_by_deadline": check_filing, "retention_period": check_retention, "name_screening": check_screening,
}


def default_rules() -> RuleSet:
    """The fifteen rules shipped with the package, with placeholder parameters.

    The defaults are neutral starting points for the simulated data, not
    requirements. Review every parameter before relying on a rule.
    """
    R = Rule
    return RuleSet([
        R("DOC-01", "Shipment documents on file", "document completeness", "high", "shipment", "documents_on_file",
          "Every document in required_documents is marked as on file for the shipment, once grace_days have passed since entry.",
          "Obtain the missing document from the supplier or forwarder and attach it to the entry file.",
          {"required_documents": list(DOCUMENTS), "grace_days": 0}),
        R("DOC-02", "Code of conduct acknowledged", "document completeness", "medium", "supplier", "acknowledgment_on_file",
          "The supplier has a code-of-conduct acknowledgment date, once grace_days have passed since onboarding.",
          "Send the code of conduct to the supplier and record the date of the signed acknowledgment.", {"grace_days": 30}),
        R("DOC-03", "Material declaration on file", "document completeness", "medium", "product", "declaration_on_file",
          "The product has a material declaration date.",
          "Request a material declaration for the product from the supplier and record its date.", {}),
        R("EXP-01", "Certificate expired", "expiry", "high", "certificate", "certificate_expired",
          "The certificate's expiry date plus grace_days is before the check date.",
          "Request the renewed certificate from the supplier and update the expiry date.", {"grace_days": 0}),
        R("EXP-02", "Certificate expiring soon", "expiry", "low", "certificate", "certificate_expiring",
          "The certificate has not expired and expires within days_ahead days.",
          "Ask the supplier for the renewal plan before the certificate lapses.", {"days_ahead": 30}),
        R("EXP-03", "Insurance expired", "expiry", "high", "supplier", "insurance_expired",
          "The supplier's insurance expiry date plus grace_days is before the check date, or no date is recorded.",
          "Request a current certificate of insurance and update the expiry date.", {"grace_days": 0}),
        R("EXP-04", "Supplier audit overdue", "expiry", "medium", "supplier", "audit_age",
          "The supplier's last audit is older than max_age_months, or no audit date is recorded.",
          "Schedule an audit and record its date.", {"max_age_months": 12}),
        R("EXP-05", "Material declaration out of date", "expiry", "low", "product", "declaration_age",
          "The product's material declaration is older than max_age_months.",
          "Request an updated material declaration from the supplier.", {"max_age_months": 24}),
        R("CON-01", "Tariff code format", "data consistency", "medium", "shipment", "code_format",
          "The tariff code fully matches the regular expression in pattern. The default expects digits written as 0000.00.0000.",
          "Correct the tariff code on the entry; confirm the classification with your customs broker.", {"pattern": r"\d{4}\.\d{2}\.\d{4}"}),
        R("CON-02", "Origin differs from supplier's countries", "data consistency", "medium", "shipment", "origin_on_record",
          "The shipment's country of origin is the supplier's country or one of the supplier's other_origins.",
          "Confirm the origin with the supplier. Correct the entry, or add the country to the supplier's other_origins if it is legitimate.", {}),
        R("CON-03", "Declared unit value out of line", "data consistency", "medium", "shipment", "unit_value_outlier",
          "The declared value per unit is not an outlier among shipments with the same tariff code (robust score within threshold).",
          "Compare the declared value and quantity with the commercial invoice; correct a keying or unit error, or note why the value is right.",
          {"method": "robust", "threshold": 4.5, "min_group": 10, "contamination": 0.02}),
        R("CON-04", "Description does not fit tariff chapter", "data consistency", "medium", "shipment", "description_fits_chapter",
          "A text classifier trained on these records does not find the declared chapter unlikely for the goods description.",
          "Have the classification reviewed by your customs broker; correct the code or the description.",
          {"chapter_digits": 2, "max_declared_probability": 0.05, "min_predicted_probability": 0.5}),
        R("TIM-01", "Filed after deadline", "timeliness", "high", "shipment", "filed_by_deadline",
          "The entry was filed by the end of the day deadline_days days after the entry date.",
          "File the entry if it is still open, and record the reason for the delay.", {"deadline_days": 5}),
        R("RET-01", "Records purged before retention period", "record retention", "high", "shipment", "retention_period",
          "The shipment's records were not purged within retention_years years of the entry date.",
          "Restore the records from backup or obtain copies from the broker or supplier.", {"retention_years": 5}),
        R("SCR-01", "Name resembles a screening-list entry", "screening", "critical", "supplier", "name_screening",
          "The supplier's name does not score at or above threshold against any entry on the loaded screening list.",
          "Have a person review the potential match against identifying details (address, registration, ownership) and record the outcome. "
          "The score alone decides nothing.", {"threshold": 0.85}),
    ])


def run_checks(records: Records, rules: RuleSet | None = None, as_of=None, screening_list: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every enabled rule on the records and return the findings.

    ``as_of`` is the check date (default: the latest shipment entry date).
    One row per finding with the rule, the record, a one-line summary, an
    evidence snapshot (a dict of the values the rule looked at) and, for
    scored checks, a score. ``finding_key`` identifies "this rule on this
    record" and is how a finding is recognised on the next run. The enabled
    rule ids are stored in ``findings.attrs["rules_checked"]``.
    """
    rules = default_rules() if rules is None else rules
    if as_of is None:
        as_of = records.shipments["entry_date"].max() if len(records.shipments) else pd.Timestamp.today()
    as_of = pd.Timestamp(as_of).normalize()
    ctx = {"records": records, "as_of": as_of, "screening_list": screening_list}
    rows = []
    for rule in rules.enabled():
        rows.extend(CHECKS[rule.check](rule, ctx))
    out = pd.DataFrame(rows, columns=[c for c in FINDING_COLUMNS if c != "supplier_name"])
    names = dict(zip(records.suppliers["supplier_id"], records.suppliers["supplier_name"]))
    out.insert(out.columns.get_loc("supplier_id") + 1, "supplier_name", out["supplier_id"].map(names).fillna(""))
    order = out["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    out = out.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)
    out.attrs["rules_checked"] = [r.id for r in rules.enabled()]
    out.attrs["as_of"] = as_of
    return out
