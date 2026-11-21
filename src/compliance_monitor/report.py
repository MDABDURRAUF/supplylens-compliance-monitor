"""Printable outputs: the supplier compliance profile and the audit summary report."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Records
from .rules import SEVERITIES, RuleSet
from .scores import ageing_table, headline, rule_summary, supplier_scores
from .store import FindingStore

DISCLAIMER = ("Decision support, not legal advice. This report lists records that break the rules as they are configured in this tool. "
              "The deadlines, grace periods and retention periods used are settings chosen by the user, not statements of what any law, "
              "regulation or contract requires. A clean result does not show that a supplier or shipment is compliant, and a screening "
              "result is only ever a potential match for a person to review. Confirm requirements and decisions with your customs "
              "broker or counsel.")


def _date(x) -> str:
    return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%d %b %Y}"


def _num(x) -> str:
    return "" if x is None or pd.isna(x) else f"{x:,.0f}"


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4;vertical-align:top}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff;white-space:nowrap}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""
_TAG = {"critical": "#7a1f2b", "high": "#d1495b", "medium": "#e9a23b", "low": "#9aa5b1",
        "open": "#d1495b", "in review": "#3a6ea5", "resolved": "#2a9d8f", "waived": "#9aa5b1"}


def _table(df: pd.DataFrame, formats: dict | None = None, empty: str = "Nothing to show.") -> str:
    if df is None or df.empty:
        return f'<p class="note">{escape(empty)}</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if c in ("severity", "status"):
                cells.append(f'<td><span class="tag" style="background:{_TAG.get(v, "#9aa5b1")}">{escape(str(v))}</span></td>')
            elif c in formats:
                cells.append(f'<td class="num">{escape(formats[c](v))}</td>')
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
                cells.append(f'<td class="num">{escape(_num(v))}</td>')
            else:
                cells.append(f"<td>{escape('' if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str) -> str:
    body = f'<p class="caveat">{escape(DISCLAIMER)}</p>' + body
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">{escape(subtitle)}</p>{body}</main></body></html>')


def _evidence_text(evidence: dict) -> str:
    return "; ".join(f"{k.replace('_', ' ')}: {json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in evidence.items() if v is not None)


_FINDING_COLUMNS = ["finding_id", "severity", "status", "title", "record_id", "summary", "owner", "first_seen", "due_date", "recurrence"]
_FINDING_FORMATS = {"first_seen": _date, "due_date": _date, "closed_on": _date, "finding_id": lambda v: f"{v}", "recurrence": lambda v: f"{v}"}


def build_supplier_profile(store: FindingStore, records: Records, supplier_id: str, as_of, simulated: bool = False) -> str:
    """One supplier's compliance profile as an HTML page: score, what is on file, open findings with evidence, and history."""
    as_of = pd.Timestamp(as_of).normalize()
    hit = records.suppliers[records.suppliers["supplier_id"] == supplier_id]
    if hit.empty:
        raise KeyError(f"No supplier {supplier_id}")
    s = hit.iloc[0]
    findings = store.findings(as_of)
    mine = findings[findings["supplier_id"] == supplier_id]
    active, closed = mine[mine["active"]], mine[~mine["active"]]
    score = supplier_scores(findings, records.suppliers)
    score = score[score["supplier_id"] == supplier_id].iloc[0]
    shipments = records.shipments[records.shipments["supplier_id"] == supplier_id]
    lead = (f"{s['supplier_name']} has a compliance score of {score['score']:.0f} out of 100 on {_date(as_of)}, with {len(active)} open "
            f"finding(s), {int(active['overdue'].sum())} of them past the due date. {len(closed)} earlier finding(s) are closed.")
    if (active["category"] == "screening").any():
        lead += " A potential screening match is waiting for human review; it is not a determination."
    body = f'<p class="lead">{escape(lead)}</p>'
    body += _kpis([("compliance score", f"{score['score']:.0f}"), ("open findings", f"{len(active)}"), ("overdue", f"{int(active['overdue'].sum())}"),
                   ("closed findings", f"{len(closed)}"), ("shipments on record", f"{len(shipments):,}")])
    facts = pd.DataFrame([("Supplier id", s["supplier_id"]), ("Country", s["country"]), ("Other origins on record", s["other_origins"] or "none"),
                          ("Onboarded", _date(s["onboarded_date"])), ("Insurance expiry", _date(s["insurance_expiry"]) or "not recorded"),
                          ("Code of conduct acknowledged", _date(s["code_of_conduct_ack_date"]) or "not recorded"),
                          ("Last audit", _date(s["last_audit_date"]) or "not recorded")], columns=["item", "on record"])
    body += "<h2>On record</h2>" + _table(facts)
    certs = records.certificates[records.certificates["supplier_id"] == supplier_id].copy()
    certs["days_to_expiry"] = (certs["expiry_date"] - as_of).dt.days
    body += "<h2>Certificates</h2>" + _table(certs[["certificate_id", "certificate_type", "issue_date", "expiry_date", "days_to_expiry"]],
                                             {"issue_date": _date, "expiry_date": _date, "days_to_expiry": lambda v: f"{v:+.0f}"},
                                             "No certificates on record.")
    body += "<h2>Open findings</h2>" + _table(active[_FINDING_COLUMNS], _FINDING_FORMATS, "No open findings.")
    if len(active):
        ev = active.assign(evidence=active["evidence"].map(_evidence_text))[["finding_id", "rule_id", "evidence", "remediation"]]
        body += ('<h2>Evidence and next step</h2><p class="note">The values each rule looked at when the finding was raised.</p>'
                 + _table(ev, {"finding_id": lambda v: f"{v}"}))
    body += "<h2>Closed findings</h2>" + _table(
        closed[["finding_id", "severity", "status", "title", "record_id", "first_seen", "closed_on", "resolution", "waiver_reason"]].head(60),
        _FINDING_FORMATS, "No closed findings.")
    trail = store.audit_trail()
    trail = trail[trail["supplier_id"] == supplier_id][["at", "finding_id", "rule_id", "actor", "field", "old_value", "new_value", "note"]]
    body += ('<h2>Audit trail</h2><p class="note">Every change to this supplier\'s findings, oldest first (latest 80 shown).</p>'
             + _table(trail.tail(80), {"finding_id": lambda v: f"{v}"}, "No changes recorded."))
    body += ('<h2>How the score is worked out</h2><p class="note">100 minus points for each finding that is open or in review '
             "(critical 25, high 10, medium 4, low 1 by default), counting at most three findings per rule. It summarises this tool's "
             "open findings and is not a rating of the supplier.</p>")
    sub = "SupplyLens Compliance Monitor" + (" | simulated data, fictional supplier" if simulated else "")
    return _page(f"Supplier compliance profile: {s['supplier_name']}", sub, body)


def build_audit_report(store: FindingStore, records: Records, rules: RuleSet, as_of, simulated: bool = False) -> str:
    """The audit summary as an HTML page: where things stand, what is overdue, what changed, and the rules in force."""
    as_of = pd.Timestamp(as_of).normalize()
    findings = store.findings(as_of)
    active = findings[findings["active"]] if len(findings) else findings
    scores = supplier_scores(findings, records.suppliers)
    trail = store.audit_trail()
    body = f'<p class="lead">{escape(headline(findings, as_of))}</p>'
    by = active["severity"].value_counts() if len(active) else {}
    body += _kpis([("open findings", f"{len(active)}"), *[(s, f"{int(by.get(s, 0))}") for s in SEVERITIES],
                   ("overdue", f"{int(active['overdue'].sum()) if len(active) else 0}"),
                   ("average supplier score", f"{scores['score'].mean():.1f}" if len(scores) else "n/a")])
    counts = records.counts()
    body += (f'<p class="note">Records checked: {counts["suppliers"]:,} suppliers, {counts["certificates"]:,} certificates, '
             f'{counts["shipments"]:,} shipments, {counts["products"]:,} products. {len(rules.enabled())} of {len(rules)} rules switched on.</p>')
    if len(findings):
        body += "<h2>Open findings by severity and age</h2>" + _table(ageing_table(findings))
        overdue = active[active["overdue"]].sort_values("due_date")
        body += (f"<h2>Overdue</h2><p class=\"note\">{len(overdue)} open finding(s) past the due date; the 30 longest overdue are shown.</p>"
                 + _table(overdue[["finding_id", "severity", "status", "title", "supplier_name", "record_id", "owner", "due_date"]].head(30),
                          _FINDING_FORMATS, "Nothing is overdue."))
        body += "<h2>By rule</h2>" + _table(rule_summary(findings))
        body += ('<h2>Suppliers with the lowest scores</h2><p class="note">The score summarises open findings in this tool; it is not a '
                 "rating of the supplier.</p>" + _table(scores.head(15), {"score": lambda v: f"{v:.0f}"}))
        screening = active[active["category"] == "screening"]
        body += ('<h2>Screening: potential matches awaiting review</h2><p class="note">Name similarity only. Each row needs a person to '
                 "compare identifying details; none is a determination.</p>"
                 + _table(screening[["finding_id", "supplier_name", "summary", "score", "owner", "status"]],
                          {"finding_id": lambda v: f"{v}", "score": lambda v: f"{v:.2f}"}, "No potential matches are open."))
        repeats = active[active["recurrence"] > 0]
        body += (f"<h2>Repeat findings</h2><p class=\"note\">{len(repeats)} open finding(s) are repeats: the same supplier has had a finding "
                 "under the same rule before.</p>")
        waived = findings[findings["status"] == "waived"]
        body += "<h2>Waivers</h2>" + _table(waived[["finding_id", "severity", "title", "supplier_name", "record_id", "closed_on", "waiver_reason"]],
                                            _FINDING_FORMATS, "No findings have been waived.")
    runs = store.runs()
    if len(runs):
        body += ('<h2>Trend</h2><p class="note">Totals after each check run (latest 20 shown).</p>'
                 + _table(runs[["as_of", "shipments", "new", "auto_closed", "open_total", "open_critical", "open_high", "overdue", "mean_score"]].tail(20),
                          {"as_of": _date, "mean_score": lambda v: f"{v:.1f}"}))
    if len(trail):
        people = trail[trail["actor"] != "system"]
        activity = pd.DataFrame([("changes recorded in the audit trail", len(trail)), ("made by the system (raised or closed on re-check)", len(trail) - len(people)),
                                 ("made by people", len(people)), ("people who made changes", people["actor"].nunique())], columns=["audit trail", "count"])
        body += "<h2>Workflow activity</h2>" + _table(activity)
    table = rules.to_table()[["id", "title", "category", "severity", "enabled", "params"]]
    table["enabled"] = table["enabled"].map({True: "on", False: "off"})
    body += ('<h2>Rules in force</h2><p class="note">Parameters are the user\'s settings on the date of this report. They are not '
             "statements of any legal requirement.</p>" + _table(table))
    sub = "SupplyLens Compliance Monitor" + (" | simulated data" if simulated else "")
    return _page(f"Compliance audit summary: {_date(as_of)}", sub, body)


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
