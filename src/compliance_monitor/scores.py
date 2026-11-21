"""Scores and summaries computed from the stored findings."""

from __future__ import annotations

import pandas as pd

from .rules import SEVERITIES

# Points taken off a supplier's score for each open finding, by severity. Placeholders: set them to your own policy.
SEVERITY_WEIGHTS = {"critical": 25, "high": 10, "medium": 4, "low": 1}
AGE_BANDS = ["0-7 days", "8-30 days", "31-90 days", "over 90 days"]


def supplier_scores(findings: pd.DataFrame, suppliers: pd.DataFrame, weights: dict | None = None, per_rule_cap: int = 3) -> pd.DataFrame:
    """A compliance score from 0 to 100 for every supplier, lowest first.

    Score = 100 minus the penalty, never below 0. The penalty adds the
    severity weight of each finding that is open or in review, counting at
    most ``per_rule_cap`` findings per rule, so that a supplier with many
    shipments is not driven to zero by one repeated problem. Resolved and
    waived findings do not count. A supplier with no open finding scores 100.

    The score summarises this tool's open findings. It is not a rating of the
    supplier and says nothing about problems the rules do not look for.

    ``findings`` is the table from ``FindingStore.findings(as_of)``.
    """
    weights = {**SEVERITY_WEIGHTS, **(weights or {})}
    active = findings[findings["status"].isin(["open", "in review"])].copy()
    active["points"] = active["severity"].map(weights).astype(float)
    active["position"] = active.groupby(["supplier_id", "rule_id"]).cumcount()
    penalty = active[active["position"] < per_rule_cap].groupby("supplier_id")["points"].sum()
    counts = active.groupby(["supplier_id", "severity"]).size().unstack(fill_value=0).reindex(columns=SEVERITIES, fill_value=0)
    out = suppliers[["supplier_id", "supplier_name", "country"]].copy()
    out["open_findings"] = out["supplier_id"].map(active.groupby("supplier_id").size()).fillna(0).astype(int)
    for s in SEVERITIES:
        out[s] = out["supplier_id"].map(counts[s]).fillna(0).astype(int) if len(counts) else 0
    overdue = active[active["overdue"]].groupby("supplier_id").size() if "overdue" in active.columns else pd.Series(dtype=int)
    out["overdue"] = out["supplier_id"].map(overdue).fillna(0).astype(int)
    out["score"] = (100 - out["supplier_id"].map(penalty).fillna(0.0)).clip(lower=0).round(1)
    return out.sort_values(["score", "open_findings", "supplier_id"], ascending=[True, False, True]).reset_index(drop=True)


def ageing_table(findings: pd.DataFrame) -> pd.DataFrame:
    """Open and in-review findings counted by severity (rows) and age since first seen (columns).

    Needs ``age_days``, so pass ``FindingStore.findings(as_of)``.
    """
    active = findings[findings["status"].isin(["open", "in review"])]
    band = pd.cut(active["age_days"], [-1, 7, 30, 90, 10**6], labels=AGE_BANDS)
    table = pd.crosstab(active["severity"], band, dropna=False).reindex(index=SEVERITIES, columns=AGE_BANDS, fill_value=0)
    table["total"] = table.sum(axis=1)
    table.columns.name = None
    return table.rename_axis("severity").reset_index()


def rule_summary(findings: pd.DataFrame) -> pd.DataFrame:
    """Findings per rule: open, in review, resolved, waived, overdue, and how many are repeats for the same supplier."""
    if findings.empty:
        return pd.DataFrame(columns=["rule_id", "title", "severity", "open", "in review", "resolved", "waived", "overdue", "repeats"])
    by = findings.groupby(["rule_id", "title", "severity", "status"]).size().unstack(fill_value=0)
    by = by.reindex(columns=["open", "in review", "resolved", "waived"], fill_value=0)
    extra = findings.groupby(["rule_id", "title", "severity"]).agg(
        overdue=("overdue", "sum") if "overdue" in findings.columns else ("recurrence", lambda s: 0),
        repeats=("recurrence", lambda s: int((s > 0).sum())))
    out = by.join(extra).reset_index()
    out["overdue"] = out["overdue"].astype(int)
    return out.sort_values(["open", "in review"], ascending=False).reset_index(drop=True)


def headline(findings: pd.DataFrame, as_of) -> str:
    """One paragraph on where things stand on ``as_of``."""
    active = findings[findings["status"].isin(["open", "in review"])]
    if findings.empty:
        return "No findings have been recorded yet. Run the checks first."
    if active.empty:
        return f"No findings are open on {pd.Timestamp(as_of):%d %B %Y}. {len(findings)} earlier findings are closed."
    counts = active["severity"].value_counts()
    parts = ", ".join(f"{int(counts.get(s, 0))} {s}" for s in SEVERITIES if counts.get(s, 0))
    text = (f"{len(active)} findings are open across {active['supplier_id'].nunique()} suppliers on {pd.Timestamp(as_of):%d %B %Y}: {parts}. "
            f"{int(active['overdue'].sum())} are past their due date and {int((active['owner'] == '').sum())} have no owner.")
    top = active.groupby("title").size().sort_values(ascending=False)
    text += f" The most common is \"{top.index[0]}\" ({int(top.iloc[0])})."
    screening = active[active["category"] == "screening"]
    if len(screening):
        text += f" {len(screening)} potential screening {'match needs' if len(screening) == 1 else 'matches need'} human review."
    return text
