"""The daily run: check the records, update the stored findings, record the totals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Records, SampleWorld
from .rules import RuleSet, default_rules, run_checks
from .scores import supplier_scores
from .store import FindingStore

SIMULATED_REVIEWERS = ["Analyst A", "Analyst B", "Analyst C"]


def daily_check(records: Records, store: FindingStore, rules: RuleSet | None = None, as_of=None,
                screening_list: pd.DataFrame | None = None) -> dict:
    """Run the enabled rules on ``as_of`` and bring the findings store up to date.

    New violations become open findings, findings whose record has been fixed
    are closed automatically, and the day's totals are stored for the trend
    (see ``FindingStore.sync``). Runs are meant to be made in date order.
    Returns the counts for the run and the findings that were raised.
    """
    rules = default_rules() if rules is None else rules
    findings = run_checks(records, rules, as_of, screening_list)
    as_of = findings.attrs["as_of"]
    counts = store.sync(findings, as_of, findings.attrs["rules_checked"])
    stored = store.findings(as_of)
    active = stored[stored["active"]]
    by = active["severity"].value_counts()
    scores = supplier_scores(stored, records.suppliers)
    totals = dict(suppliers=len(records.suppliers), shipments=len(records.shipments), detected=counts["detected"], new=counts["new"],
                  auto_closed=counts["auto_closed"], open_total=len(active), open_critical=int(by.get("critical", 0)),
                  open_high=int(by.get("high", 0)), open_medium=int(by.get("medium", 0)), open_low=int(by.get("low", 0)),
                  overdue=int(active["overdue"].sum()), mean_score=float(scores["score"].mean()) if len(scores) else None)
    store.record_run(as_of, **totals)
    return {**counts, **totals, "as_of": as_of, "findings": findings}


def replay_sample(world: SampleWorld, store: FindingStore, rules: RuleSet | None = None, weeks: int = 12, step_days: int = 7,
                  simulate_reviewers: bool = True, seed: int = 0) -> pd.DataFrame:
    """Replay the simulated records into a findings store, one check every ``step_days`` days, ending on the last simulated day.

    This exists to show the workflow and the trend on the sample data. Each
    step checks the records as they looked on that date, so new shipments
    arrive, certificates lapse and simulated fixes take effect. With
    ``simulate_reviewers`` a scripted reviewer also acts between runs: about
    half of the unowned findings older than two days are assigned to
    "Analyst A", "B" or "C" and moved to in review, about a third of the
    findings in review for a week are resolved with a note, and a few are
    waived with a reason. Potential screening matches are assigned but never
    closed by the script: they wait for a person. These actions are made up,
    like the data.
    Returns the run history.
    """
    rng = np.random.default_rng(seed)
    for k in range(weeks, -1, -1):
        as_of = world.as_of - pd.Timedelta(days=k * step_days)
        daily_check(world.records(as_of), store, rules, as_of, world.screening_list)
        if not simulate_reviewers or k == 0:
            continue
        at = as_of + pd.Timedelta(hours=10)
        current = store.findings(as_of)
        for _, f in current[current["status"] == "in review"].iterrows():
            if (as_of - f["first_seen"]).days < 7 or f["category"] == "screening":
                continue
            draw = rng.random()
            if draw < 0.30:
                store.update(f["finding_id"], f["owner"], status="resolved", note="Reviewed with the supplier and documented (simulated action)", at=at)
            elif draw < 0.34:
                store.update(f["finding_id"], "Compliance lead", status="waived",
                             waiver_reason="Accepted for this record after review (simulated action)", at=at)
        for _, f in current[(current["status"] == "open") & (current["owner"] == "")].iterrows():
            if (as_of - f["first_seen"]).days >= 2 and rng.random() < 0.5:
                store.update(f["finding_id"], "Compliance lead", status="in review", owner=str(rng.choice(SIMULATED_REVIEWERS)),
                             note="Assigned at the weekly review (simulated action)", at=at)
    return store.runs()
