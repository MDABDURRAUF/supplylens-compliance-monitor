"""How well the rules and models find the violations injected into the simulated data.

Three measures are reported for every rule:

``precision``                 of the records a rule raised, the share that really had the injected violation
``recall``                    of the records with the injected violation, the share the rule raised
``false_positives_per_1000``  records raised without the violation, per 1,000 records the rule looked at

These numbers describe the simulation. Rules that test a recorded fact
(a date, a flag, a format) recover what was injected exactly, which only
shows that the plumbing works. The numbers that carry information are those of
the checks that can be wrong: origin, unit value, description and screening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _STEM_A, _STEM_B, SAMPLE_SCREENING_NAMES, Records, SampleWorld, generate_sample
from .learning import chapter_fit, unit_value_outliers
from .rules import RECORD_TABLE, RuleSet, default_rules, run_checks
from .screening import name_similarity, screen_names


def _measures(flagged: set, truth: set, checked: int) -> dict:
    tp = len(flagged & truth)
    fp, fn = len(flagged) - tp, len(truth) - tp
    return {"records_checked": checked, "true_violations": len(truth), "flagged": len(flagged), "true_positives": tp,
            "false_positives": fp, "missed": fn, "precision": tp / len(flagged) if flagged else np.nan,
            "recall": tp / len(truth) if truth else np.nan, "false_positives_per_1000": 1000 * fp / checked if checked else np.nan}


def detection_table(findings: pd.DataFrame, truth: pd.DataFrame, records: Records, rules: RuleSet | None = None) -> pd.DataFrame:
    """Precision, recall and false positives per 1,000 records for every enabled rule, plus a row for all rules together.

    A finding is a true positive when the ground truth holds a violation for
    the same rule and the same record. ``records_checked`` is the number of
    records of the kind the rule applies to.
    """
    rules = default_rules() if rules is None else rules
    counts = records.counts()
    rows, all_flagged, all_truth, all_checked = [], set(), set(), 0
    for rule in rules.enabled():
        flagged = set(findings.loc[findings["rule_id"] == rule.id, "record_id"])
        actual = set(truth.loc[truth["rule_id"] == rule.id, "record_id"])
        checked = counts[RECORD_TABLE[rule.applies_to]]
        rows.append({"rule_id": rule.id, "title": rule.title, "category": rule.category, **_measures(flagged, actual, checked)})
        all_flagged |= {(rule.id, r) for r in flagged}
        all_truth |= {(rule.id, r) for r in actual}
        all_checked += checked
    rows.append({"rule_id": "ALL", "title": "All rules together", "category": "", **_measures(all_flagged, all_truth, all_checked)})
    return pd.DataFrame(rows)


def evaluate_sample(world: SampleWorld | None = None, rules: RuleSet | None = None) -> pd.DataFrame:
    """Run the rules on the simulated records with every injected violation still in place, and score them against the ground truth."""
    world = generate_sample() if world is None else world
    records = world.records(apply_fixes=False)
    findings = run_checks(records, rules, world.as_of, world.screening_list)
    return detection_table(findings, world.ground_truth(apply_fixes=False), records, rules)


def _truth_ids(world: SampleWorld, kind: str) -> set:
    truth = world.ground_truth(apply_fixes=False)
    return set(truth.loc[truth["kind"] == kind, "record_id"])


def outlier_methods(world: SampleWorld, thresholds=(3.5, 4.5, 6.0), contamination: float = 0.02) -> pd.DataFrame:
    """The unit-value check at several robust-score thresholds and with IsolationForest, against the injected implausible values."""
    sh = world.records(apply_fixes=False).shipments
    actual = _truth_ids(world, "implausible_unit_value")
    rows = []
    for t in thresholds:
        res = unit_value_outliers(sh, threshold=t)
        rows.append({"method": f"robust score above {t:g}", **_measures(set(sh.loc[res["outlier"].to_numpy(), "entry_id"]), actual, len(sh))})
    res = unit_value_outliers(sh, method="isolation_forest", contamination=contamination)
    rows.append({"method": f"IsolationForest, contamination {contamination:g}",
                 **_measures(set(sh.loc[res["outlier"].to_numpy(), "entry_id"]), actual, len(sh))})
    return pd.DataFrame(rows)


def classifier_summary(world: SampleWorld) -> dict:
    """Cross-fitted accuracy of the chapter classifier on shipments whose description was not tampered with."""
    sh = world.records(apply_fixes=False).shipments
    fit = chapter_fit(sh)
    clean = fit["scored"].to_numpy() & ~sh["entry_id"].isin(_truth_ids(world, "description_mismatch")).to_numpy()
    return {"shipments_scored": int(fit["scored"].sum()), "chapters": int(fit.loc[fit["scored"], "declared_chapter"].nunique()),
            "accuracy_on_untampered": float((fit["predicted_chapter"] == fit["declared_chapter"])[clean].mean())}


def classifier_thresholds(world: SampleWorld, thresholds=(0.01, 0.05, 0.10, 0.20, 0.40)) -> pd.DataFrame:
    """The description check at several values of ``max_declared_probability``, against the injected mismatches."""
    sh = world.records(apply_fixes=False).shipments
    actual = _truth_ids(world, "description_mismatch")
    fit = chapter_fit(sh)
    rows = []
    for t in thresholds:
        hit = (fit["scored"] & (fit["p_declared"] < t) & (fit["p_predicted"] >= 0.5)).to_numpy()
        rows.append({"max_declared_probability": t, **_measures(set(sh.loc[hit, "entry_id"]), actual, len(sh))})
    return pd.DataFrame(rows)


def screening_thresholds(world: SampleWorld, thresholds=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)) -> pd.DataFrame:
    """Name screening of the simulated suppliers at several thresholds, against the suppliers injected as resembling a list entry."""
    suppliers = world.base.suppliers
    actual = _truth_ids(world, "name_resembles_screening_list")
    matches = screen_names(suppliers, world.screening_list, threshold=min(thresholds))
    return pd.DataFrame([{"threshold": t, **_measures(set(matches.loc[matches["score"] >= t, "supplier_id"]), actual, len(suppliers))}
                         for t in thresholds])


_PERTURBATIONS = ["legal form added", "word order changed", "one typo", "two typos", "punctuation and accents", "one word abbreviated",
                  "one word dropped"]


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    how = int(rng.integers(0, 3))
    if how == 0:
        return word[:i] + word[i + 1:]                              # a letter dropped
    if how == 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]      # two letters swapped
    return word[:i] + "abcdefghijklmnopqrstuvwxyz"[int(rng.integers(0, 26))] + word[i + 1:]  # a letter replaced


def perturb(name: str, how: str, rng: np.random.Generator) -> str:
    """A variant of a name of the given kind, as a careless or evasive spelling might produce."""
    words = name.split()
    k = int(rng.integers(0, len(words)))
    if how == "legal form added":
        return f"{name} {rng.choice(['Ltd', 'LLC', 'Inc', 'GmbH', 'Co'])}"
    if how == "word order changed":
        return " ".join(words[1:] + words[:1])
    if how == "one typo":
        words[k] = _typo(words[k], rng)
    elif how == "two typos":
        words[k] = _typo(words[k], rng)
        j = (k + 1) % len(words)
        words[j] = _typo(words[j], rng)
    elif how == "punctuation and accents":
        return "-".join(words[:2]).replace("e", "é", 1) + " " + " ".join(words[2:]) + "."
    elif how == "one word abbreviated":
        words[k] = words[k][0] + "."
    elif how == "one word dropped":
        del words[k]
    return " ".join(words)


def screening_benchmark(threshold: float = 0.85, per_kind: int = 60, unrelated: int = 400, seed: int = 0) -> pd.DataFrame:
    """A larger test of the name matcher than the handful of simulated suppliers allows.

    For each kind of alteration, ``per_kind`` variants of entries on the
    fictional list are generated and compared with the original entry;
    ``caught`` is the share scoring at least ``threshold``. The last row
    compares ``unrelated`` invented company names with every list entry and
    reports the share wrongly scoring at or above the threshold. The invented
    names share business words ("Trading", "Group") with the list but nothing
    else, which is an easier test than real lists of similar names.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for how in _PERTURBATIONS:
        scores = []
        for _ in range(per_kind):
            original = str(rng.choice(SAMPLE_SCREENING_NAMES))
            scores.append(name_similarity(perturb(original, how, rng), original))
        rows.append({"test": how, "names": per_kind, "caught": float(np.mean(np.array(scores) >= threshold)),
                     "average_score": float(np.mean(scores)), "lowest_score": float(np.min(scores))})
    last_words = sorted({n.split()[-1] for n in SAMPLE_SCREENING_NAMES})
    best = []
    for _ in range(unrelated):
        name = f"{rng.choice(_STEM_A)} {rng.choice(_STEM_B)} {rng.choice(last_words)}"
        best.append(max(name_similarity(name, entry) for entry in SAMPLE_SCREENING_NAMES))
    rows.append({"test": "unrelated invented names (wrongly raised)", "names": unrelated, "caught": float(np.mean(np.array(best) >= threshold)),
                 "average_score": float(np.mean(best)), "lowest_score": float(np.min(best))})
    return pd.DataFrame(rows)
