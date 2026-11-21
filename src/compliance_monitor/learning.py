"""The two checks that learn from the records themselves.

1. ``unit_value_outliers``: is a shipment's declared value per unit far from
   what other shipments under the same tariff code declare?
2. ``chapter_fit``: does the goods description read like the other
   descriptions filed under the same tariff chapter?

Both compare a record with its peers. Neither knows what the right value or
the right code is, so both produce candidates for a person to look at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def unit_value_outliers(shipments: pd.DataFrame, threshold: float = 4.5, min_group: int = 10, method: str = "robust",
                        contamination: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Score each shipment's declared unit value against other shipments with the same tariff code.

    Unit value is declared value divided by quantity. Within each tariff code
    with at least ``min_group`` usable shipments:

    ``method="robust"`` (default)
        The score is the distance of the log unit value from the group median,
        in robust standard deviations (1.4826 times the median absolute
        deviation). A shipment is an outlier when the absolute score exceeds
        ``threshold``. A score of +5 means far above its peers, -5 far below.

    ``method="isolation_forest"``
        One IsolationForest per tariff code on the log unit value; the
        ``contamination`` share with the most isolated values in each group is
        flagged. By construction it flags about that share of every group
        whether or not anything is wrong. The robust score is still reported.

    Shipments with no positive value or quantity, and tariff codes with fewer
    than ``min_group`` shipments, are not scored (``scored`` is False).

    Returns one row per shipment, in the same order: ``unit_value``,
    ``group_median``, ``group_size``, ``score``, ``scored``, ``outlier``.
    """
    if method not in ("robust", "isolation_forest"):
        raise ValueError("method must be 'robust' or 'isolation_forest'")
    value = pd.to_numeric(shipments["declared_value_usd"], errors="coerce").to_numpy(dtype=float)
    qty = pd.to_numeric(shipments["quantity"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(value) & np.isfinite(qty) & (value > 0) & (qty > 0)
    unit = np.where(usable, value / np.where(usable, qty, 1.0), np.nan)
    out = pd.DataFrame({"unit_value": unit, "group_median": np.nan, "group_size": 0, "score": np.nan, "scored": False, "outlier": False},
                       index=shipments.index)
    codes = shipments["tariff_code"].to_numpy()
    for _, idx in pd.Series(np.arange(len(shipments)))[usable].groupby(codes[usable]).indices.items():
        rows = np.flatnonzero(usable)[idx]
        out.iloc[rows, out.columns.get_loc("group_size")] = len(rows)
        if len(rows) < min_group:
            continue
        logs = np.log(unit[rows])
        median = np.median(logs)
        spread = 1.4826 * np.median(np.abs(logs - median))
        spread = spread if spread > 1e-9 else max(float(np.std(logs)), 1e-9)
        score = (logs - median) / spread
        if method == "robust":
            flagged = np.abs(score) > threshold
        else:
            forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=seed)
            flagged = forest.fit_predict(logs.reshape(-1, 1)) == -1
        out.iloc[rows, out.columns.get_loc("group_median")] = float(np.exp(median))
        out.iloc[rows, out.columns.get_loc("score")] = score
        out.iloc[rows, out.columns.get_loc("scored")] = True
        out.iloc[rows, out.columns.get_loc("outlier")] = flagged
    return out


def declared_chapter(tariff_code: pd.Series, chapter_digits: int = 2) -> pd.Series:
    """The chapter of each tariff code: its first ``chapter_digits`` characters when they are all digits, otherwise blank."""
    lead = tariff_code.astype(object).where(tariff_code.notna(), "").astype(str).str.strip().str[:chapter_digits]
    return lead.where(lead.str.fullmatch(rf"\d{{{chapter_digits}}}"), "")


def chapter_fit(shipments: pd.DataFrame, chapter_digits: int = 2, max_declared_probability: float = 0.05,
                min_predicted_probability: float = 0.5, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Predict each shipment's tariff chapter from its goods description and compare it with the declared chapter.

    The classifier is TF-IDF on the words of the description followed by logistic
    regression. It is trained on the shipments themselves, using the declared
    chapter as the label, so it learns how each chapter is usually described
    *in this data*; it knows nothing about any tariff schedule. Predictions
    are cross-fitted: the shipments are split into ``folds`` parts and each
    part is predicted by a model trained on the others, so no shipment is
    judged by a model that saw its own label.

    A shipment is flagged when the model gives its declared chapter a
    probability below ``max_declared_probability`` *and* gives another chapter
    at least ``min_predicted_probability``.

    Shipments with a blank description, a code whose chapter cannot be read,
    or a chapter with fewer than ``folds`` shipments are not scored. Nothing
    is scored when fewer than two chapters remain.

    Returns one row per shipment, in the same order: ``declared_chapter``,
    ``predicted_chapter``, ``p_declared``, ``p_predicted``, ``scored``,
    ``unlikely``.
    """
    chapter = declared_chapter(shipments["tariff_code"], chapter_digits)
    text = shipments["goods_description"].astype(object).where(shipments["goods_description"].notna(), "").astype(str).str.strip()
    out = pd.DataFrame({"declared_chapter": chapter, "predicted_chapter": "", "p_declared": np.nan, "p_predicted": np.nan,
                        "scored": False, "unlikely": False}, index=shipments.index)
    sizes = chapter.value_counts()
    usable = ((chapter != "") & (text != "") & chapter.map(sizes).ge(folds)).to_numpy()
    if usable.sum() == 0 or chapter[usable].nunique() < 2:
        return out
    x, y = text[usable].to_numpy(), chapter[usable].to_numpy()
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(C=10.0, max_iter=1000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    position = pd.Series(np.arange(len(classes)), index=classes)
    p_declared = proba[np.arange(len(y)), position[y].to_numpy()]
    best = proba.argmax(axis=1)
    rows = np.flatnonzero(usable)
    out.iloc[rows, out.columns.get_loc("predicted_chapter")] = classes[best]
    out.iloc[rows, out.columns.get_loc("p_declared")] = p_declared
    out.iloc[rows, out.columns.get_loc("p_predicted")] = proba.max(axis=1)
    out.iloc[rows, out.columns.get_loc("scored")] = True
    out.iloc[rows, out.columns.get_loc("unlikely")] = (p_declared < max_declared_probability) & (proba.max(axis=1) >= min_predicted_probability)
    return out
