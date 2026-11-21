"""Name screening: fuzzy matching of supplier names against a screening list.

The output is a list of *potential matches for a person to review*, each with a
similarity score. It is never a determination that a supplier is, or is not,
the party on the list. Similar names often belong to unrelated companies, and
a listed party can trade under a name this comparison does not catch.

The package ships only a small fictional list for the demo. For real use, load
the list or lists that apply to you with ``load_screening_list``.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Words describing a company's legal form. They are dropped before comparing, so "Example Ltd" and "Example LLC" compare as "example".
LEGAL_FORMS = {"ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "gmbh", "sa", "srl", "bv",
               "plc", "pte", "pty", "ag", "ab", "nv", "kk", "sarl", "spa", "lp", "llp", "the"}
POTENTIAL_MATCH = "Potential match: needs human review"


def normalise(name) -> str:
    """Reduce a name to comparable form.

    Accents are removed, letters are lower-cased, "&" becomes "and",
    punctuation becomes a space (so "A-B" is two words and "S.A." is "s a",
    which is then joined back to "sa"), and legal-form words such as "ltd" or
    "inc" are dropped.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower().replace("&", " and ")
    text = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", text)  # "s.a." -> "sa."
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(t for t in tokens if t not in LEGAL_FORMS)


@lru_cache(maxsize=200_000)
def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def token_similarity(a: str, b: str) -> float:
    """How well the words of two normalised names pair up, from 0 to 1, ignoring word order.

    Each word of one name is matched to its most similar word in the other
    (character ratio of the two words). The result is the average of those
    best matches, taken in both directions and averaged.
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    forward = sum(max(_ratio(t, u) for u in tb) for t in ta) / len(ta)
    backward = sum(max(_ratio(u, t) for t in ta) for u in tb) / len(tb)
    return (forward + backward) / 2


def character_similarity(a: str, b: str) -> float:
    """Character ratio of two normalised names after sorting their words alphabetically, from 0 to 1.

    The ratio is twice the number of matching characters divided by the total
    length of both strings (``difflib.SequenceMatcher``).
    """
    return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def name_similarity(name: str, other: str) -> float:
    """Similarity of two names from 0 to 1: the mean of the token and character similarities of their normalised forms."""
    a, b = normalise(name), normalise(other)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 0.5 * token_similarity(a, b) + 0.5 * character_similarity(a, b)


def load_screening_list(path: str | Path) -> pd.DataFrame:
    """Read a screening list from a CSV file with a ``name`` column.

    Optional columns ``list_id`` and ``source`` are kept and shown with each
    potential match. Blank names are dropped.
    """
    return validate_screening_list(pd.read_csv(path, dtype=str))


def validate_screening_list(df: pd.DataFrame) -> pd.DataFrame:
    if "name" not in df.columns:
        raise ValueError("A screening list needs a 'name' column")
    out = df.copy()
    out["name"] = out["name"].astype(object).where(out["name"].notna(), "").astype(str).str.strip()
    out = out[out["name"] != ""].reset_index(drop=True)
    for c, default in (("list_id", None), ("source", "User-supplied list")):
        if c not in out.columns:
            out[c] = [f"L{i + 1:05d}" for i in range(len(out))] if c == "list_id" else default
        out[c] = out[c].astype(object).where(out[c].notna(), "").astype(str)
    return out[["list_id", "name", "source"] + [c for c in out.columns if c not in ("list_id", "name", "source")]]


def screen_names(names: pd.DataFrame, screening_list: pd.DataFrame, threshold: float = 0.85, id_column: str = "supplier_id",
                 name_column: str = "supplier_name") -> pd.DataFrame:
    """Compare every name with every list entry and return the pairs scoring at least ``threshold``.

    One row per (name, list entry) pair, highest score first. Every row is a
    potential match for a person to review, not a finding of fact. Every name
    is compared with every entry, so the time taken grows with the product of
    the two table sizes.
    """
    columns = [id_column, name_column, "list_id", "list_name", "source", "score", "token_similarity", "character_similarity", "status"]
    if names.empty or screening_list is None or screening_list.empty:
        return pd.DataFrame(columns=columns)
    entries = [(r["list_id"], r["name"], r["source"], normalise(r["name"])) for _, r in validate_screening_list(screening_list).iterrows()]
    rows = []
    for rid, name in zip(names[id_column], names[name_column]):
        a = normalise(name)
        if not a:
            continue
        for list_id, list_name, source, b in entries:
            if not b:
                continue
            tok, char = (1.0, 1.0) if a == b else (token_similarity(a, b), character_similarity(a, b))
            score = 0.5 * tok + 0.5 * char
            if score >= threshold:
                rows.append((rid, name, list_id, list_name, source, round(score, 4), round(tok, 4), round(char, 4), POTENTIAL_MATCH))
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["score", id_column], ascending=[False, True]).reset_index(drop=True)
