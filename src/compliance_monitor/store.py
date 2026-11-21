"""Persistent findings workflow and audit trail (SQLite).

A *finding* is one rule broken by one record. It is created the first time a
check raises it and then tracked: status, owner, due date, the evidence seen
when it was raised, and when it was last seen.

Every change is written to the *audit trail*: who changed which field of which
finding, from what to what, and when. The trail is only ever appended to.

A *run* is one execution of the checks on one date; its totals feed the trend.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

STATUSES = ["open", "in review", "resolved", "waived"]
ACTIVE = ["open", "in review"]
# Days allowed to deal with a finding, by severity. Placeholders: set them to your own policy.
DEFAULT_DUE_DAYS = {"critical": 2, "high": 7, "medium": 30, "low": 60}
SYSTEM = "system"
AUTO_CLOSED = "Closed automatically on re-check: the record no longer breaks the rule"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    finding_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_key    TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    title          TEXT NOT NULL,
    category       TEXT NOT NULL,
    severity       TEXT NOT NULL,
    record_type    TEXT NOT NULL,
    record_id      TEXT NOT NULL,
    supplier_id    TEXT NOT NULL,
    supplier_name  TEXT DEFAULT '',
    summary        TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    score          REAL,
    remediation    TEXT DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'open',
    owner          TEXT DEFAULT '',
    due_date       TEXT NOT NULL,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    closed_on      TEXT,
    resolution     TEXT DEFAULT '',
    waiver_reason  TEXT DEFAULT '',
    recurrence     INTEGER NOT NULL DEFAULT 0,
    present        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS findings_key ON findings (finding_key);
CREATE TABLE IF NOT EXISTS audit (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id  INTEGER NOT NULL REFERENCES findings(finding_id),
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    note        TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    as_of TEXT PRIMARY KEY, ran_at TEXT NOT NULL, suppliers INTEGER, shipments INTEGER, detected INTEGER, new INTEGER,
    auto_closed INTEGER, open_total INTEGER, open_critical INTEGER, open_high INTEGER, open_medium INTEGER, open_low INTEGER,
    overdue INTEGER, mean_score REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_EDITABLE = {"status", "owner", "due_date"}


def _stamp(at=None) -> str:
    return f"{pd.Timestamp.now() if at is None else pd.Timestamp(at):%Y-%m-%d %H:%M:%S}"


class FindingStore:
    """Findings, their audit trail and the run history in one SQLite file. Use ``":memory:"`` for tests."""

    def __init__(self, path: str | Path = "compliance_monitor.db", due_days: dict | None = None):
        self.path = str(path)
        self.due_days = {**DEFAULT_DUE_DAYS, **(due_days or {})}
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _log(self, finding_id: int, at: str, actor: str, field: str, old, new, note: str = "") -> None:
        self._conn.execute("INSERT INTO audit (finding_id, at, actor, field, old_value, new_value, note) VALUES (?,?,?,?,?,?,?)",
                           (finding_id, at, actor, field, None if old is None else str(old), None if new is None else str(new), note))

    # ------------------------------------------------------------------- sync
    def sync(self, findings: pd.DataFrame, as_of, rules_checked: list[str] | None = None) -> dict:
        """Bring the stored findings in line with the findings of a check run on ``as_of``.

        For each finding raised by the run:

        * already open or in review: its last-seen date, summary and score are refreshed;
        * waived or resolved by a person, and raised by every run since: it
          stays closed. A person's decision covers the violation for as long
          as it persists and is not undone by a re-check;
        * never seen, or closed and then absent from at least one run (the
          record was fixed and has broken the rule again): a new finding is
          opened, due ``due_days[severity]`` days after ``as_of``. Its
          ``recurrence`` is the number of earlier findings for the same rule
          and the same supplier that were first seen before ``as_of``
          (0 means first time).

        Each open or in-review finding that the run did *not* raise, and whose
        rule was among ``rules_checked``, is closed automatically as resolved:
        the record no longer breaks the rule. Findings of rules that were
        switched off are left alone. Note that a record removed from the data
        also stops breaking its rules, so its findings close.

        ``rules_checked`` defaults to ``findings.attrs["rules_checked"]``, or
        failing that to the rules present in ``findings``. Every change is
        written to the audit trail with actor "system".
        Returns counts: detected, new, recurring, still_open, auto_closed.
        """
        as_of = pd.Timestamp(as_of).normalize()
        day, at = f"{as_of:%Y-%m-%d}", f"{as_of:%Y-%m-%d} 00:00:00"
        if rules_checked is None:
            rules_checked = findings.attrs.get("rules_checked", sorted(findings["rule_id"].unique()))
        counts = dict(detected=len(findings), new=0, recurring=0, still_open=0, auto_closed=0)
        with self._conn:
            latest = {}
            for row in self._conn.execute("SELECT finding_id, finding_key, status, present, rule_id FROM findings ORDER BY finding_id"):
                latest[row[1]] = (row[0], row[2], row[3], row[4])
            earlier = {}
            for rule_id, supplier_id, n in self._conn.execute(
                    "SELECT rule_id, supplier_id, COUNT(*) FROM findings WHERE first_seen < ? GROUP BY rule_id, supplier_id", (day,)):
                earlier[(rule_id, supplier_id)] = n
            seen = set()
            for f in findings.to_dict("records"):
                key = f["finding_key"]
                seen.add(key)
                score = None if f.get("score") is None or pd.isna(f.get("score")) else float(f["score"])
                known = latest.get(key)
                if known and known[1] in ACTIVE:
                    self._conn.execute("UPDATE findings SET last_seen = ?, summary = ?, score = ? WHERE finding_id = ?",
                                       (day, f["summary"], score, known[0]))
                    counts["still_open"] += 1
                    continue
                if known and known[2] == 1:  # closed by a person and still present: the decision stands
                    self._conn.execute("UPDATE findings SET last_seen = ? WHERE finding_id = ?", (day, known[0]))
                    continue
                recurrence = int(earlier.get((f["rule_id"], f["supplier_id"]), 0))
                due = as_of + pd.Timedelta(days=int(self.due_days[f["severity"]]))
                cur = self._conn.execute(
                    "INSERT INTO findings (finding_key, rule_id, title, category, severity, record_type, record_id, supplier_id,"
                    " supplier_name, summary, evidence, score, remediation, due_date, first_seen, last_seen, recurrence)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (key, f["rule_id"], f["title"], f["category"], f["severity"], f["record_type"], f["record_id"], f["supplier_id"],
                     f.get("supplier_name", ""), f["summary"], json.dumps(f["evidence"], default=str), score, f.get("remediation", ""),
                     f"{due:%Y-%m-%d}", day, day, recurrence))
                note = f"Raised by rule {f['rule_id']}" + (f"; this supplier has had this finding {recurrence} time(s) before" if recurrence else "")
                self._log(cur.lastrowid, at, SYSTEM, "status", None, "open", note)
                counts["new"] += 1
                counts["recurring"] += recurrence > 0
            checked = set(rules_checked)
            for finding_id, key, rule_id, status in self._conn.execute(
                    "SELECT finding_id, finding_key, rule_id, status FROM findings WHERE status IN ('open', 'in review')").fetchall():
                if key in seen or rule_id not in checked:
                    continue
                self._conn.execute("UPDATE findings SET status = 'resolved', closed_on = ?, resolution = ? WHERE finding_id = ?",
                                   (day, AUTO_CLOSED, finding_id))
                self._log(finding_id, at, SYSTEM, "status", status, "resolved", AUTO_CLOSED)
                counts["auto_closed"] += 1
            for key, (finding_id, _, present, rule_id) in latest.items():
                if rule_id in checked and present != int(key in seen):
                    self._conn.execute("UPDATE findings SET present = ? WHERE finding_id = ?", (int(key in seen), finding_id))
        return counts

    # ----------------------------------------------------------------- update
    def update(self, finding_id: int, actor: str, status: str | None = None, owner: str | None = None, due_date=None,
               note: str = "", waiver_reason: str = "", at=None) -> None:
        """Change the status, owner or due date of a finding, and record who did it.

        ``actor`` is required. Waiving requires ``waiver_reason``. Resolving
        or waiving records the closing date; moving back to open or in review
        clears it. Each changed field becomes one audit entry holding the old
        and new value and the note. A note with no field change is recorded
        as a comment.
        """
        if not actor or not str(actor).strip():
            raise ValueError("A change needs an actor (who is making it)")
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if status == "waived" and not str(waiver_reason).strip():
            raise ValueError("Waiving a finding requires a reason")
        with closing(self._conn.execute("SELECT status, owner, due_date FROM findings WHERE finding_id = ?", (int(finding_id),))) as cur:
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"No finding {finding_id}")
        current = dict(zip(("status", "owner", "due_date"), row))
        changes = {}
        if status is not None and status != current["status"]:
            changes["status"] = status
        if owner is not None and str(owner).strip() != current["owner"]:
            changes["owner"] = str(owner).strip()
        if due_date is not None and f"{pd.Timestamp(due_date):%Y-%m-%d}" != current["due_date"]:
            changes["due_date"] = f"{pd.Timestamp(due_date):%Y-%m-%d}"
        stamp = _stamp(at)
        with self._conn:
            for name, new in changes.items():
                self._conn.execute(f"UPDATE findings SET {name} = ? WHERE finding_id = ?", (new, int(finding_id)))
                detail = f"Waiver reason: {str(waiver_reason).strip()}" + (f". {note}" if note else "") if (name, new) == ("status", "waived") else note
                self._log(int(finding_id), stamp, str(actor).strip(), name, current[name], new, detail)
            if "status" in changes:
                closed = changes["status"] in ("resolved", "waived")
                self._conn.execute("UPDATE findings SET closed_on = ?, resolution = ?, waiver_reason = ? WHERE finding_id = ?",
                                   (stamp[:10] if closed else None, note if changes["status"] == "resolved" else "",
                                    str(waiver_reason).strip() if changes["status"] == "waived" else "", int(finding_id)))
            if not changes and note:
                self._log(int(finding_id), stamp, str(actor).strip(), "comment", None, None, note)

    # ---------------------------------------------------------------- queries
    def findings(self, as_of=None, status: list[str] | str | None = None) -> pd.DataFrame:
        """Stored findings, newest first within severity.

        ``evidence`` is returned as a dict. With ``as_of``: ``age_days``
        (since first seen, up to the closing date for closed findings) and
        ``overdue`` (open or in review and past the due date).
        """
        df = pd.read_sql_query("SELECT * FROM findings ORDER BY finding_id", self._conn)
        for c in ("due_date", "first_seen", "last_seen", "closed_on"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df["evidence"] = [json.loads(e) for e in df["evidence"]]
        df["active"] = df["status"].isin(ACTIVE)
        if as_of is not None:
            as_of = pd.Timestamp(as_of).normalize()
            until = df["closed_on"].where(~df["active"], as_of).fillna(as_of)
            df["age_days"] = (until - df["first_seen"]).dt.days.clip(lower=0)
            df["overdue"] = df["active"] & (df["due_date"] < as_of)
        if status is not None:
            df = df[df["status"].isin([status] if isinstance(status, str) else list(status))]
        rank = df["severity"].map({"critical": 0, "high": 1, "medium": 2, "low": 3})
        return df.assign(_rank=rank).sort_values(["_rank", "finding_id"], ascending=[True, False]).drop(columns="_rank").reset_index(drop=True)

    def finding(self, finding_id: int) -> pd.Series:
        df = self.findings()
        hit = df[df["finding_id"] == int(finding_id)]
        if hit.empty:
            raise KeyError(f"No finding {finding_id}")
        return hit.iloc[0]

    def audit_trail(self, finding_id: int | None = None) -> pd.DataFrame:
        """The audit trail, oldest first; for one finding or for all."""
        sql = "SELECT a.*, f.rule_id, f.record_id, f.supplier_id FROM audit a JOIN findings f USING (finding_id)"
        if finding_id is not None:
            return pd.read_sql_query(sql + " WHERE a.finding_id = ? ORDER BY a.audit_id", self._conn, params=(int(finding_id),))
        return pd.read_sql_query(sql + " ORDER BY a.audit_id", self._conn)

    # ------------------------------------------------------------------- runs
    def record_run(self, as_of, **totals) -> None:
        """Store the totals of one check run. A second run on the same date replaces the first."""
        columns = ["suppliers", "shipments", "detected", "new", "auto_closed", "open_total", "open_critical", "open_high", "open_medium",
                   "open_low", "overdue", "mean_score"]
        with self._conn:
            self._conn.execute(f"INSERT OR REPLACE INTO runs (as_of, ran_at, {', '.join(columns)}) VALUES ({', '.join('?' * 14)})",
                               (f"{pd.Timestamp(as_of):%Y-%m-%d}", _stamp(), *[totals.get(c) for c in columns]))

    def runs(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY as_of", self._conn, parse_dates=["as_of"])

    # ------------------------------------------------------------------- meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with closing(self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,))) as cur:
            row = cur.fetchone()
        return default if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    def reset(self) -> None:
        """Delete every finding, audit entry, run and stored setting."""
        with self._conn:
            for table in ("audit", "findings", "runs", "meta"):
                self._conn.execute(f"DELETE FROM {table}")
