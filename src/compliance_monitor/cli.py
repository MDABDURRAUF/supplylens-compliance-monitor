"""Command line.

    compliance-monitor sample --out data
    compliance-monitor check --data data --as-of 2026-06-30 --out findings.csv
    compliance-monitor findings --status open --severity critical high
    compliance-monitor update --id 12 --actor "Analyst A" --status "in review" --owner "Analyst A"
    compliance-monitor profile --data data --supplier SUP-001 --out profile.html
    compliance-monitor report --data data --out audit_summary.html
    compliance-monitor evaluate
    compliance-monitor replay --weeks 12

The findings database is ``compliance_monitor.db`` in the current folder unless
``--db`` or the environment variable ``COMPLIANCE_MONITOR_DB`` says otherwise.
Decision support, not legal advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import __version__, evaluate
from .data import SchemaError, generate_sample, load_records, save_records
from .monitor import daily_check, replay_sample
from .report import build_audit_report, build_supplier_profile, write_html
from .rules import SEVERITIES, RuleSet, default_rules
from .scores import headline
from .screening import load_screening_list
from .store import STATUSES, FindingStore

NOTICE = "Decision support, not legal advice. Parameters are your settings, not legal requirements."


def _show(df: pd.DataFrame, formatters: dict | None = None) -> None:
    with pd.option_context("display.width", 240, "display.max_columns", None, "display.max_colwidth", 70):
        print(df.to_string(index=False, formatters=formatters or {}))


def _db(args) -> str:
    return args.db or os.environ.get("COMPLIANCE_MONITOR_DB", "compliance_monitor.db")


def _rules(args) -> RuleSet:
    return RuleSet.load(args.rules) if getattr(args, "rules", None) else default_rules()


def _as_of(args, store: FindingStore) -> pd.Timestamp:
    if args.as_of:
        return pd.Timestamp(args.as_of).normalize()
    runs = store.runs()
    if runs.empty:
        raise ValueError("No check has been run on this database yet; run 'check' first or pass --as-of")
    return runs["as_of"].max()


def _export(findings: pd.DataFrame) -> pd.DataFrame:
    out = findings.drop(columns=[c for c in ("active", "present") if c in findings.columns]).copy()
    out["evidence"] = out["evidence"].map(json.dumps)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compliance-monitor", description="Supply chain compliance monitoring. " + NOTICE)
    p.add_argument("--version", action="version", version=f"compliance-monitor {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def db(parser):
        parser.add_argument("--db", help="findings database (default: COMPLIANCE_MONITOR_DB or compliance_monitor.db)")

    s = sub.add_parser("sample", help="write simulated records, the fictional screening list, the ground truth and the default rules to a folder")
    s.add_argument("--out", default="data")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--shipments", type=int, default=5000, help="number of simulated shipments (default 5000)")
    s.add_argument("--as-of", help="write the records as they looked on this date (default: the last simulated day)")

    c = sub.add_parser("check", help="run the rules on the records in a folder and update the findings database")
    c.add_argument("--data", required=True, help="folder with suppliers.csv, certificates.csv, shipments.csv, products.csv")
    c.add_argument("--as-of", help="check date, YYYY-MM-DD (default: the latest shipment entry date)")
    c.add_argument("--rules", help="rules JSON file (default: the built-in rules)")
    c.add_argument("--screening-list", help="CSV with a 'name' column (default: screening_list.csv in the data folder, if present)")
    c.add_argument("--out", help="write the findings raised by this run to this CSV file")
    db(c)

    f = sub.add_parser("findings", help="list stored findings")
    f.add_argument("--status", nargs="*", choices=STATUSES, default=["open", "in review"])
    f.add_argument("--severity", nargs="*", choices=SEVERITIES)
    f.add_argument("--supplier", help="only this supplier_id")
    f.add_argument("--as-of", help="date used for age and overdue (default: the latest check date)")
    f.add_argument("--out", help="write the list to this CSV file")
    db(f)

    u = sub.add_parser("update", help="change the status, owner or due date of a finding")
    u.add_argument("--id", type=int, required=True)
    u.add_argument("--actor", required=True, help="who is making the change (recorded in the audit trail)")
    u.add_argument("--status", choices=STATUSES)
    u.add_argument("--owner")
    u.add_argument("--due", help="new due date, YYYY-MM-DD")
    u.add_argument("--note", default="")
    u.add_argument("--reason", default="", help="required when waiving")
    db(u)

    pr = sub.add_parser("profile", help="write one supplier's compliance profile (HTML)")
    pr.add_argument("--data", required=True)
    pr.add_argument("--supplier", required=True, help="supplier_id")
    pr.add_argument("--as-of")
    pr.add_argument("--out", default="supplier_profile.html")
    db(pr)

    r = sub.add_parser("report", help="write the audit summary report (HTML)")
    r.add_argument("--data", required=True)
    r.add_argument("--rules")
    r.add_argument("--as-of")
    r.add_argument("--out", default="audit_summary.html")
    db(r)

    e = sub.add_parser("evaluate", help="measure every rule against the violations injected into the simulated data")
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--out", help="write the detection table to this CSV file")

    rp = sub.add_parser("replay", help="fill a findings database by replaying the simulated records week by week (demo)")
    rp.add_argument("--weeks", type=int, default=12)
    rp.add_argument("--seed", type=int, default=7)
    rp.add_argument("--data-out", help="also write the simulated records of the last day to this folder")
    db(rp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pc = lambda v: "" if pd.isna(v) else f"{v:.1%}"  # noqa: E731
    day = lambda d: "" if pd.isna(d) else f"{pd.Timestamp(d):%Y-%m-%d}"  # noqa: E731
    store = None
    try:
        if args.command == "sample":
            world = generate_sample(seed=args.seed, n_shipments=args.shipments)
            records = world.records(args.as_of)
            folder = save_records(records, args.out)
            world.screening_list.to_csv(folder / "screening_list.csv", index=False)
            world.ground_truth(args.as_of).to_csv(folder / "ground_truth.csv", index=False)
            default_rules().save(folder / "rules.json")
            n = records.counts()
            print(f"Wrote {n['suppliers']} suppliers, {n['certificates']} certificates, {n['shipments']:,} shipments and {n['products']} products "
                  f"to {folder}, with a fictional screening list, the ground truth and rules.json. Simulated data, not real records.")
        elif args.command == "check":
            records = load_records(args.data)
            listed = args.screening_list or (Path(args.data) / "screening_list.csv")
            screening = load_screening_list(listed) if Path(listed).exists() else None
            if args.screening_list and screening is None:
                raise FileNotFoundError(f"No screening list at {args.screening_list}")
            store = FindingStore(_db(args))
            store.set_meta("simulated", "1" if (Path(args.data) / "ground_truth.csv").exists() else "0")
            result = daily_check(records, store, _rules(args), args.as_of, screening)
            print(f"Checked on {result['as_of']:%Y-%m-%d}: {result['detected']} findings raised; {result['new']} new "
                  f"({result['recurring']} repeat for the same supplier), {result['still_open']} already open, {result['auto_closed']} closed "
                  "because the record was fixed.")
            if screening is None:
                print("No screening list was loaded, so no names were screened.")
            print(headline(store.findings(result["as_of"]), result["as_of"]))
            if args.out:
                out = result["findings"].copy()
                out["evidence"] = out["evidence"].map(json.dumps)
                out.to_csv(args.out, index=False)
                print(f"Wrote {args.out}")
            print(NOTICE)
        elif args.command == "findings":
            store = FindingStore(_db(args))
            found = store.findings(_as_of(args, store), status=args.status)
            if args.severity:
                found = found[found["severity"].isin(args.severity)]
            if args.supplier:
                found = found[found["supplier_id"] == args.supplier]
            print(f"{len(found)} finding(s).")
            if len(found):
                _show(found.head(40)[["finding_id", "severity", "status", "rule_id", "supplier_id", "record_id", "summary", "owner", "due_date", "overdue"]],
                      {"due_date": day})
            if args.out:
                _export(found).to_csv(args.out, index=False)
                print(f"Wrote {args.out}")
        elif args.command == "update":
            store = FindingStore(_db(args))
            store.update(args.id, args.actor, status=args.status, owner=args.owner, due_date=args.due, note=args.note, waiver_reason=args.reason)
            _show(store.audit_trail(args.id)[["at", "actor", "field", "old_value", "new_value", "note"]])
        elif args.command in ("profile", "report"):
            records = load_records(args.data)
            store = FindingStore(_db(args))
            simulated = store.get_meta("simulated", "0") == "1"
            as_of = _as_of(args, store)
            if args.command == "profile":
                html = build_supplier_profile(store, records, args.supplier, as_of, simulated)
            else:
                html = build_audit_report(store, records, _rules(args), as_of, simulated)
            print(f"Wrote {write_html(html, args.out)}")
        elif args.command == "evaluate":
            world = generate_sample(seed=args.seed)
            table = evaluate.evaluate_sample(world)
            n = world.records(apply_fixes=False).counts()
            print(f"Simulated data: {n['suppliers']} suppliers, {n['certificates']} certificates, {n['shipments']:,} shipments, {n['products']} "
                  f"products, {int(table['true_violations'].iloc[-1])} injected violations. These numbers describe the simulation only.")
            _show(table.drop(columns=["category"]), {"precision": pc, "recall": pc, "false_positives_per_1000": "{:.1f}".format})
            print("\nUnit value check, by method:")
            _show(evaluate.outlier_methods(world), {"precision": pc, "recall": pc, "false_positives_per_1000": "{:.1f}".format})
            if args.out:
                table.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
        elif args.command == "replay":
            world = generate_sample(seed=args.seed)
            store = FindingStore(_db(args))
            if len(store.runs()):
                raise ValueError(f"{store.path} already holds check runs; replay needs an empty database")
            store.set_meta("simulated", "1")
            runs = replay_sample(world, store, weeks=args.weeks)
            _show(runs[["as_of", "shipments", "new", "auto_closed", "open_total", "open_critical", "open_high", "overdue", "mean_score"]],
                  {"as_of": day, "mean_score": "{:.1f}".format})
            print(headline(store.findings(world.as_of), world.as_of))
            if args.data_out:
                folder = save_records(world.records(), args.data_out)
                world.screening_list.to_csv(folder / "screening_list.csv", index=False)
                world.ground_truth().to_csv(folder / "ground_truth.csv", index=False)
                print(f"Wrote the simulated records to {folder}")
    except (SchemaError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
