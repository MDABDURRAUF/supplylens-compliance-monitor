"""SupplyLens Compliance Monitor dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import compliance_monitor as cm
from compliance_monitor import evaluate as ev
from compliance_monitor.data import (CERTIFICATE_COLUMNS, CERTIFICATE_OPTIONAL, PRODUCT_COLUMNS, PRODUCT_OPTIONAL, SHIPMENT_COLUMNS,
                                     SHIPMENT_OPTIONAL, SUPPLIER_COLUMNS, SUPPLIER_OPTIONAL, TABLES)
from compliance_monitor.rules import CONFIRM
from compliance_monitor.scores import AGE_BANDS

st.set_page_config(page_title="Compliance Monitor", layout="wide")

RED, DARK, AMBER, GREEN, BLUE, GREY = "#d1495b", "#7a1f2b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
SEVERITY_COLORS = {"critical": DARK, "high": RED, "medium": AMBER, "low": GREY}
STATUS_COLORS = {"open": RED, "in review": BLUE, "resolved": GREEN, "waived": GREY}
DB_PATH = os.environ.get("COMPLIANCE_MONITOR_DB", str(Path(__file__).with_name("compliance_monitor.db")))
PCT = st.column_config.NumberColumn(format="percent")
DAY = st.column_config.DateColumn(format="DD MMM YYYY")
ONE = st.column_config.NumberColumn(format="%.1f")
NOT_ADVICE = "Decision support, not legal advice."


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


# ------------------------------------------------------------------- caching
@st.cache_resource
def get_store() -> cm.FindingStore:
    return cm.FindingStore(DB_PATH)


@st.cache_resource(show_spinner="Generating simulated records...")
def get_world() -> cm.SampleWorld:
    return cm.generate_sample()


@st.cache_data(max_entries=4, show_spinner="Measuring the rules against the injected violations...")
def detection(rules_json: str) -> dict:
    world, rules = get_world(), cm.RuleSet.from_json(rules_json)
    return {"table": ev.evaluate_sample(world, rules), "outliers": ev.outlier_methods(world), "classifier": ev.classifier_summary(world),
            "classifier_thresholds": ev.classifier_thresholds(world), "screening": ev.screening_thresholds(world),
            "benchmark": ev.screening_benchmark(threshold=float(rules["SCR-01"].params["threshold"]) if "SCR-01" in [r.id for r in rules] else 0.85)}


@st.cache_data(max_entries=4, show_spinner="Comparing names...")
def all_matches(key: tuple, _suppliers: pd.DataFrame, _screening: pd.DataFrame) -> pd.DataFrame:
    return cm.screen_names(_suppliers, _screening, threshold=0.5)


def rerun_check(records, screening) -> dict:
    """Save the rules and run the check again on the current check date."""
    store.set_meta("rules", st.session_state["rules"].to_json())
    return cm.daily_check(records, store, st.session_state["rules"], as_of, screening)


def start_sample() -> None:
    rules = st.session_state["rules"]
    store.reset()
    store.set_meta("source", "sample")
    store.set_meta("simulated", "1")
    store.set_meta("rules", rules.to_json())
    cm.replay_sample(get_world(), store, rules)
    for k in ("records", "screening_list"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Compliance Monitor")
st.sidebar.caption("Supply chain compliance checks, findings and audit trail")
page = st.sidebar.radio("View", ["Open findings", "Supplier profile", "Screening review", "Rules", "Detection performance", "Trends",
                                 "Data", "How it works"])

store = get_store()
if "rules" not in st.session_state:
    saved = store.get_meta("rules")
    try:
        st.session_state["rules"] = cm.RuleSet.from_json(saved) if saved else cm.default_rules()
    except ValueError:
        st.session_state["rules"] = cm.default_rules()
rules: cm.RuleSet = st.session_state["rules"]
source = store.get_meta("source")
if source is None or (source == "sample" and store.runs().empty):
    with st.spinner("First start: replaying twelve weeks of simulated records into the findings database..."):
        start_sample()
    source = "sample"
if source == "sample":
    records, screening = get_world().records(), st.session_state.get("screening_list", get_world().screening_list)
    st.sidebar.info("Using simulated records and a fictional screening list. Load your own on the **Data** view.")
else:
    records, screening = st.session_state.get("records"), st.session_state.get("screening_list")
runs = store.runs()
as_of = runs["as_of"].max() if len(runs) else pd.Timestamp.today().normalize()
findings = store.findings(as_of)
active = findings[findings["active"]]
st.sidebar.caption(f"Last check: {as_of:%d %b %Y}. {len(active)} open findings, {len(findings) - len(active)} closed.")
st.sidebar.warning(f"{NOT_ADVICE} Deadlines and periods are your settings, not legal requirements.")


def need_records() -> None:
    if records is None:
        st.warning("The findings in the database come from records you uploaded in an earlier session, and uploaded records are not "
                   "stored. Upload them again on the **Data** view to use this view.")
        st.stop()


# ============================================================= OPEN FINDINGS
if page == "Open findings":
    st.title(f"Findings on {as_of:%A %d %B %Y}")
    st.info(cm.headline(findings, as_of))
    k = st.columns(5)
    k[0].metric("Open or in review", len(active))
    k[1].metric("Critical", int((active["severity"] == "critical").sum()))
    k[2].metric("High", int((active["severity"] == "high").sum()))
    k[3].metric("Past due date", int(active["overdue"].sum()) if len(active) else 0)
    k[4].metric("No owner", int((active["owner"] == "").sum()))

    if len(active):
        left, right = st.columns(2)
        with left:
            by = active.groupby(["title", "severity"]).size().reset_index(name="findings")
            fig = px.bar(by, x="findings", y="title", color="severity", orientation="h", color_discrete_map=SEVERITY_COLORS,
                         category_orders={"severity": cm.SEVERITIES}, labels={"findings": "Open findings", "title": ""})
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            show(fig, 360)
        with right:
            ages = cm.ageing_table(findings).melt(id_vars="severity", value_vars=AGE_BANDS, var_name="age", value_name="findings")
            fig = px.bar(ages, x="age", y="findings", color="severity", color_discrete_map=SEVERITY_COLORS,
                         category_orders={"severity": cm.SEVERITIES, "age": AGE_BANDS}, labels={"age": "Days since first seen", "findings": "Open findings"})
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            show(fig, 360)

    st.subheader("Findings")
    c = st.columns(4)
    statuses = c[0].multiselect("Status", cm.STATUSES, default=["open", "in review"])
    severities = c[1].multiselect("Severity", cm.SEVERITIES)
    categories = c[2].multiselect("Category", cm.CATEGORIES)
    names = sorted(findings["supplier_name"].unique()) if len(findings) else []
    suppliers = c[3].multiselect("Supplier", names)
    c = st.columns(4)
    only_overdue = c[0].checkbox("Past due date only")
    only_repeat = c[1].checkbox("Repeat findings only", help="The same supplier has had a finding under the same rule before")
    view = findings
    if statuses:
        view = view[view["status"].isin(statuses)]
    if severities:
        view = view[view["severity"].isin(severities)]
    if categories:
        view = view[view["category"].isin(categories)]
    if suppliers:
        view = view[view["supplier_name"].isin(suppliers)]
    if only_overdue:
        view = view[view["overdue"]]
    if only_repeat:
        view = view[view["recurrence"] > 0]
    st.caption(f"{len(view)} finding(s). Click a row to see the evidence and update it.")
    cols = ["finding_id", "severity", "status", "title", "supplier_name", "record_id", "summary", "owner", "due_date", "age_days", "recurrence"]
    event = st.dataframe(view[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="findings_table", height=320,
                         column_config={"due_date": DAY, "age_days": st.column_config.NumberColumn("Age (days)"),
                                        "recurrence": st.column_config.NumberColumn("Earlier findings, same rule and supplier")})
    if len(view):
        chosen = view.iloc[event.selection.rows[0] if event.selection.rows and event.selection.rows[0] < len(view) else 0]
        fid = int(chosen["finding_id"])
        st.subheader(f"Finding {fid}: {chosen['title']}")
        st.markdown(f"**{chosen['supplier_name']}** ({chosen['supplier_id']}), {chosen['record_type']} `{chosen['record_id']}`. {chosen['summary']}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Evidence when raised**")
            evidence = pd.DataFrame([(key.replace("_", " "), json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                                     for key, v in chosen["evidence"].items() if v is not None], columns=["what the rule looked at", "value"])
            st.dataframe(evidence, hide_index=True)
            st.markdown(f"**Suggested next step.** {chosen['remediation']}")
            flags = [f"first seen {chosen['first_seen']:%d %b %Y}", f"last seen {chosen['last_seen']:%d %b %Y}", f"due {chosen['due_date']:%d %b %Y}"]
            if chosen["recurrence"] > 0:
                flags.append(f"repeat: {int(chosen['recurrence'])} earlier finding(s) for this rule and supplier")
            if not chosen["active"] and chosen["present"] == 1:
                flags.append("closed by a person; the record still breaks the rule")
            st.caption("; ".join(flags) + ".")
        with right:
            st.markdown("**Update**")
            with st.form("update_finding"):
                actor = st.text_input("Your name (recorded in the audit trail)", st.session_state.get("actor", ""))
                f1, f2 = st.columns(2)
                status = f1.selectbox("Status", cm.STATUSES, index=cm.STATUSES.index(chosen["status"]))
                owner = f2.text_input("Owner", chosen["owner"])
                due = f1.date_input("Due date", chosen["due_date"].date())
                reason = f2.text_input("Waiver reason (required to waive)")
                note = st.text_input("Note")
                if st.form_submit_button("Save change", type="primary"):
                    try:
                        store.update(fid, actor, status=status, owner=owner, due_date=due, note=note, waiver_reason=reason)
                    except (ValueError, KeyError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["actor"] = actor
                        st.rerun()
        st.markdown("**Audit trail of this finding**")
        st.dataframe(store.audit_trail(fid)[["at", "actor", "field", "old_value", "new_value", "note"]], hide_index=True)

    st.subheader("Take it with you")
    export = view.drop(columns=["active", "present"]).assign(evidence=view["evidence"].map(json.dumps))
    d = st.columns(2)
    d[0].download_button("Findings shown (CSV)", export.to_csv(index=False), f"findings_{as_of:%Y-%m-%d}.csv", "text/csv", width="stretch",
                         type="primary")
    d[1].download_button("Full audit trail (CSV)", store.audit_trail().to_csv(index=False), "audit_trail.csv", "text/csv", width="stretch")

# ========================================================== SUPPLIER PROFILE
elif page == "Supplier profile":
    need_records()
    st.title("Supplier compliance profile")
    scores = cm.supplier_scores(findings, records.suppliers)
    labels = {f"{r['supplier_name']} ({r['supplier_id']}), score {r['score']:.0f}": r["supplier_id"] for _, r in scores.iterrows()}
    pick = st.selectbox("Supplier (lowest score first)", list(labels))
    sid = labels[pick]
    s = records.suppliers[records.suppliers["supplier_id"] == sid].iloc[0]
    row = scores[scores["supplier_id"] == sid].iloc[0]
    mine = findings[findings["supplier_id"] == sid]
    mine_open = mine[mine["active"]]
    text = (f"**{s['supplier_name']}** scores **{row['score']:.0f}** out of 100 with {len(mine_open)} open finding(s), "
            f"{int(row['overdue'])} past the due date. {len(mine) - len(mine_open)} earlier finding(s) are closed.")
    if (mine_open["category"] == "screening").any():
        text += f" **{cm.POTENTIAL_MATCH}** for its name; this is not a determination."
    st.info(text)
    k = st.columns(5)
    k[0].metric("Compliance score", f"{row['score']:.0f}", help="100 minus points for open findings; not a rating of the supplier")
    k[1].metric("Open findings", len(mine_open))
    k[2].metric("Past due date", int(row["overdue"]))
    k[3].metric("Closed findings", len(mine) - len(mine_open))
    k[4].metric("Shipments on record", int((records.shipments["supplier_id"] == sid).sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("On record")
        day = lambda d: "not recorded" if pd.isna(d) else f"{d:%d %b %Y}"  # noqa: E731
        st.dataframe(pd.DataFrame([("Country", s["country"]), ("Other origins on record", s["other_origins"] or "none"),
                                   ("Onboarded", day(s["onboarded_date"])), ("Insurance expiry", day(s["insurance_expiry"])),
                                   ("Code of conduct acknowledged", day(s["code_of_conduct_ack_date"])),
                                   ("Last audit", day(s["last_audit_date"]))], columns=["item", "on record"]), hide_index=True)
    with right:
        st.subheader("Certificates")
        certs = records.certificates[records.certificates["supplier_id"] == sid].copy()
        if certs.empty:
            st.caption("No certificates on record.")
        else:
            certs["days_to_expiry"] = (certs["expiry_date"] - as_of).dt.days
            fig = go.Figure(go.Bar(x=certs["days_to_expiry"], y=certs["certificate_type"], orientation="h",
                                   marker_color=[RED if d < 0 else AMBER if d <= 30 else GREEN for d in certs["days_to_expiry"]]))
            fig.update_layout(xaxis_title=f"Days until expiry, counted from {as_of:%d %b %Y}")
            show(fig, 90 + 50 * len(certs))
            st.dataframe(certs[["certificate_id", "certificate_type", "expiry_date", "days_to_expiry"]], hide_index=True,
                         column_config={"expiry_date": DAY})
    st.subheader("Findings")
    if mine.empty:
        st.caption("No findings have been recorded for this supplier.")
    else:
        fig = px.histogram(mine, x="first_seen", color="status", color_discrete_map=STATUS_COLORS, category_orders={"status": cm.STATUSES},
                           labels={"first_seen": "First seen", "status": ""}, nbins=20)
        fig.update_layout(yaxis_title="Findings", legend=dict(orientation="h", y=-0.3))
        show(fig, 260)
        st.dataframe(mine[["finding_id", "severity", "status", "title", "record_id", "summary", "owner", "first_seen", "due_date", "closed_on",
                           "recurrence"]], hide_index=True, column_config={"first_seen": DAY, "due_date": DAY, "closed_on": DAY})
    html = cm.build_supplier_profile(store, records, sid, as_of, simulated=source == "sample")
    st.download_button("Supplier profile (HTML)", html, f"supplier_profile_{sid}.html", "text/html", type="primary")
    st.caption("The score summarises the open findings in this tool. It is not a rating of the supplier and says nothing about problems "
               "the rules do not look for.")

# ========================================================== SCREENING REVIEW
elif page == "Screening review":
    need_records()
    st.title("Screening: potential matches for human review")
    rule = rules["SCR-01"] if "SCR-01" in [r.id for r in rules] else None
    threshold = float(rule.params["threshold"]) if rule else 0.85
    if screening is None or screening.empty:
        st.info("No screening list is loaded, so no names are screened. Upload the list or lists that apply to you below.")
        matches = pd.DataFrame(columns=["supplier_id", "supplier_name", "list_id", "list_name", "source", "score", "token_similarity",
                                        "character_similarity", "status"])
    else:
        key = (len(records.suppliers), "|".join(records.suppliers["supplier_name"]), len(screening), "|".join(screening["name"]))
        matches = all_matches(key, records.suppliers, screening)
        above = matches[matches["score"] >= threshold]
        waiting = active[active["category"] == "screening"]
        st.info(f"{above['supplier_id'].nunique()} of {len(records.suppliers)} supplier names resemble an entry on the screening list "
                f"({len(screening)} entries) at the rule's threshold of {threshold:.2f}. {len(waiting)} potential match(es) are waiting for a "
                "person to review them. A similar name is **not** a determination: every row needs someone to compare identifying "
                "details such as address, registration and ownership.")
    if source == "sample":
        st.warning("The list in use is a fictional sample with made-up names. It is not a real screening list.")
    k = st.columns(4)
    k[0].metric("List entries", 0 if screening is None else len(screening))
    k[1].metric("Suppliers compared", len(records.suppliers))
    k[2].metric("Potential matches at threshold", int((matches["score"] >= threshold).sum()))
    k[3].metric("Waiting for review", int((active["category"] == "screening").sum()))

    if len(matches):
        explore = st.slider("Explore another threshold (does not change the rule)", 0.50, 1.00, threshold, 0.01)
        left, right = st.columns(2)
        with left:
            best = matches.groupby("supplier_id")["score"].max().reset_index()
            fig = px.histogram(best, x="score", nbins=25, color_discrete_sequence=[BLUE],
                               labels={"score": "Best similarity of each supplier name to any list entry (0.5 and above)"})
            fig.add_vline(x=explore, line_dash="dash", line_color=RED)
            fig.update_layout(yaxis_title="Suppliers")
            show(fig, 300)
        with right:
            counts = pd.DataFrame({"threshold": [round(0.5 + 0.05 * i, 2) for i in range(10)]})
            counts["suppliers raised"] = [int(best[best["score"] >= t].shape[0]) for t in counts["threshold"]]
            fig = px.line(counts, x="threshold", y="suppliers raised", markers=True, color_discrete_sequence=[BLUE])
            fig.add_vline(x=explore, line_dash="dash", line_color=RED)
            show(fig, 300)
        st.caption("A lower threshold raises more names for review and misses fewer altered spellings; a higher one does the opposite.")
        shown = matches[matches["score"] >= explore]
        st.dataframe(shown, hide_index=True, column_config={
            "score": st.column_config.ProgressColumn("Similarity", format="%.2f", min_value=0, max_value=1),
            "token_similarity": st.column_config.NumberColumn(format="%.2f"), "character_similarity": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Potential matches shown (CSV)", shown.to_csv(index=False), "potential_matches.csv", "text/csv")
        st.caption("Record the outcome of each review on the **Open findings** view (filter Category to screening): resolve with a note, "
                   "or keep in review and escalate.")

    st.subheader("Compare two names")
    c = st.columns(2)
    a = c[0].text_input("Name", "Zorvex Marlune Trading Ltd")
    b = c[1].text_input("Other name", "Marlune Zorvex Trading")
    st.markdown(f"Compared as `{cm.normalise(a)}` and `{cm.normalise(b)}`: similarity **{cm.name_similarity(a, b):.2f}**.")

    st.subheader("Screening list")
    up = st.file_uploader("Your screening list (CSV with a 'name' column; optional 'list_id' and 'source')", type="csv")
    c = st.columns(2)
    if up is not None and c[0].button("Use this list and re-run the check", type="primary"):
        try:
            loaded = cm.validate_screening_list(pd.read_csv(BytesIO(up.getvalue()), dtype=str))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use this file: {exc}")
        else:
            st.session_state["screening_list"] = loaded
            rerun_check(records, loaded)
            st.rerun()
    if source == "sample" and "screening_list" in st.session_state and c[1].button("Back to the fictional sample list"):
        st.session_state.pop("screening_list")
        rerun_check(records, get_world().screening_list)
        st.rerun()
    if screening is not None:
        st.dataframe(screening.head(200), hide_index=True, height=240)

# ===================================================================== RULES
elif page == "Rules":
    st.title("Rules")
    st.info(f"{len(rules.enabled())} of {len(rules)} rules are switched on. Every deadline, grace period, age limit and retention period "
            "here is a setting with a neutral placeholder default, not a statement of what any law or regulation requires. "
            f"{CONFIRM}")
    k = st.columns(4)
    k[0].metric("Rules", len(rules))
    k[1].metric("Switched on", len(rules.enabled()))
    k[2].metric("Categories", len({r.category for r in rules}))
    k[3].metric("Open findings", len(active))
    table = rules.to_table()
    by = table.groupby(["category", "severity"]).size().reset_index(name="rules")
    fig = px.bar(by, x="rules", y="category", color="severity", orientation="h", color_discrete_map=SEVERITY_COLORS,
                 category_orders={"severity": cm.SEVERITIES}, labels={"rules": "Rules", "category": ""})
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    show(fig, 260)

    st.subheader("Switch rules on or off, change severity")
    edited = st.data_editor(table[["id", "enabled", "severity", "title", "category", "applies_to", "description", "params"]], hide_index=True,
                            key="rules_editor", disabled=["id", "title", "category", "applies_to", "description", "params"],
                            column_config={"enabled": st.column_config.CheckboxColumn("On"),
                                           "severity": st.column_config.SelectboxColumn("Severity", options=cm.SEVERITIES, required=True)})
    st.subheader("Parameters")
    rid = st.selectbox("Rule", [r.id for r in rules], format_func=lambda i: f"{i}: {rules[i].title}")
    rule = rules[rid]
    st.markdown(f"*Checks that:* {rule.description}  \n*Suggested next step:* {rule.remediation}")
    new_params = {}
    if not rule.params:
        st.caption("This rule has no parameters.")
    for name, value in rule.params.items():
        label, wkey = name.replace("_", " "), f"param_{rid}_{name}"
        if name == "method":
            new_params[name] = st.selectbox(label, ["robust", "isolation_forest"], index=["robust", "isolation_forest"].index(value), key=wkey)
        elif isinstance(value, bool):
            new_params[name] = st.checkbox(label, value, key=wkey)
        elif isinstance(value, int):
            new_params[name] = int(st.number_input(label, value=value, step=1, key=wkey))
        elif isinstance(value, float):
            new_params[name] = float(st.number_input(label, value=value, step=0.01, format="%.3f", key=wkey))
        elif isinstance(value, list):
            new_params[name] = [v.strip() for v in st.text_input(label + " (comma separated)", ", ".join(value), key=wkey).split(",") if v.strip()]
        else:
            new_params[name] = st.text_input(label, str(value), key=wkey)
    st.caption(CONFIRM)

    c = st.columns(3)
    if c[0].button("Save rules and re-run the check", type="primary", disabled=records is None):
        for _, r in edited.iterrows():
            rules.set_enabled(r["id"], bool(r["enabled"]))
            rules[r["id"]].severity = r["severity"]
        if new_params:
            rules.set_params(rid, **new_params)
        try:
            result = rerun_check(records, screening)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["rules_message"] = (f"Checked again on {as_of:%d %b %Y}: {result['new']} new finding(s), {result['auto_closed']} "
                                                 "closed because the record no longer breaks a rule.")
            st.rerun()
    if c[1].button("Back to the default rules", disabled=records is None):
        st.session_state["rules"] = cm.default_rules()
        st.session_state.pop("rules_editor", None)
        rerun_check(records, screening)
        st.rerun()
    c[2].download_button("Download rules (JSON)", rules.to_json(), "rules.json", "application/json")
    if "rules_message" in st.session_state:
        st.success(st.session_state.pop("rules_message"))
    st.caption("Switching a rule off stops it raising findings; findings it raised earlier stay as they are. Changing severity applies "
               "to findings raised from now on.")
    up = st.file_uploader("Load rules from a JSON file", type="json")
    if up is not None and st.button("Use these rules and re-run the check", disabled=records is None):
        try:
            st.session_state["rules"] = cm.RuleSet.from_json(up.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("rules_editor", None)
            rerun_check(records, screening)
            st.rerun()

# ===================================================== DETECTION PERFORMANCE
elif page == "Detection performance":
    st.title("How well the rules find what was planted")
    d = detection(rules.to_json())
    table = d["table"]
    total, per_rule = table.iloc[-1], table.iloc[:-1]
    fuzzy = per_rule[per_rule["precision"].lt(1) | per_rule["recall"].lt(1)]
    st.info(f"On the simulated records, with every injected violation in place and the rules as they are set now, the rules raise "
            f"{int(total['flagged'])} findings for {int(total['true_violations'])} injected violations: precision "
            f"**{total['precision']:.1%}**, recall **{total['recall']:.1%}**, {total['false_positives_per_1000']:.1f} false positives per "
            f"1,000 record checks. Rules that test a recorded fact recover it exactly; the {len(fuzzy)} rule(s) that can be wrong are "
            f"{', '.join(fuzzy['rule_id']) or 'none'}. These figures describe the simulation, not your data.")
    k = st.columns(4)
    k[0].metric("Precision, all rules", f"{total['precision']:.1%}", help="Of the findings raised, the share that were injected violations")
    k[1].metric("Recall, all rules", f"{total['recall']:.1%}", help="Of the injected violations, the share that were raised")
    k[2].metric("False positives per 1,000", f"{total['false_positives_per_1000']:.1f}")
    k[3].metric("Injected violations missed", int(total["missed"]))
    long = per_rule.melt(id_vars="rule_id", value_vars=["precision", "recall"], var_name="measure", value_name="value")
    fig = px.bar(long, x="rule_id", y="value", color="measure", barmode="group", color_discrete_map={"precision": BLUE, "recall": GREEN},
                 labels={"rule_id": "", "value": ""})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    show(fig, 320)
    st.dataframe(table.drop(columns="category"), hide_index=True, column_config={
        "precision": PCT, "recall": PCT, "false_positives_per_1000": ONE})
    st.download_button("Detection table (CSV)", table.to_csv(index=False), "detection_performance.csv", "text/csv")

    left, right = st.columns(2)
    with left:
        st.subheader("Declared unit value")
        st.dataframe(d["outliers"][["method", "flagged", "true_positives", "false_positives", "missed", "precision", "recall"]], hide_index=True,
                     column_config={"precision": PCT, "recall": PCT})
        st.caption("A higher threshold raises fewer honest extremes and misses more of the smaller errors. IsolationForest flags a "
                   "fixed share of every tariff code whether or not anything is wrong.")
    with right:
        st.subheader("Description against tariff chapter")
        c = d["classifier"]
        st.dataframe(d["classifier_thresholds"][["max_declared_probability", "flagged", "true_positives", "false_positives", "missed",
                                                 "precision", "recall"]], hide_index=True, column_config={"precision": PCT, "recall": PCT})
        st.caption(f"The classifier names the declared chapter for {c['accuracy_on_untampered']:.1%} of {c['shipments_scored']:,} untampered "
                   f"shipments across {c['chapters']} simulated chapters (cross-fitted).")
    st.subheader("Name screening")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=d["screening"]["threshold"], y=d["screening"]["recall"], name="Recall", mode="lines+markers", line_color=GREEN)
        fig.add_scatter(x=d["screening"]["threshold"], y=d["screening"]["precision"], name="Precision", mode="lines+markers", line_color=BLUE)
        fig.update_layout(xaxis_title="Threshold", legend=dict(orientation="h", y=-0.3))
        fig.update_yaxes(tickformat=".0%")
        show(fig, 300)
        st.caption("The simulated suppliers: 8 with names altered from a list entry, 3 different companies with similar names.")
    with right:
        st.dataframe(d["benchmark"], hide_index=True, column_config={"caught": st.column_config.NumberColumn("Share raised", format="percent"),
                                                                     "average_score": st.column_config.NumberColumn(format="%.2f"),
                                                                     "lowest_score": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Generated variants of the fictional list entries, by kind of alteration, at the rule's threshold. Abbreviated and "
                   "dropped words are mostly missed.")

# ==================================================================== TRENDS
elif page == "Trends":
    need_records()
    st.title("Trends")
    scores = cm.supplier_scores(findings, records.suppliers)
    if len(runs) < 2:
        st.info("Only one check has been run so far. Trends appear once checks have been run on several dates.")
    else:
        first, last = runs.iloc[0], runs.iloc[-1]
        st.info(f"Over {len(runs)} checks from {first['as_of']:%d %b %Y} to {last['as_of']:%d %b %Y}, open findings went from "
                f"{int(first['open_total'])} to {int(last['open_total'])} and the average supplier score from {first['mean_score']:.1f} to "
                f"{last['mean_score']:.1f}. {int(runs['new'].sum())} findings were raised and {int(runs['auto_closed'].sum())} closed "
                f"automatically because the record was fixed. {int(last['overdue'])} are past their due date now.")
    k = st.columns(4)
    k[0].metric("Open findings", len(active), int(runs["open_total"].iloc[-1] - runs["open_total"].iloc[-2]) if len(runs) > 1 else None,
                delta_color="inverse")
    k[1].metric("Past due date", int(active["overdue"].sum()) if len(active) else 0)
    k[2].metric("Average supplier score", f"{scores['score'].mean():.1f}")
    k[3].metric("Suppliers with open findings", int((scores["open_findings"] > 0).sum()))
    if len(runs):
        left, right = st.columns(2)
        with left:
            long = runs.melt(id_vars="as_of", value_vars=[f"open_{s}" for s in cm.SEVERITIES], var_name="severity", value_name="findings")
            long["severity"] = long["severity"].str.replace("open_", "")
            fig = px.area(long, x="as_of", y="findings", color="severity", color_discrete_map=SEVERITY_COLORS,
                          category_orders={"severity": cm.SEVERITIES}, labels={"as_of": "", "findings": "Open findings after each check"})
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            show(fig, 320)
        with right:
            fig = go.Figure()
            fig.add_bar(x=runs["as_of"], y=runs["new"], name="Raised", marker_color=RED)
            fig.add_bar(x=runs["as_of"], y=runs["auto_closed"], name="Closed on re-check", marker_color=GREEN)
            fig.add_scatter(x=runs["as_of"], y=runs["overdue"], name="Past due date", mode="lines+markers", line_color=AMBER)
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2), yaxis_title="Findings")
            show(fig, 320)
        st.caption("The first check picks up the whole backlog, so it raises far more than later ones." if len(runs) > 1 else "")
        left, right = st.columns(2)
        with left:
            fig = px.line(runs, x="as_of", y="mean_score", markers=True, labels={"as_of": "", "mean_score": "Average supplier score"})
            fig.update_traces(line_color=BLUE)
            show(fig, 280)
        with right:
            fig = px.histogram(scores, x="score", nbins=20, color_discrete_sequence=[BLUE], labels={"score": "Supplier score today"})
            fig.update_layout(yaxis_title="Suppliers")
            show(fig, 280)
    st.subheader("Suppliers, lowest score first")
    st.dataframe(scores, hide_index=True, height=300, column_config={"score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100)})
    st.subheader("Open findings by severity and age")
    st.dataframe(cm.ageing_table(findings), hide_index=True)
    st.subheader("Check runs")
    st.dataframe(runs.drop(columns="ran_at"), hide_index=True, column_config={"as_of": DAY, "mean_score": ONE})
    d = st.columns(3)
    d[0].download_button("Audit summary (HTML)", cm.build_audit_report(store, records, rules, as_of, simulated=source == "sample"),
                         f"audit_summary_{as_of:%Y-%m-%d}.html", "text/html", width="stretch", type="primary")
    d[1].download_button("Supplier scores (CSV)", scores.to_csv(index=False), "supplier_scores.csv", "text/csv", width="stretch")
    d[2].download_button("Check runs (CSV)", runs.to_csv(index=False), "check_runs.csv", "text/csv", width="stretch")

# ====================================================================== DATA
elif page == "Data":
    st.title("Data")
    if source == "sample":
        st.info("The dashboard is showing simulated records with fictional suppliers. Upload your own four tables to check them instead; "
                "this starts a new findings database.")
    elif records is None:
        st.info("The findings database holds findings from records you uploaded earlier. Upload today's records to run the next check.")
    else:
        st.info("The dashboard is showing your records. Upload a newer extract to run the next check; findings carry over, and those "
                "whose record has been fixed close automatically.")
    c = st.columns(4)
    ups = {name: c[i].file_uploader(f"{name}.csv", type="csv", key=f"up_{name}") for i, name in enumerate(TABLES)}
    c = st.columns(3)
    up_list = c[0].file_uploader("Screening list (optional)", type="csv", key="up_list")
    check_date = c[1].date_input("Check date", pd.Timestamp.today().date())
    fresh = c[2].checkbox("Start a new findings database", value=source == "sample", disabled=source == "sample",
                          help="Clears all findings, the audit trail and the run history")
    c = st.columns(3)
    if c[0].button("Use these records and run the check", type="primary", disabled=ups["suppliers"] is None):
        try:
            tables = {name: pd.read_csv(BytesIO(f.getvalue()), dtype=str) if f is not None else None for name, f in ups.items()}
            loaded = cm.validate_records(**tables)
            listed = cm.validate_screening_list(pd.read_csv(BytesIO(up_list.getvalue()), dtype=str)) if up_list is not None else None
            if source == "sample" or fresh:
                store.reset()
            store.set_meta("source", "yours")
            store.set_meta("simulated", "0")
            store.set_meta("rules", rules.to_json())
            cm.daily_check(loaded, store, rules, check_date, listed)
        except (cm.SchemaError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use these files: {exc}")
        else:
            st.session_state.update(records=loaded, screening_list=listed)
            st.rerun()
    if c[1].button("Reset to simulated data", help="Clears the findings database and replays the simulated records"):
        with st.spinner("Replaying twelve weeks of simulated records..."):
            start_sample()
        st.rerun()
    if records is not None:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name in TABLES:
                z.writestr(f"{name}.csv", records.table(name).to_csv(index=False))
        c[2].download_button("Download current records (ZIP of CSV files)", buffer.getvalue(), "records.zip", "application/zip")
    st.caption(f"Findings are kept in `{DB_PATH}`. Set the environment variable COMPLIANCE_MONITOR_DB to keep them elsewhere. "
               "Uploaded records are held in memory for this session only.")

    st.subheader("Columns")
    meaning = {
        "supplier_id": "Unique identifier", "supplier_name": "Name, used for screening", "country": "Supplier's country",
        "other_origins": "Other countries the supplier ships from, separated by semicolons", "onboarded_date": "Date the supplier was approved",
        "insurance_expiry": "Expiry date of the supplier's insurance", "code_of_conduct_ack_date": "Date the code of conduct was acknowledged; blank if not",
        "last_audit_date": "Date of the last audit", "certificate_id": "Unique identifier", "certificate_type": "What the certificate covers",
        "expiry_date": "Date the certificate expires", "issue_date": "Date the certificate was issued", "entry_id": "Unique identifier of the import entry",
        "entry_date": "Date of entry", "tariff_code": "Tariff code as declared", "goods_description": "Free-text description of the goods",
        "declared_value_usd": "Declared value", "quantity": "Declared quantity", "country_of_origin": "Country of origin as declared",
        "product_id": "Product shipped", "unit": "Unit of quantity", "filed_at": "Date and time the entry was filed; blank if not yet",
        "records_purged_on": "Date the entry's records were deleted; blank if still kept",
        "has_commercial_invoice": "Commercial invoice on file (yes/no)", "has_packing_list": "Packing list on file (yes/no)",
        "has_bill_of_lading": "Bill of lading on file (yes/no)", "has_certificate_of_origin": "Certificate of origin on file (yes/no)",
        "product_name": "Name", "category": "Product category", "material_declaration_date": "Date of the material declaration; blank if none",
    }
    layout = {"suppliers": (SUPPLIER_COLUMNS, SUPPLIER_OPTIONAL), "certificates": (CERTIFICATE_COLUMNS, CERTIFICATE_OPTIONAL),
              "shipments": (SHIPMENT_COLUMNS, SHIPMENT_OPTIONAL), "products": (PRODUCT_COLUMNS, PRODUCT_OPTIONAL)}
    reference = pd.DataFrame([(f"{name}.csv", col, "yes" if col in req else "no", meaning.get(col, "Supplier the row belongs to"))
                              for name, (req, opt) in layout.items() for col in req + opt], columns=["file", "column", "required", "meaning"])
    st.dataframe(reference, hide_index=True, height=300)
    st.caption("Only suppliers.csv is required. A rule whose columns are blank raises findings for the blanks, so switch off rules "
               "you have no data for.")
    if records is not None:
        st.subheader("Current records")
        n = records.counts()
        k = st.columns(4)
        for i, name in enumerate(TABLES):
            k[i].metric(name.capitalize(), f"{n[name]:,}")
        if n["shipments"]:
            monthly = records.shipments.groupby(pd.Grouper(key="entry_date", freq="MS")).size().reset_index(name="shipments")
            fig = px.bar(monthly, x="entry_date", y="shipments", color_discrete_sequence=[BLUE], labels={"entry_date": "", "shipments": "Shipments by entry month"})
            show(fig, 260)
        tabs = st.tabs([name.capitalize() for name in TABLES])
        for tab, name in zip(tabs, TABLES):
            tab.dataframe(records.table(name).tail(200), hide_index=True, height=280)

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.warning(cm.DISCLAIMER)
    st.info("Each day the tool checks supplier, certificate, shipment and product records against the rules you have switched on, "
            "raises a finding with evidence for every record that breaks one, and tracks each finding until someone resolves it or "
            "the record is fixed.")
    k = st.columns(4)
    k[0].metric("Rules", len(rules))
    k[1].metric("Rule categories", len(cm.CATEGORIES))
    k[2].metric("Checks run so far", len(runs))
    k[3].metric("Audit trail entries", len(store.audit_trail()))
    st.markdown(
        """
**Rules are data.** A rule has an id, a title, a category, a severity, the kind of record it applies to, parameters and
remediation text. You can switch each rule off, change its parameters, and save or load the whole set as JSON. Six categories:
document completeness, expiry, data consistency, timeliness, record retention and screening.

**Parameters are yours.** Every deadline, grace period, age limit and retention period is a setting with a neutral placeholder
default. None is a statement of what a law, regulation or contract requires. Set each to the requirement that applies to you
and confirm it with your customs broker or counsel.

**Screening produces potential matches, never determinations.** Supplier names are reduced to a comparable form (accents,
punctuation and legal-form words removed) and compared with each list entry by word and by character. A score at or above
the threshold raises the supplier for a person to review against identifying details. The list shipped with the tool is
fictional; load the list or lists that apply to you.

**Two checks learn from the records.** Declared unit value is compared with other shipments under the same tariff code using
the median and a robust measure of spread. A text classifier (TF-IDF and logistic regression) learns how each tariff chapter
is usually described in your own records and raises entries whose declared chapter looks unlikely for the description. Both
compare a record with its peers; neither knows the right value or the right code.

**Findings are tracked.** A finding is open, in review, resolved or waived (a waiver needs a reason). It has an owner and a
due date set from its severity. The evidence seen when it was raised is kept. Every change is written to an audit trail with
who, what and when. If the same supplier breaks the same rule again, the new finding is marked as a repeat.

**Re-check closes what has been fixed.** At each run, an open finding whose record no longer breaks the rule is closed
automatically. A finding that a person resolved or waived stays closed while the violation persists; if the record is fixed
and later breaks the rule again, a new finding opens.

**Scores.** A supplier's score is 100 minus points for each open finding (critical 25, high 10, medium 4, low 1), counting at
most three findings per rule. It summarises open findings in this tool and is not a rating of the supplier.

**Measured against planted violations.** The simulated records contain known violations, so each rule's precision, recall
and false positives per 1,000 records can be measured. See the Detection performance view.

**Limits.** The sample data is simulated and cleaner than real records. The rules test what is recorded: a document marked as
on file is not read, and a certificate date is not verified. A record that passes every rule is not thereby compliant, because
the rules cover only what they are configured to cover. Name matching misses names that are abbreviated, translated or
transliterated, and raises unrelated companies with similar names. The learning checks need enough peers and can be wrong in
both directions. Nothing here replaces a customs broker, counsel or a person's judgement.
"""
    )
