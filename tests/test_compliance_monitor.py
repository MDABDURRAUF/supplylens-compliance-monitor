import json
import math

import numpy as np
import pandas as pd
import pytest

import compliance_monitor as cm
from compliance_monitor import evaluate as ev
from compliance_monitor.cli import main
from compliance_monitor.screening import character_similarity, token_similarity
from compliance_monitor.store import AUTO_CLOSED

AS_OF = "2026-06-30"


@pytest.fixture(scope="module")
def world():
    return cm.generate_sample()


@pytest.fixture(scope="module")
def detection(world):
    return ev.evaluate_sample(world).set_index("rule_id")


@pytest.fixture()
def store():
    s = cm.FindingStore(":memory:")
    yield s
    s.close()


# ---------------------------------------------------- small hand-made records
def supplier(sid="S1", **over):
    return {"supplier_id": sid, "supplier_name": f"Alderwick Test Supplies {sid}", "country": "Country A", "other_origins": "",
            "onboarded_date": "2025-01-01", "insurance_expiry": "2027-01-01", "code_of_conduct_ack_date": "2025-01-10",
            "last_audit_date": "2026-01-15", **over}


def shipment(eid, **over):
    return {"entry_id": eid, "supplier_id": "S1", "entry_date": "2026-06-01", "filed_at": "2026-06-02 09:00", "tariff_code": "1001.10.0001",
            "goods_description": "cotton shirts", "declared_value_usd": 1000.0, "quantity": 100, "country_of_origin": "Country A",
            "has_commercial_invoice": True, "has_packing_list": True, "has_bill_of_lading": True, "has_certificate_of_origin": True,
            "records_purged_on": None, **over}


def records(suppliers=None, certificates=None, shipments=None, products=None):
    return cm.validate_records(pd.DataFrame(suppliers or [supplier()]),
                               pd.DataFrame(certificates) if certificates else None,
                               pd.DataFrame(shipments) if shipments else None,
                               pd.DataFrame(products) if products else None)


def only(*rule_ids, **params):
    """The default rules with everything switched off except ``rule_ids``; ``params`` re-parameterise the first of them."""
    rules = cm.default_rules()
    for r in rules:
        r.enabled = r.id in rule_ids
    if params:
        rules.set_params(rule_ids[0], **params)
    return rules


def raised(recs, rules, as_of=AS_OF, screening_list=None):
    return sorted(cm.run_checks(recs, rules, as_of, screening_list)["record_id"])


# --------------------------------------------------------------------- data
def test_sample_data_is_reproducible_and_carries_its_ground_truth(world):
    again = cm.generate_sample()
    for name in ("suppliers", "certificates", "shipments", "products"):
        pd.testing.assert_frame_equal(world.base.table(name), again.base.table(name))
    assert world.base.counts() == {"suppliers": 60, "certificates": 145, "shipments": 5000, "products": 120}
    truth = world.ground_truth(apply_fixes=False)
    assert set(truth["kind"]) == set(cm.data.KIND_RULE) and truth["rule_id"].isin([r.id for r in cm.default_rules()]).all()
    assert truth.groupby("kind").size().min() >= 5 and not truth.duplicated(["rule_id", "record_id"]).any()
    # Every country label is a placeholder and every list entry is marked as fictional.
    assert set(world.base.suppliers["country"]) | set(world.base.shipments["country_of_origin"]) <= set(cm.data.COUNTRIES)
    assert all(c.startswith("Country ") for c in cm.data.COUNTRIES)
    assert world.screening_list["source"].str.contains("Fictional").all()
    # The violations really are in the records: a missing-document shipment has a document flag off.
    sh = world.base.shipments.set_index("entry_id")
    missing = truth[truth["kind"] == "missing_document"]["record_id"]
    flags = [f"has_{d}" for d in cm.DOCUMENTS]
    assert (~sh.loc[missing, flags]).any(axis=1).all() and sh.drop(index=missing)[flags].all(axis=None)


def test_records_as_of_replays_arrivals_and_fixes(world):
    early, late = world.records("2026-04-07"), world.records()
    assert len(early.shipments) < len(late.shipments) == 5000 and early.shipments["entry_date"].max() <= pd.Timestamp("2026-04-07")
    assert (early.shipments["filed_at"].dropna() < pd.Timestamp("2026-04-08")).all()
    fixed = [e for e in world.events if e["kind"] == "malformed_tariff_code" and pd.notna(e["fixed_on"]) and e["fixed_on"] <= world.as_of]
    assert len(fixed) > 10
    e = fixed[0]
    before = world.records(e["fixed_on"] - pd.Timedelta(days=1)).shipments.set_index("entry_id").loc[e["record_id"], "tariff_code"]
    after = world.records(e["fixed_on"]).shipments.set_index("entry_id").loc[e["record_id"], "tariff_code"]
    assert before != after and after == e["fixes"][0][2]
    gt = world.ground_truth(e["fixed_on"])
    assert e["record_id"] not in set(gt[gt["kind"] == "malformed_tariff_code"]["record_id"])
    assert len(world.ground_truth()) < len(world.ground_truth(apply_fixes=False))


def test_validation_reports_problems(tmp_path):
    good = records(shipments=[shipment("E1", has_packing_list="no", filed_at=None)])
    assert good.shipments.loc[0, "has_commercial_invoice"] and not good.shipments.loc[0, "has_packing_list"]
    assert pd.isna(good.shipments.loc[0, "filed_at"]) and good.certificates.empty and good.suppliers.loc[0, "other_origins"] == ""
    with pytest.raises(cm.SchemaError, match="missing required column"):
        cm.validate_records(pd.DataFrame([supplier()]).drop(columns="country"))
    with pytest.raises(cm.SchemaError, match="duplicate supplier_id"):
        records(suppliers=[supplier(), supplier()])
    with pytest.raises(cm.SchemaError, match="not in the suppliers table"):
        records(shipments=[shipment("E1", supplier_id="S9")])
    with pytest.raises(cm.SchemaError, match="unreadable 'entry_date'"):
        records(shipments=[shipment("E1", entry_date="soon")])
    with pytest.raises(cm.SchemaError, match="no readable 'expiry_date'"):
        records(certificates=[{"certificate_id": "C1", "supplier_id": "S1", "certificate_type": "Quality", "expiry_date": ""}])
    cm.save_records(good, tmp_path)
    back = cm.load_records(tmp_path)
    assert back.shipments.loc[0, "tariff_code"] == "1001.10.0001" and not back.shipments.loc[0, "has_packing_list"]
    with pytest.raises(FileNotFoundError):
        cm.load_records(tmp_path / "nowhere")


# -------------------------------------------------------------------- rules
def test_document_completeness_rules():
    recs = records(
        suppliers=[supplier("S1"), supplier("S2", code_of_conduct_ack_date=None), supplier("S3", code_of_conduct_ack_date=None, onboarded_date="2026-06-20")],
        shipments=[shipment("E1"), shipment("E2", has_packing_list=False, has_certificate_of_origin=False), shipment("E3", has_bill_of_lading=False, entry_date="2026-06-29")],
        products=[{"product_id": "P1", "product_name": "Shirts", "category": "Apparel", "material_declaration_date": "2026-01-01"},
                  {"product_id": "P2", "product_name": "Socks", "category": "Apparel", "material_declaration_date": None}])
    found = cm.run_checks(recs, only("DOC-01"), AS_OF)
    assert sorted(found["record_id"]) == ["E2", "E3"]
    e2 = found.set_index("record_id").loc["E2"]
    assert e2["evidence"]["missing_documents"] == ["packing list", "certificate of origin"] and e2["severity"] == "high"
    assert e2["finding_key"] == "DOC-01|shipment|E2" and e2["supplier_name"] == "Alderwick Test Supplies S1"
    assert raised(recs, only("DOC-01", grace_days=3)) == ["E2"]  # E3 entered one day ago: not checked yet
    assert raised(recs, only("DOC-01", required_documents=["commercial_invoice"])) == []
    with pytest.raises(ValueError, match="no column for document"):
        cm.run_checks(recs, only("DOC-01", required_documents=["inspection_report"]), AS_OF)
    assert raised(recs, only("DOC-02")) == ["S2"]  # S3 was onboarded 10 days ago, inside the 30-day grace period
    assert raised(recs, only("DOC-02", grace_days=5)) == ["S2", "S3"]
    assert raised(recs, only("DOC-03")) == ["P2"]


def test_expiry_rules():
    certs = [{"certificate_id": c, "supplier_id": "S1", "certificate_type": "Quality", "expiry_date": d}
             for c, d in (("C1", "2026-06-29"), ("C2", "2026-06-30"), ("C3", "2026-07-30"), ("C4", "2026-07-31"), ("C5", "2026-06-20"))]
    recs = records(
        suppliers=[supplier("S1"), supplier("S2", insurance_expiry="2026-06-01", last_audit_date="2025-06-29"),
                   supplier("S3", insurance_expiry=None, last_audit_date="2025-06-30")],
        certificates=certs,
        products=[{"product_id": "P1", "product_name": "Shirts", "category": "Apparel", "material_declaration_date": "2024-06-29"},
                  {"product_id": "P2", "product_name": "Socks", "category": "Apparel", "material_declaration_date": "2024-06-30"}])
    assert raised(recs, only("EXP-01")) == ["C1", "C5"]            # expired the day before, and ten days before
    assert raised(recs, only("EXP-01", grace_days=5)) == ["C5"]
    assert raised(recs, only("EXP-02")) == ["C2", "C3"]            # expires today, and in exactly 30 days; 31 days is outside
    assert raised(recs, only("EXP-02", days_ahead=31)) == ["C2", "C3", "C4"]
    soon = cm.run_checks(recs, only("EXP-02"), AS_OF).set_index("record_id")
    assert soon.loc["C3", "evidence"]["days_to_expiry"] == 30 and soon.loc["C3", "severity"] == "low"
    assert raised(recs, only("EXP-03")) == ["S2", "S3"]            # expired, and no date on file
    assert raised(recs, only("EXP-03", grace_days=40)) == ["S3"]
    assert raised(recs, only("EXP-04")) == ["S2"]                  # one day more than 12 months ago; exactly 12 months passes
    assert raised(recs, only("EXP-04", max_age_months=6)) == ["S2", "S3"]
    assert raised(recs, only("EXP-05")) == ["P1"]                  # one day more than 24 months ago


def test_data_consistency_rules():
    recs = records(
        suppliers=[supplier("S1"), supplier("S2", country="Country B", other_origins="Country C; country d")],
        shipments=[shipment("E1"), shipment("E2", tariff_code="1001100001"), shipment("E3", tariff_code="1OO1.10.0001"),
                   shipment("E4", tariff_code=""), shipment("E5", tariff_code=" 1001.10.0001 "),
                   shipment("E6", country_of_origin="Country B"),
                   shipment("E7", supplier_id="S2", country_of_origin="Country D"), shipment("E8", supplier_id="S2", country_of_origin="Country A"),
                   shipment("E9", supplier_id="S2", country_of_origin="country b")])
    assert raised(recs, only("CON-01")) == ["E2", "E3", "E4"]
    assert raised(recs, only("CON-01", pattern=r"\d{10}|\d{4}\.\d{2}\.\d{4}")) == ["E3", "E4"]  # the format is configurable
    with pytest.raises(ValueError, match="not a valid regular expression"):
        cm.run_checks(recs, only("CON-01", pattern="("), AS_OF)
    origin = cm.run_checks(recs, only("CON-02"), AS_OF).set_index("record_id")
    assert sorted(origin.index) == ["E6", "E8"]  # E7 and E9 are on record for S2 (other origin, and case ignored)
    assert origin.loc["E8", "evidence"]["countries_on_record"] == ["country b", "country c", "country d"]


def test_timeliness_and_retention_rules():
    recs = records(shipments=[
        shipment("E1", entry_date="2026-06-01", filed_at="2026-06-06 23:30"),   # on the deadline day
        shipment("E2", entry_date="2026-06-01", filed_at="2026-06-07 00:10"),   # one day late
        shipment("E3", entry_date="2026-06-24", filed_at=None),                 # not filed, deadline 29 June has passed
        shipment("E4", entry_date="2026-06-26", filed_at=None),                 # not filed, deadline 1 July still ahead
        shipment("E5", entry_date="2020-01-10", filed_at="2020-01-11", records_purged_on="2025-01-09"),  # purged a day early
        shipment("E6", entry_date="2020-01-10", filed_at="2020-01-11", records_purged_on="2025-01-10")])
    late = cm.run_checks(recs, only("TIM-01"), AS_OF).set_index("record_id")
    assert sorted(late.index) == ["E2", "E3"]
    assert late.loc["E2", "evidence"]["days_late"] == 1 and late.loc["E2", "evidence"]["deadline"] == "2026-06-06"
    assert late.loc["E3", "summary"].startswith("Not filed")
    assert raised(recs, only("TIM-01", deadline_days=0)) == ["E1", "E2", "E3", "E4", "E5", "E6"]
    assert raised(recs, only("TIM-01", deadline_days=10)) == []
    assert raised(recs, only("RET-01")) == ["E5"]
    assert raised(recs, only("RET-01", retention_years=7)) == ["E5", "E6"]
    assert raised(recs, only("RET-01", retention_years=4)) == []


def test_rules_can_be_switched_off_reparameterised_and_saved(tmp_path):
    rules = cm.default_rules()
    assert len(rules) == 15 and {r.category for r in rules} == set(cm.CATEGORIES) and len(rules.enabled()) == 15
    assert all(r.remediation and r.description and r.check in cm.CHECKS for r in rules)
    rules.set_enabled("TIM-01", False)
    rules.set_params("RET-01", retention_years=7)
    with pytest.raises(ValueError, match="no parameter"):
        rules.set_params("RET-01", years=7)
    back = cm.RuleSet.load(rules.save(tmp_path / "rules.json"))
    assert not back["TIM-01"].enabled and back["RET-01"].params == {"retention_years": 7} and back.to_json() == rules.to_json()
    assert "TIM-01" not in cm.run_checks(records(shipments=[shipment("E1", filed_at="2026-07-01")]), back, AS_OF)["rule_id"].tolist()
    # A rules file may reuse a check under a new id with other parameters.
    extra = json.loads(rules.to_json()) + [{**json.loads(rules.to_json())[0], "id": "DOC-01B", "title": "Invoice on file",
                                            "params": {"required_documents": ["commercial_invoice"], "grace_days": 0}}]
    assert len(cm.RuleSet.from_json(json.dumps(extra))) == 16
    for bad in ({"severity": "urgent"}, {"check": "no_such_check"}, {"category": "other"}):
        with pytest.raises(ValueError):
            cm.RuleSet.from_json(json.dumps([{**json.loads(rules.to_json())[0], **bad}]))
    with pytest.raises(ValueError, match="Not a valid rules file"):
        cm.RuleSet.from_json("{not json")


# ---------------------------------------------------------------- screening
def test_name_normalisation_and_similarity_by_hand():
    assert cm.normalise("Zorvex Marlune Trading Ltd.") == "zorvex marlune trading"
    assert cm.normalise("  OSTRÉLLE-Pemberwick & Co., S.A. ") == "ostrelle pemberwick and"
    assert cm.normalise(None) == "" and cm.normalise("Ltd") == ""
    assert cm.name_similarity("Zorvex Marlune Trading Ltd", "Trading, Marlune ZORVEX (LLC)") == 1.0  # order, case, legal form
    # "abcd efgh" against "abcd efgx": words pair up as 1.0 and 0.75; 16 of 18 characters match.
    assert token_similarity("abcd efgh", "abcd efgx") == pytest.approx((1.0 + 0.75) / 2)
    assert character_similarity("abcd efgh", "abcd efgx") == pytest.approx(16 / 18)
    assert cm.name_similarity("Abcd Efgh", "Abcd Efgx") == pytest.approx(0.5 * 0.875 + 0.5 * 16 / 18)
    assert cm.name_similarity("Dravenmor Halcyon Export Co", "Dravenmoor Halcyon Exports") > 0.9   # typos
    assert cm.name_similarity("Quorvane Aldermist Textiles", "Dravenmoor Halcyon Exports") < 0.6   # unrelated
    assert cm.name_similarity("", "Dravenmoor Halcyon Exports") == 0.0


def test_screening_raises_potential_matches_for_review_only():
    sup = [supplier("S1", supplier_name="Ghalwick Obsidian Holding LLC"), supplier("S2", supplier_name="Quorvane Aldermist Textiles Ltd"),
           supplier("S3", supplier_name="Shipping Tessarow Nightjar Inc")]
    recs, listed = records(suppliers=sup), cm.sample_screening_list()
    matches = cm.screen_names(recs.suppliers, listed, threshold=0.85)
    assert matches["supplier_id"].tolist() == ["S3", "S1"] and matches["score"].iloc[0] == 1.0
    assert (matches["status"] == cm.POTENTIAL_MATCH).all() and "needs human review" in cm.POTENTIAL_MATCH
    found = cm.run_checks(recs, only("SCR-01"), AS_OF, listed)
    assert sorted(found["record_id"]) == ["S1", "S3"] and (found["severity"] == "critical").all()
    assert found["summary"].str.startswith("Potential match: needs human review").all()
    assert found.iloc[0]["evidence"]["list_source"] == "Fictional sample list (not a real list)"
    assert raised(recs, only("SCR-01", threshold=0.99), screening_list=listed) == ["S3"]
    assert raised(recs, only("SCR-01")) == []  # no list loaded: nothing is screened
    own = cm.validate_screening_list(pd.DataFrame({"name": ["Quorvane Aldermist Textiles", None, " "]}))
    assert len(own) == 1 and own.loc[0, "source"] == "User-supplied list"
    assert raised(recs, only("SCR-01"), screening_list=own) == ["S2"]
    with pytest.raises(ValueError, match="'name' column"):
        cm.validate_screening_list(pd.DataFrame({"entity": ["x"]}))


# ----------------------------------------------------------------- learning
def test_unit_value_outlier_score_by_hand():
    units = [8, 9, 9, 10, 10, 10, 10, 10, 11, 11, 12, 100]
    rows = [shipment(f"E{i}", quantity=10, declared_value_usd=10 * u) for i, u in enumerate(units)]
    rows += [shipment("X1", tariff_code="2002.20.0002", declared_value_usd=5000), shipment("X2", quantity=0)]
    sh = records(shipments=rows).shipments.set_index("entry_id")
    res = cm.unit_value_outliers(sh)
    # Median unit value 10; median absolute deviation of the logs is ln(1.1); so 100 scores ln(10) / (1.4826 ln 1.1).
    assert res.loc["E11", "score"] == pytest.approx(math.log(10) / (1.4826 * math.log(1.1)))
    assert res.loc["E11", "outlier"] and res["outlier"].sum() == 1 and res.loc["E0", "group_median"] == pytest.approx(10)
    assert res.loc["E0", "score"] == pytest.approx(math.log(0.8) / (1.4826 * math.log(1.1)))
    assert not res.loc["X1", "scored"] and not res.loc["X2", "scored"]  # a code with one shipment; a zero quantity
    assert cm.unit_value_outliers(sh, threshold=20)["outlier"].sum() == 0
    assert not cm.unit_value_outliers(sh, min_group=13)["scored"].any()
    assert cm.unit_value_outliers(sh, method="isolation_forest", contamination=0.1).loc["E11", "outlier"]
    found = cm.run_checks(records(shipments=rows), only("CON-03"), AS_OF)
    assert found["record_id"].tolist() == ["E11"] and "far above" in found.loc[0, "summary"] and found.loc[0, "score"] == pytest.approx(16.2948, abs=1e-3)


def test_chapter_classifier_flags_a_description_filed_under_the_wrong_chapter():
    clothes = ["cotton shirts", "wool socks", "knitted cotton trousers", "linen shirts woven", "wool jackets", "cotton socks boxed"]
    pans = ["steel saucepans", "aluminium frying pans", "steel kettles with lids", "glass mixing bowls", "steel cutlery sets", "aluminium saucepans boxed"]
    rows = [shipment(f"A{i}", tariff_code="1001.10.0001", goods_description=clothes[i % 6]) for i in range(30)]
    rows += [shipment(f"B{i}", tariff_code="3003.30.0003", goods_description=pans[i % 6]) for i in range(30)]
    rows += [shipment("WRONG", tariff_code="1001.10.0001", goods_description="steel frying pans with lids"),
             shipment("NOCODE", tariff_code="", goods_description="steel kettles")]
    recs = records(shipments=rows)
    fit = cm.chapter_fit(recs.shipments).set_index(recs.shipments["entry_id"])
    assert fit.loc["WRONG", "declared_chapter"] == "10" and fit.loc["WRONG", "predicted_chapter"] == "30" and fit.loc["WRONG", "p_declared"] < 0.2
    assert not fit.loc["NOCODE", "scored"] and fit.loc["A0", "predicted_chapter"] == "10" and fit.loc["A0", "p_declared"] > 0.8
    assert raised(recs, only("CON-04", max_declared_probability=0.2)) == ["WRONG"]
    assert cm.declared_chapter(pd.Series(["1001.10.0001", "1O01", "", None, "7"])).tolist() == ["10", "", "", "", ""]
    assert not cm.chapter_fit(recs.shipments[recs.shipments["tariff_code"].str.startswith("10")])["scored"].any()  # one chapter only


# -------------------------------------------------------- workflow and audit
def test_findings_are_stored_with_evidence_due_date_and_audit_trail(store):
    recs = records(suppliers=[supplier("S1", insurance_expiry="2026-06-01", code_of_conduct_ack_date=None)],
                   shipments=[shipment("E1", has_packing_list=False)])
    counts = store.sync(cm.run_checks(recs, only("DOC-01", "DOC-02", "EXP-03"), AS_OF), AS_OF)
    assert counts == {"detected": 3, "new": 3, "recurring": 0, "still_open": 0, "auto_closed": 0}
    f = store.findings(AS_OF).set_index("rule_id")
    assert (f["status"] == "open").all() and (f["owner"] == "").all() and (f["first_seen"] == pd.Timestamp(AS_OF)).all()
    # Due dates come from severity: high 7 days, medium 30 days.
    assert f.loc["DOC-01", "due_date"] == pd.Timestamp("2026-07-07") and f.loc["DOC-02", "due_date"] == pd.Timestamp("2026-07-30")
    assert f.loc["DOC-01", "evidence"]["missing_documents"] == ["packing list"] and f.loc["EXP-03", "evidence"]["insurance_expiry"] == "2026-06-01"
    trail = store.audit_trail()
    assert len(trail) == 3 and (trail["actor"] == "system").all() and (trail["new_value"] == "open").all()
    later = store.findings("2026-07-10").set_index("rule_id")
    assert later.loc["DOC-01", "overdue"] and not later.loc["DOC-02", "overdue"] and later.loc["DOC-01", "age_days"] == 10
    custom = cm.FindingStore(":memory:", due_days={"high": 1})
    custom.sync(cm.run_checks(recs, only("DOC-01"), AS_OF), AS_OF)
    assert custom.findings().loc[0, "due_date"] == pd.Timestamp("2026-07-01")


def test_status_owner_and_waiver_changes_are_audited(store):
    recs = records(shipments=[shipment("E1", has_packing_list=False), shipment("E2", filed_at="2026-06-20")])
    store.sync(cm.run_checks(recs, only("DOC-01", "TIM-01"), AS_OF), AS_OF)
    ids = store.findings().set_index("record_id")["finding_id"]
    with pytest.raises(ValueError, match="needs an actor"):
        store.update(ids["E1"], "", status="in review")
    with pytest.raises(ValueError, match="requires a reason"):
        store.update(ids["E2"], "Analyst A", status="waived")
    with pytest.raises(ValueError, match="status must be one of"):
        store.update(ids["E1"], "Analyst A", status="closed")
    with pytest.raises(KeyError):
        store.update(999, "Analyst A", status="resolved")
    store.update(ids["E1"], "Analyst A", status="in review", owner="Analyst B", due_date="2026-07-15", note="asked the forwarder", at="2026-07-01 09:00")
    store.update(ids["E1"], "Analyst B", note="forwarder replied", at="2026-07-02 09:00")
    store.update(ids["E2"], "Compliance lead", status="waived", waiver_reason="carrier outage, documented", at="2026-07-02 10:00")
    f = store.findings("2026-07-03").set_index("record_id")
    assert f.loc["E1", ["status", "owner"]].tolist() == ["in review", "Analyst B"] and f.loc["E1", "due_date"] == pd.Timestamp("2026-07-15")
    assert f.loc["E2", "status"] == "waived" and f.loc["E2", "waiver_reason"] == "carrier outage, documented"
    assert f.loc["E2", "closed_on"] == pd.Timestamp("2026-07-02") and not f.loc["E2", "active"]
    trail = store.audit_trail(ids["E1"])
    assert trail["field"].tolist() == ["status", "status", "owner", "due_date", "comment"]
    assert trail["actor"].tolist() == ["system", "Analyst A", "Analyst A", "Analyst A", "Analyst B"]
    row = trail.iloc[2]
    assert (row["old_value"], row["new_value"], row["note"], row["at"]) == ("", "Analyst B", "asked the forwarder", "2026-07-01 09:00:00")
    waiver = store.audit_trail(ids["E2"]).iloc[-1]
    assert (waiver["old_value"], waiver["new_value"]) == ("open", "waived") and "carrier outage" in waiver["note"]
    store.update(ids["E2"], "Compliance lead", status="open", at="2026-07-03 08:00")  # reopening clears the waiver
    assert store.finding(ids["E2"])["waiver_reason"] == "" and pd.isna(store.finding(ids["E2"])["closed_on"])
    assert len(store.audit_trail()) == 8  # two raised, three changes, a comment, a waiver, a reopening: nothing is ever removed from the trail


def test_recheck_closes_fixed_findings_and_keeps_a_persons_decision(store):
    rules = only("DOC-01", "TIM-01")
    broken = [shipment("E1", has_packing_list=False), shipment("E2", filed_at="2026-06-20"), shipment("E3", has_bill_of_lading=False)]
    store.sync(cm.run_checks(records(shipments=broken), rules, "2026-06-28"), "2026-06-28")
    ids = store.findings().set_index("record_id")["finding_id"]
    # A late filing cannot be undone: a person resolves it, and the next run must not raise it again.
    store.update(ids["E2"], "Analyst A", status="resolved", note="reason for delay documented", at="2026-06-28 12:00")
    fixed = [shipment("E1"), shipment("E2", filed_at="2026-06-20"), shipment("E3", has_bill_of_lading=False)]
    counts = store.sync(cm.run_checks(records(shipments=fixed), rules, "2026-06-29"), "2026-06-29")
    assert counts == {"detected": 2, "new": 0, "recurring": 0, "still_open": 1, "auto_closed": 1}
    f = store.findings("2026-06-29").set_index("record_id")
    assert f.loc["E1", "status"] == "resolved" and f.loc["E1", "resolution"] == AUTO_CLOSED and f.loc["E1", "closed_on"] == pd.Timestamp("2026-06-29")
    assert f.loc["E2", "status"] == "resolved" and f.loc["E2", "resolution"] == "reason for delay documented" and f.loc["E2", "present"] == 1
    assert f.loc["E3", "status"] == "open" and f.loc["E3", "last_seen"] == pd.Timestamp("2026-06-29") and len(f) == 3
    closing = store.audit_trail(ids["E1"]).iloc[-1]
    assert (closing["actor"], closing["old_value"], closing["new_value"], closing["note"]) == ("system", "open", "resolved", AUTO_CLOSED)
    # The document goes missing again: a new finding opens, marked as a repeat for this rule and supplier.
    counts = store.sync(cm.run_checks(records(shipments=broken), rules, "2026-06-30"), "2026-06-30")
    assert (counts["new"], counts["recurring"], counts["auto_closed"]) == (1, 1, 0)
    again = store.findings("2026-06-30")
    e1 = again[again["record_id"] == "E1"].sort_values("finding_id")
    assert e1["status"].tolist() == ["resolved", "open"] and e1["recurrence"].tolist() == [0, 2]  # E1 and E3 came before
    assert "2 time(s) before" in store.audit_trail(int(e1["finding_id"].iloc[1])).iloc[0]["note"]


def test_switched_off_rules_leave_their_findings_alone(store):
    recs = records(shipments=[shipment("E1", has_packing_list=False, filed_at="2026-06-20")])
    store.sync(cm.run_checks(recs, only("DOC-01", "TIM-01"), "2026-06-29"), "2026-06-29")
    counts = store.sync(cm.run_checks(recs, only("DOC-01"), AS_OF), AS_OF)
    assert counts["auto_closed"] == 0 and (store.findings()["status"] == "open").all()
    counts = store.sync(cm.run_checks(recs, only("TIM-01", "DOC-01", deadline_days=30), AS_OF), AS_OF)  # the deadline is relaxed
    assert counts["auto_closed"] == 1 and store.findings().set_index("rule_id").loc["TIM-01", "status"] == "resolved"


# ------------------------------------------------------------------- scores
def test_supplier_score_and_ageing_by_hand(store):
    sup = [supplier("S1", supplier_name="Ghalwick Obsidian Holding LLC"), supplier("S2")]
    ship = [shipment(f"E{i}", tariff_code="bad") for i in range(4)] + [shipment("L1", has_packing_list=False, supplier_id="S2")]
    certs = [{"certificate_id": "C1", "supplier_id": "S1", "certificate_type": "Quality", "expiry_date": "2026-07-10"}]
    recs = records(suppliers=sup, certificates=certs, shipments=ship)
    result = cm.daily_check(recs, store, only("SCR-01", "CON-01", "EXP-02", "DOC-01"), "2026-06-01", cm.sample_screening_list())
    assert result["open_total"] == 6  # the certificate is more than 30 days from expiry on 1 June
    result = cm.daily_check(recs, store, only("SCR-01", "CON-01", "EXP-02", "DOC-01"), AS_OF, cm.sample_screening_list())
    scores = cm.supplier_scores(store.findings(AS_OF), recs.suppliers).set_index("supplier_id")
    # S1: one critical (25) + four medium under one rule, of which three count (12) + one low (1) = 38 points.
    assert scores.loc["S1", "score"] == 62 and scores.loc["S1", ["critical", "medium", "low", "open_findings"]].tolist() == [1, 4, 1, 6]
    assert scores.loc["S2", "score"] == 90 and scores.index[0] == "S1"
    assert result["mean_score"] == pytest.approx(76) and result["open_critical"] == 1 and result["overdue"] == 2  # critical and high are past due
    ages = cm.ageing_table(store.findings(AS_OF)).set_index("severity")
    assert ages.loc["medium", "8-30 days"] == 4 and ages.loc["low", "0-7 days"] == 1 and ages["total"].sum() == 7
    fid = store.findings().query("severity == 'critical'")["finding_id"].iloc[0]
    store.update(fid, "Analyst A", status="waived", waiver_reason="reviewed: different party")
    assert cm.supplier_scores(store.findings(AS_OF), recs.suppliers).set_index("supplier_id").loc["S1", "score"] == 87
    assert "6 findings are open across 2 suppliers" in cm.headline(store.findings(AS_OF), AS_OF)
    runs = store.runs()
    assert runs["open_total"].tolist() == [6, 7] and runs["new"].tolist() == [6, 1]


# --------------------------------------------------------------- evaluation
def test_detection_table_by_hand():
    recs = records(shipments=[shipment(f"E{i}") for i in range(10)])
    findings = pd.DataFrame({"rule_id": ["DOC-01"] * 4, "record_id": ["E0", "E1", "E2", "E3"]})
    truth = pd.DataFrame({"rule_id": ["DOC-01"] * 3 + ["TIM-01"], "record_id": ["E0", "E1", "E9", "E5"]})
    table = ev.detection_table(findings, truth, recs, only("DOC-01", "TIM-01")).set_index("rule_id")
    doc = table.loc["DOC-01"]
    assert (doc["true_positives"], doc["false_positives"], doc["missed"]) == (2, 2, 1)
    assert doc["precision"] == 0.5 and doc["recall"] == pytest.approx(2 / 3) and doc["false_positives_per_1000"] == 200
    assert table.loc["TIM-01", "recall"] == 0 and np.isnan(table.loc["TIM-01", "precision"])
    assert table.loc["ALL", "recall"] == 0.5 and table.loc["ALL", "false_positives_per_1000"] == 100


def test_rules_find_the_injected_violations(world, detection):
    exact = ["DOC-01", "DOC-02", "DOC-03", "EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05", "CON-01", "TIM-01", "RET-01"]
    assert (detection.loc[exact, "precision"] == 1).all() and (detection.loc[exact, "recall"] == 1).all()
    assert detection.loc[exact, "true_violations"].min() >= 5
    # Origin: every injected mismatch is found, but honest second sites that are not on record are raised too.
    assert detection.loc["CON-02", "recall"] == 1 and 0.3 < detection.loc["CON-02", "precision"] < 0.9
    for rule in ("CON-03", "CON-04"):
        assert detection.loc[rule, "precision"] > 0.8 and 0.7 < detection.loc[rule, "recall"] < 1 and detection.loc[rule, "false_positives_per_1000"] < 5
    # Screening: the abbreviated name is missed, and the three look-alike companies are raised.
    assert detection.loc["SCR-01", ["true_violations", "true_positives", "false_positives", "missed"]].tolist() == [8, 7, 3, 1]
    assert detection.loc["ALL", "recall"] > 0.9 and detection.loc["ALL", "precision"] > 0.8


def test_learning_checks_against_alternatives(world):
    methods = ev.outlier_methods(world).set_index("method")
    assert methods["recall"].is_monotonic_decreasing or methods.iloc[:3]["recall"].is_monotonic_decreasing
    assert methods.iloc[:3]["false_positives"].is_monotonic_decreasing
    assert methods.iloc[3]["false_positives"] > methods.iloc[1]["false_positives"]  # IsolationForest flags a fixed share of every group
    summary = ev.classifier_summary(world)
    assert summary["chapters"] == 8 and summary["accuracy_on_untampered"] > 0.95
    thresholds = ev.classifier_thresholds(world)
    assert thresholds["flagged"].is_monotonic_increasing and thresholds["precision"].iloc[0] > thresholds["precision"].iloc[-1]
    screening = ev.screening_thresholds(world).set_index("threshold")
    assert screening.loc[0.70, "recall"] == 1 and screening.loc[0.95, "false_positives"] == 0
    bench = ev.screening_benchmark(per_kind=20, unrelated=60).set_index("test")
    assert bench.loc[["legal form added", "word order changed", "one typo"], "caught"].eq(1).all()
    assert bench.loc["one word abbreviated", "caught"] < 0.5 and bench.loc["unrelated invented names (wrongly raised)", "caught"] == 0


# ----------------------------------------------------- replay, reports, CLI
def test_replay_builds_a_trend_and_reports_carry_the_disclaimer(store):
    small = cm.generate_sample(n_shipments=1200)
    runs = cm.replay_sample(small, store, weeks=3)
    assert len(runs) == 4 and runs["as_of"].iloc[-1] == small.as_of and runs["shipments"].is_monotonic_increasing
    assert runs["new"].iloc[0] > runs["new"].iloc[1:].max() and runs["auto_closed"].sum() > 0 and runs["mean_score"].between(0, 100).all()
    found = store.findings(small.as_of)
    trail = store.audit_trail()
    assert {"open", "in review", "resolved"} <= set(found["status"]) and {"system", "Compliance lead"} <= set(trail["actor"])
    assert (found.loc[found["status"] == "waived", "waiver_reason"] != "").all()
    recs = small.records()
    worst = cm.supplier_scores(found, recs.suppliers).iloc[0]
    profile = cm.build_supplier_profile(store, recs, worst["supplier_id"], small.as_of, simulated=True)
    report = cm.build_audit_report(store, recs, cm.default_rules(), small.as_of, simulated=True)
    for html in (profile, report):
        assert html.startswith("<!doctype html>") and "Decision support, not legal advice" in html and "simulated data" in html
    assert "Rules in force" in report and "none is a determination" in report and "Audit trail" in profile
    assert f"{worst['score']:.0f}" in profile and "Supplier compliance profile" in profile
    with pytest.raises(KeyError):
        cm.build_supplier_profile(store, recs, "SUP-999", small.as_of)
    odd = records(suppliers=[supplier("S1", supplier_name="<b>Angle & Co</b>", insurance_expiry="2026-01-01")])
    other = cm.FindingStore(":memory:")
    cm.daily_check(odd, other, None, AS_OF)
    assert "<b>Angle" not in cm.build_supplier_profile(other, odd, "S1", AS_OF) and "&lt;b&gt;Angle &amp; Co" in cm.build_audit_report(other, odd, cm.default_rules(), AS_OF)


def test_command_line_end_to_end(tmp_path, capsys, monkeypatch):
    data, db = tmp_path / "data", tmp_path / "findings.db"
    monkeypatch.setenv("COMPLIANCE_MONITOR_DB", str(db))
    assert main(["sample", "--out", str(data)]) == 0
    assert {p.name for p in data.iterdir()} == {"suppliers.csv", "certificates.csv", "shipments.csv", "products.csv", "screening_list.csv",
                                               "ground_truth.csv", "rules.json"}
    assert "Simulated data" in capsys.readouterr().out
    assert main(["check", "--data", str(data), "--as-of", AS_OF, "--out", str(tmp_path / "raised.csv")]) == 0
    out = capsys.readouterr().out
    assert "436 findings raised; 436 new" in out and "not legal advice" in out and db.exists()
    assert main(["check", "--data", str(data), "--as-of", AS_OF]) == 0
    assert "0 new" in capsys.readouterr().out
    raised_csv = pd.read_csv(tmp_path / "raised.csv")
    assert len(raised_csv) == 436 and json.loads(raised_csv.loc[0, "evidence"])["status"] == cm.POTENTIAL_MATCH
    assert main(["findings", "--severity", "critical", "--out", str(tmp_path / "critical.csv")]) == 0
    listed = pd.read_csv(tmp_path / "critical.csv")
    assert len(listed) == 10 and (listed["rule_id"] == "SCR-01").all()
    fid = str(listed["finding_id"].iloc[0])
    assert main(["update", "--id", fid, "--actor", "Analyst A", "--status", "waived"]) == 1
    assert "requires a reason" in capsys.readouterr().err
    assert main(["update", "--id", fid, "--actor", "Analyst A", "--status", "resolved", "--note", "different party"]) == 0
    assert main(["findings", "--severity", "critical"]) == 0
    assert "9 finding(s)" in capsys.readouterr().out
    assert main(["profile", "--data", str(data), "--supplier", "SUP-006", "--out", str(tmp_path / "profile.html")]) == 0
    assert main(["report", "--data", str(data), "--out", str(tmp_path / "audit.html")]) == 0
    assert "simulated data" in (tmp_path / "profile.html").read_text(encoding="utf-8")
    assert "Compliance audit summary: 30 Jun 2026" in (tmp_path / "audit.html").read_text(encoding="utf-8")
    assert main(["evaluate", "--out", str(tmp_path / "detection.csv")]) == 0
    assert "injected violations" in capsys.readouterr().out and len(pd.read_csv(tmp_path / "detection.csv")) == 16
    assert main(["replay", "--weeks", "1", "--db", str(tmp_path / "replay.db")]) == 0
    assert main(["replay", "--weeks", "1", "--db", str(tmp_path / "replay.db")]) == 1  # needs an empty database
    assert main(["check", "--data", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
