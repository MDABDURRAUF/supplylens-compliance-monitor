"""Supplier, certificate, shipment and product records: expected columns, validation, and simulated sample data.

Four tables are checked:

``suppliers``     one row per supplier
``certificates``  one row per certificate held by a supplier
``shipments``     one row per inbound shipment (import entry)
``products``      one row per product bought

The sample data is made up. Every company name, country label, tariff code,
chapter number and goods description in it is invented for the simulation and
does not describe any real company, place or tariff schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DOCUMENTS = ["commercial_invoice", "packing_list", "bill_of_lading", "certificate_of_origin"]

SUPPLIER_COLUMNS = ["supplier_id", "supplier_name", "country"]
SUPPLIER_OPTIONAL = ["other_origins", "onboarded_date", "insurance_expiry", "code_of_conduct_ack_date", "last_audit_date"]
CERTIFICATE_COLUMNS = ["certificate_id", "supplier_id", "certificate_type", "expiry_date"]
CERTIFICATE_OPTIONAL = ["issue_date"]
SHIPMENT_COLUMNS = ["entry_id", "supplier_id", "entry_date", "tariff_code", "goods_description", "declared_value_usd",
                    "quantity", "country_of_origin"]
SHIPMENT_OPTIONAL = ["product_id", "unit", "filed_at", "records_purged_on"] + [f"has_{d}" for d in DOCUMENTS]
PRODUCT_COLUMNS = ["product_id", "product_name", "category"]
PRODUCT_OPTIONAL = ["supplier_id", "tariff_code", "material_declaration_date"]

_DATE_COLUMNS = {
    "suppliers": ["onboarded_date", "insurance_expiry", "code_of_conduct_ack_date", "last_audit_date"],
    "certificates": ["issue_date", "expiry_date"],
    "shipments": ["entry_date", "records_purged_on"],
    "products": ["material_declaration_date"],
}
TABLES = ["suppliers", "certificates", "shipments", "products"]
_ID = {"suppliers": "supplier_id", "certificates": "certificate_id", "shipments": "entry_id", "products": "product_id"}


class SchemaError(ValueError):
    """Raised when a table does not match the expected layout."""


@dataclass
class Records:
    """The four tables that are checked, already validated."""

    suppliers: pd.DataFrame
    certificates: pd.DataFrame
    shipments: pd.DataFrame
    products: pd.DataFrame

    def table(self, name: str) -> pd.DataFrame:
        return getattr(self, name)

    def counts(self) -> dict[str, int]:
        return {name: len(self.table(name)) for name in TABLES}


def _text(s: pd.Series) -> pd.Series:
    """Text column with blanks as empty strings (never the text 'nan')."""
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def _flag(s: pd.Series) -> pd.Series:
    """Read a yes/no column: True/False, 1/0, yes/no, y/n. Blank counts as no."""
    if s.dtype == bool:
        return s
    return _text(s).str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])


def _timestamp(s: pd.Series) -> pd.Series:
    """Parse dates or date-times; values in one column may be written in different formats."""
    try:
        out = pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dtype, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def _date(s: pd.Series) -> pd.Series:
    return _timestamp(s).dt.normalize()


def validate_records(suppliers: pd.DataFrame, certificates: pd.DataFrame | None = None, shipments: pd.DataFrame | None = None,
                     products: pd.DataFrame | None = None) -> Records:
    """Check the four tables and return cleaned copies.

    Dates are parsed, yes/no columns are read, optional columns are added
    (empty) when absent, and a missing table becomes an empty one. A document
    flag that is absent is treated as "not on file". Problems raise a
    ``SchemaError`` saying how many rows are affected; nothing is dropped
    silently.
    """
    given = {"suppliers": suppliers, "certificates": certificates, "shipments": shipments, "products": products}
    required = {"suppliers": SUPPLIER_COLUMNS, "certificates": CERTIFICATE_COLUMNS, "shipments": SHIPMENT_COLUMNS, "products": PRODUCT_COLUMNS}
    optional = {"suppliers": SUPPLIER_OPTIONAL, "certificates": CERTIFICATE_OPTIONAL, "shipments": SHIPMENT_OPTIONAL, "products": PRODUCT_OPTIONAL}
    out, problems = {}, []
    for name in TABLES:
        df = given[name]
        if df is None:
            df = pd.DataFrame(columns=required[name])
        missing = [c for c in required[name] if c not in df.columns]
        if missing:
            raise SchemaError(f"{name}: missing required column(s): {', '.join(missing)}")
        df = df.copy()
        for c in optional[name]:
            if c not in df.columns:
                df[c] = False if c.startswith("has_") else None
        for c in _DATE_COLUMNS[name] + (["filed_at"] if name == "shipments" else []):
            given_value = df[c].notna() & (_text(df[c]) != "")
            df[c] = _timestamp(df[c]) if c == "filed_at" else _date(df[c])
            if (given_value & df[c].isna()).any():
                problems.append(f"{name}: {int((given_value & df[c].isna()).sum())} row(s) with unreadable '{c}'")
        for c in df.columns:
            if c.startswith("has_"):
                df[c] = _flag(df[c])
            elif c not in _DATE_COLUMNS[name] and c not in ("declared_value_usd", "quantity", "filed_at"):
                df[c] = _text(df[c])
        key = _ID[name]
        if (df[key] == "").any():
            problems.append(f"{name}: {int((df[key] == '').sum())} row(s) with a blank {key}")
        if df[key].duplicated().any():
            problems.append(f"{name}: {int(df[key].duplicated().sum())} duplicate {key} value(s)")
        out[name] = df

    sh = out["shipments"]
    for c in ("declared_value_usd", "quantity"):
        sh[c] = pd.to_numeric(sh[c], errors="coerce")
    if sh["entry_date"].isna().any():
        problems.append(f"shipments: {int(sh['entry_date'].isna().sum())} row(s) with no readable 'entry_date'")
    if (sh["declared_value_usd"] < 0).any() or (sh["quantity"] < 0).any():
        problems.append("shipments: declared_value_usd and quantity cannot be negative")
    known = set(out["suppliers"]["supplier_id"])
    for name in ("certificates", "shipments"):
        unknown = ~out[name]["supplier_id"].isin(known)
        if unknown.any():
            problems.append(f"{name}: {int(unknown.sum())} row(s) with a supplier_id that is not in the suppliers table")
    if out["certificates"]["expiry_date"].isna().any():
        problems.append(f"certificates: {int(out['certificates']['expiry_date'].isna().sum())} row(s) with no readable 'expiry_date'")
    if problems:
        raise SchemaError("Invalid records: " + "; ".join(problems))
    out["shipments"] = sh.sort_values(["entry_date", "entry_id"]).reset_index(drop=True)
    return Records(**{name: out[name].reset_index(drop=True) for name in TABLES})


def load_records(folder: str | Path) -> Records:
    """Read ``suppliers.csv``, ``certificates.csv``, ``shipments.csv`` and ``products.csv`` from a folder and validate them.

    Only ``suppliers.csv`` must exist; a table without a file is treated as empty.
    """
    folder = Path(folder)
    if not (folder / "suppliers.csv").exists():
        raise FileNotFoundError(f"No suppliers.csv in {folder}")
    tables = {name: pd.read_csv(folder / f"{name}.csv", dtype=str) if (folder / f"{name}.csv").exists() else None for name in TABLES}
    return validate_records(**tables)


def save_records(records: Records, folder: str | Path) -> Path:
    """Write the four tables as CSV files into a folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        records.table(name).to_csv(folder / f"{name}.csv", index=False)
    return folder


# --------------------------------------------------------------- sample data
COUNTRIES = ["Country A", "Country B", "Country C", "Country D", "Country E", "Country F"]
_GENERIC = ["retail packed", "bulk carton", "assorted", "boxed", "set of 6", "set of 12", "for household use", "for commercial use"]
# Invented chapters. Numbers, names and wording do not follow any real tariff schedule.
_CHAPTERS = {
    "10": dict(name="Apparel", unit="pcs", price=6.0, materials=["cotton", "polyester", "wool", "linen"],
               nouns=["shirts", "trousers", "jackets", "socks", "gloves", "scarves", "dresses", "caps"],
               qualifiers=["men's", "women's", "children's", "knitted", "woven"]),
    "20": dict(name="Home textiles", unit="pcs", price=4.5, materials=["cotton", "polyester", "linen", "microfibre"],
               nouns=["towels", "bed sheets", "curtains", "cushion covers", "blankets", "table cloths", "rugs"],
               qualifiers=["woven", "printed", "quilted", "hemmed"]),
    "30": dict(name="Kitchenware", unit="pcs", price=5.5, materials=["stainless steel", "aluminium", "glass", "bamboo", "ceramic"],
               nouns=["saucepans", "frying pans", "cutlery sets", "mixing bowls", "kettles", "chopping boards", "storage jars"],
               qualifiers=["non-stick", "with lids", "dishwasher safe", "nested"]),
    "40": dict(name="Hand tools and hardware", unit="pcs", price=3.2, materials=["steel", "chrome vanadium steel", "brass", "aluminium"],
               nouns=["wrenches", "screwdrivers", "hammers", "pliers", "hinges", "bolts", "drill bits"],
               qualifiers=["forged", "insulated", "adjustable", "zinc plated", "hex"]),
    "50": dict(name="Electronics accessories", unit="pcs", price=7.5, materials=["plastic", "silicone", "aluminium"],
               nouns=["charging cables", "power adapters", "phone cases", "headphones", "keyboards", "speakers"],
               qualifiers=["wireless", "fast charging", "with cable", "rechargeable", "universal"]),
    "60": dict(name="Toys and games", unit="pcs", price=4.0, materials=["plastic", "wood", "polyester", "cardboard"],
               nouns=["building blocks", "plush toys", "puzzles", "toy cars", "dolls", "board games"],
               qualifiers=["children's", "educational", "painted", "ages 3 and up"]),
    "70": dict(name="Furniture", unit="pcs", price=38.0, materials=["wood", "steel", "bamboo", "glass"],
               nouns=["chairs", "tables", "shelves", "bed frames", "cabinets", "stools"],
               qualifiers=["flat-packed", "folding", "upholstered", "unassembled", "adjustable"]),
    "80": dict(name="Safety equipment", unit="pcs", price=5.0, materials=["polyester", "plastic", "leather", "rubber", "cotton"],
               nouns=["safety gloves", "helmets", "goggles", "high-visibility vests", "ear defenders", "safety boots"],
               qualifiers=["protective", "reinforced", "adjustable", "reflective", "impact resistant"]),
}
_STEM_A = ["Quorvane", "Bellmoth", "Tindrel", "Oskarovel", "Lumentraq", "Kyvorath", "Zentiqor", "Vantrellow", "Marrowby", "Phestor", "Drumlinn",
           "Ersatzia", "Halvique", "Jorrund", "Nimbasco", "Ulvenor", "Wexholm", "Yarrowick", "Cindelle", "Farnoux"]
_STEM_B = ["Aldermist", "Brackwen", "Corvalle", "Dunmorrow", "Eskerby", "Fallowmere", "Grintholm", "Hexwold", "Islingmere", "Juniperre",
           "Kestrelby", "Lowenmark", "Mossgravenn", "Netherwyn", "Orrinshaw", "Pellaton"]
_SUFFIX = ["Ltd", "LLC", "Inc", "Co", "GmbH", "Pte Ltd", "S.A."]
_CERT_TYPES = ["Quality management certificate", "Environmental management certificate", "Social compliance audit certificate"]

# A small FICTIONAL screening list for the demo. None of these names refers to a real party.
SAMPLE_SCREENING_NAMES = [
    "Zorvex Marlune Trading", "Ghalwick Obsidian Holdings", "Tessarow Nightjar Shipping", "Brumvale Quietus Metals",
    "Ixcalune Vesper Logistics", "Dravenmoor Halcyon Exports", "Ostrelle Pemberwick Industries", "Kaldrith Emberfall Commodities",
    "Yuneska Thornquell Group", "Mordaline Fennowick Chemicals", "Vashtelle Corrimoor Enterprises", "Ulmbraith Sennacherry Supply",
]
# Simulated suppliers whose names resemble an entry on the fictional list (ground truth: should be raised for review).
_SCREENING_VARIANTS = [
    "Zorvex Marlune Trading Ltd", "Ghalwick Obsidian Holding LLC", "Shipping Tessarow Nightjar Inc", "Brumvale Quietus Metal Co",
    "Ixcalune-Vesper Logistics GmbH", "Dravenmor Halcyon Export Co", "Ostrélle Pemberwick Industries", "Kaldrith E. Commodities Pte Ltd",
]
# Simulated suppliers that are different companies with similar names (ground truth: not a match).
_SCREENING_LOOKALIKES = ["Zorvex Marlin Trading Co", "Brumvale Quartus Metals Ltd", "Yuneska Thornhall Group Inc"]

# kind of injected violation -> the default rule expected to raise it
KIND_RULE = {
    "missing_document": "DOC-01", "code_of_conduct_not_acknowledged": "DOC-02", "material_declaration_missing": "DOC-03",
    "certificate_expired": "EXP-01", "certificate_expiring_soon": "EXP-02", "insurance_expired": "EXP-03", "audit_overdue": "EXP-04",
    "material_declaration_out_of_date": "EXP-05", "malformed_tariff_code": "CON-01", "origin_mismatch": "CON-02",
    "implausible_unit_value": "CON-03", "description_mismatch": "CON-04", "late_filing": "TIM-01",
    "records_purged_early": "RET-01", "name_resembles_screening_list": "SCR-01",
}


def sample_screening_list() -> pd.DataFrame:
    """The fictional screening list shipped for the demo. Load your own list for real use."""
    return pd.DataFrame({"list_id": [f"SAMPLE-{i + 1:03d}" for i in range(len(SAMPLE_SCREENING_NAMES))],
                         "name": SAMPLE_SCREENING_NAMES, "source": "Fictional sample list (not a real list)"})


@dataclass
class SampleWorld:
    """Simulated records together with the violations that were injected into them.

    ``base`` holds every record up to the last simulated day with every
    violation still in place. ``records(as_of)`` gives the records as they
    would have looked on a date: shipments not yet entered are absent, events
    that had not yet happened are blank, and violations whose simulated fix
    date has passed are corrected. ``ground_truth(as_of)`` lists the injected
    violations present on that date, so detection can be measured.
    """

    base: Records
    events: list[dict] = field(repr=False)
    screening_list: pd.DataFrame = field(repr=False)
    as_of: pd.Timestamp = None
    start: pd.Timestamp = None

    def records(self, as_of=None, apply_fixes: bool = True) -> Records:
        as_of = self.as_of if as_of is None else pd.Timestamp(as_of).normalize()
        tables = {name: self.base.table(name).copy() for name in TABLES}
        if apply_fixes:
            for name in TABLES:
                fixes = [(e["record_id"], column, value) for e in self.events if pd.notna(e["fixed_on"]) and e["fixed_on"] <= as_of
                         for table, column, value in e["fixes"] if table == name]
                if not fixes:
                    continue
                df = tables[name].set_index(_ID[name])
                for column in sorted({c for _, c, _ in fixes}):
                    chosen = {rid: v for rid, c, v in fixes if c == column}
                    df.loc[list(chosen), column] = list(chosen.values())
                tables[name] = df.reset_index()
        sh = tables["shipments"]
        sh = sh[sh["entry_date"] <= as_of].reset_index(drop=True)
        sh["filed_at"] = sh["filed_at"].where(sh["filed_at"].dt.normalize() <= as_of)
        sh["records_purged_on"] = sh["records_purged_on"].where(sh["records_purged_on"] <= as_of)
        tables["shipments"] = sh
        sup = tables["suppliers"]
        sup["code_of_conduct_ack_date"] = sup["code_of_conduct_ack_date"].where(sup["code_of_conduct_ack_date"] <= as_of)
        pr = tables["products"]
        pr["material_declaration_date"] = pr["material_declaration_date"].where(pr["material_declaration_date"] <= as_of)
        return Records(**tables)

    def ground_truth(self, as_of=None, apply_fixes: bool = True) -> pd.DataFrame:
        """Injected violations present on ``as_of``: one row per violation with the rule expected to raise it."""
        as_of = self.as_of if as_of is None else pd.Timestamp(as_of).normalize()
        rows = []
        for e in self.events:
            if e["start"] > as_of or (pd.notna(e["end"]) and as_of > e["end"]):
                continue
            if apply_fixes and pd.notna(e["fixed_on"]) and e["fixed_on"] <= as_of:
                continue
            rows.append({k: e[k] for k in ("kind", "rule_id", "record_type", "record_id", "supplier_id", "detail", "start", "fixed_on")})
        return pd.DataFrame(rows, columns=["kind", "rule_id", "record_type", "record_id", "supplier_id", "detail", "start", "fixed_on"])


def _describe(rng: np.random.Generator, spec: dict, noun: str, material: str, vague: bool = False) -> str:
    """A free-text goods description in one of several word orders."""
    generic = str(rng.choice(_GENERIC))
    if vague:
        return f"{material} articles, {generic}" if rng.random() < 0.6 else f"assorted goods, {generic}"
    qualifier = str(rng.choice(spec["qualifiers"]))
    form = int(rng.integers(0, 4))
    if form == 0:
        return f"{qualifier} {material} {noun}, {generic}"
    if form == 1:
        return f"{noun} of {material}, {qualifier}"
    if form == 2:
        return f"{material} {noun} {generic}"
    return f"{noun}, {qualifier}, {generic}"


def generate_sample(seed: int = 7, n_suppliers: int = 60, n_shipments: int = 5000, n_products: int = 120, days: int = 540,
                    end: str = "2026-06-30") -> SampleWorld:
    """Simulated suppliers, certificates, shipments and products with known violations, for demos and tests.

    The data is made up. Clean records are generated first, then violations
    are injected and written down: missing documents, expired and
    soon-to-expire certificates, expired insurance, overdue audits, missing
    acknowledgments and declarations, malformed tariff codes, origins that
    differ from the supplier's country, implausible unit values, descriptions
    that belong to another chapter, late filings, records purged early, and
    supplier names resembling an entry on the fictional screening list.

    Some clean records are deliberately awkward so that checks can be wrong:
    three suppliers ship from a second site that is not in the supplier table,
    unit values have occasional honest extremes, some descriptions are vague,
    and three suppliers have names close to a list entry without being one.

    A share of violations gets a simulated fix date, used by
    ``SampleWorld.records(as_of)`` to replay the records over time.
    """
    if n_suppliers < 20:
        raise ValueError("n_suppliers must be at least 20")
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end).normalize()
    start = end - pd.Timedelta(days=days - 1)
    day = lambda n: pd.Timedelta(days=int(n))  # noqa: E731
    events: list[dict] = []

    def event(kind, record_type, record_id, supplier_id, start_, detail, fixes=(), fixed_on=pd.NaT, end_=pd.NaT):
        events.append(dict(kind=kind, rule_id=KIND_RULE[kind], record_type=record_type, record_id=record_id, supplier_id=supplier_id,
                           start=pd.Timestamp(start_), end=end_, fixed_on=fixed_on, detail=detail, fixes=list(fixes)))

    # ------------------------------------------------------------ tariff codes
    chapters = list(_CHAPTERS)
    codes, seen = [], set()
    for ch, spec in _CHAPTERS.items():
        while sum(c["chapter"] == ch for c in codes) < 3:
            code = f"{ch}{rng.integers(1, 100):02d}.{rng.integers(10, 100):02d}.{rng.integers(0, 10000):04d}"
            if code not in seen:
                seen.add(code)
                codes.append(dict(tariff_code=code, chapter=ch, base_price=spec["price"] * float(rng.lognormal(0, 0.5))))
    codes = pd.DataFrame(codes)

    # --------------------------------------------------------------- suppliers
    pairs = [(a, b) for a in _STEM_A for b in _STEM_B]
    picked = rng.choice(len(pairs), n_suppliers, replace=False)
    main = np.array([chapters[i % len(chapters)] for i in range(n_suppliers)])
    second = np.where(rng.random(n_suppliers) < 0.35, rng.choice(chapters, n_suppliers), "")
    names = [f"{pairs[p][0]} {pairs[p][1]} {_CHAPTERS[main[i]]['name'].split()[0]} {rng.choice(_SUFFIX)}" for i, p in enumerate(picked)]
    special = rng.choice(n_suppliers, len(_SCREENING_VARIANTS) + len(_SCREENING_LOOKALIKES) + 3, replace=False)
    variant_idx, lookalike_idx, second_site_idx = (special[:len(_SCREENING_VARIANTS)],
                                                   special[len(_SCREENING_VARIANTS):-3], special[-3:])
    for i, name in zip(variant_idx, _SCREENING_VARIANTS):
        names[i] = name
    for i, name in zip(lookalike_idx, _SCREENING_LOOKALIKES):
        names[i] = name
    country = rng.choice(COUNTRIES, n_suppliers, p=[0.3, 0.22, 0.18, 0.12, 0.1, 0.08])
    other = np.array([""] * n_suppliers, dtype=object)
    for i in np.flatnonzero(rng.random(n_suppliers) < 0.2):
        if i not in second_site_idx:
            other[i] = str(rng.choice([c for c in COUNTRIES if c != country[i]]))
    onboarded = start - pd.to_timedelta(rng.integers(200, 1500, n_suppliers), unit="D")
    suppliers = pd.DataFrame({
        "supplier_id": [f"SUP-{i + 1:03d}" for i in range(n_suppliers)], "supplier_name": names, "country": country,
        "other_origins": other, "onboarded_date": onboarded,
        "insurance_expiry": end + pd.to_timedelta(rng.integers(45, 400, n_suppliers), unit="D"),
        "code_of_conduct_ack_date": onboarded + pd.to_timedelta(rng.integers(1, 20, n_suppliers), unit="D"),
        "last_audit_date": end - pd.to_timedelta(rng.integers(95, 330, n_suppliers), unit="D"),
    })
    sid = suppliers["supplier_id"].to_numpy()
    for i in variant_idx:
        event("name_resembles_screening_list", "supplier", sid[i], sid[i], onboarded[i], f"name: {names[i]}")

    ordinary = np.setdiff1d(np.arange(n_suppliers), variant_idx)
    chosen = rng.choice(ordinary, 17, replace=False)
    for n, i in enumerate(chosen[:5]):  # insurance expired
        expiry = end - day(rng.integers(3, 100))
        suppliers.loc[i, "insurance_expiry"] = expiry
        fixed = expiry + day(rng.integers(10, 40)) if n < 2 else pd.NaT
        event("insurance_expired", "supplier", sid[i], sid[i], expiry + day(1), f"insurance expired {expiry:%Y-%m-%d}",
              [("suppliers", "insurance_expiry", expiry + day(365))], fixed)
    for n, i in enumerate(chosen[5:11]):  # code of conduct never acknowledged
        suppliers.loc[i, "code_of_conduct_ack_date"] = pd.NaT
        fixed = end - day(rng.integers(5, 60)) if n < 2 else pd.NaT
        event("code_of_conduct_not_acknowledged", "supplier", sid[i], sid[i], onboarded[i] + day(31), "no acknowledgment on file",
              [("suppliers", "code_of_conduct_ack_date", fixed)] if pd.notna(fixed) else [], fixed)
    for n, i in enumerate(chosen[11:17]):  # audit overdue
        audit = end - day(rng.integers(380, 520))
        suppliers.loc[i, "last_audit_date"] = audit
        overdue_from = audit + pd.DateOffset(months=12) + day(1)
        fixed = max(overdue_from + day(7), end - day(rng.integers(5, 40))) if n < 2 else pd.NaT
        event("audit_overdue", "supplier", sid[i], sid[i], overdue_from, f"last audit {audit:%Y-%m-%d}",
              [("suppliers", "last_audit_date", fixed)] if pd.notna(fixed) else [], fixed)

    # ------------------------------------------------------------ certificates
    certs = []
    for i in range(n_suppliers):
        for kind in rng.choice(_CERT_TYPES, int(rng.integers(2, 4)), replace=False):
            expiry = end + day(rng.integers(45, 700))
            certs.append(dict(certificate_id=f"CERT-{len(certs) + 1:04d}", supplier_id=sid[i], certificate_type=str(kind),
                              issue_date=expiry - day(1095), expiry_date=expiry))
    certificates = pd.DataFrame(certs)
    bad = rng.choice(len(certificates), 15, replace=False)
    for n, j in enumerate(bad):
        expiry = end - day(rng.integers(3, 100)) if n < 9 else end + day(rng.integers(1, 31))
        certificates.loc[j, "expiry_date"] = expiry
        cid, sup = certificates.loc[j, "certificate_id"], certificates.loc[j, "supplier_id"]
        fixed = expiry + day(rng.integers(5, 30)) if n < 4 else pd.NaT
        fix = [("certificates", "expiry_date", expiry + day(730))]
        label = f"{certificates.loc[j, 'certificate_type']} expires {expiry:%Y-%m-%d}"
        event("certificate_expiring_soon", "certificate", cid, sup, expiry - day(30), label, fix, fixed, end_=expiry)
        event("certificate_expired", "certificate", cid, sup, expiry + day(1), label, fix, fixed)

    # ---------------------------------------------------------------- products
    p_supplier = rng.integers(0, n_suppliers, n_products)
    p_chapter = np.array([main[s] if (second[s] == "" or rng.random() < 0.7) else second[s] for s in p_supplier])
    p_rows = []
    for k in range(n_products):
        spec = _CHAPTERS[p_chapter[k]]
        noun, material = str(rng.choice(spec["nouns"])), str(rng.choice(spec["materials"]))
        code = codes[codes["chapter"] == p_chapter[k]].iloc[int(rng.integers(0, 3))]
        p_rows.append(dict(product_id=f"PRD-{k + 1:04d}", product_name=f"{material} {noun}".capitalize(), category=spec["name"],
                           supplier_id=sid[p_supplier[k]], tariff_code=code["tariff_code"],
                           material_declaration_date=end - day(rng.integers(100, 600)),
                           _noun=noun, _material=material, _chapter=p_chapter[k],
                           _price=code["base_price"] * float(rng.lognormal(0, 0.15))))
    products = pd.DataFrame(p_rows)
    bad = rng.choice(n_products, 15, replace=False)
    for n, k in enumerate(bad[:8]):  # declaration missing
        products.loc[k, "material_declaration_date"] = pd.NaT
        fixed = end - day(rng.integers(5, 70)) if n < 3 else pd.NaT
        event("material_declaration_missing", "product", products.loc[k, "product_id"], products.loc[k, "supplier_id"], start,
              "no material declaration on file", [("products", "material_declaration_date", fixed)] if pd.notna(fixed) else [], fixed)
    for k in bad[8:]:  # declaration too old
        declared = end - day(rng.integers(760, 1100))
        products.loc[k, "material_declaration_date"] = declared
        event("material_declaration_out_of_date", "product", products.loc[k, "product_id"], products.loc[k, "supplier_id"],
              declared + pd.DateOffset(months=24) + day(1), f"declaration dated {declared:%Y-%m-%d}")

    # --------------------------------------------------------------- shipments
    n = n_shipments
    popularity = rng.lognormal(0, 0.6, n_products)
    prod = rng.choice(n_products, n, p=popularity / popularity.sum())
    entry = np.sort(rng.integers(0, days, n))
    entry_date = start + pd.to_timedelta(entry, unit="D")
    sup_idx = p_supplier[prod]
    quantity = np.maximum(1, np.round(rng.lognormal(np.log(400), 0.8, n))).astype(int)
    unit_value = products["_price"].to_numpy()[prod] * np.exp(0.11 * rng.standard_t(5, n))
    value = np.round(quantity * unit_value, 2)
    origin = country[sup_idx].astype(object)
    for i in np.flatnonzero(other != ""):  # second origin on record: not a violation
        rows = np.flatnonzero((sup_idx == i) & (rng.random(n) < 0.3))
        origin[rows] = other[i]
    for i in second_site_idx:  # second site NOT on record: honest, but looks like a mismatch
        rows = np.flatnonzero((sup_idx == i) & (rng.random(n) < 0.25))
        origin[rows] = str(rng.choice([c for c in COUNTRIES if c != country[i]]))
    filed_at = entry_date + pd.to_timedelta(np.round(rng.uniform(-48, 96, n) * 60), unit="m")
    description = np.array([_describe(rng, _CHAPTERS[products.loc[p, "_chapter"]], products.loc[p, "_noun"], products.loc[p, "_material"],
                                      vague=rng.random() < 0.03) for p in prod], dtype=object)
    shipments = pd.DataFrame({
        "entry_id": [f"ENT-{100000 + i}" for i in range(n)], "supplier_id": sid[sup_idx], "product_id": products["product_id"].to_numpy()[prod],
        "entry_date": entry_date, "filed_at": filed_at, "tariff_code": products["tariff_code"].to_numpy()[prod],
        "goods_description": description, "declared_value_usd": value, "quantity": quantity, "unit": "pcs",
        "country_of_origin": origin, **{f"has_{d}": True for d in DOCUMENTS}, "records_purged_on": pd.NaT,
    })
    eid, ssid = shipments["entry_id"].to_numpy(), shipments["supplier_id"].to_numpy()

    def fix_date(i, share, low=3, high=60):
        return entry_date[i] + day(rng.integers(low, high)) if rng.random() < share else pd.NaT

    order = rng.permutation(n)
    sizes = {"missing_document": 0.025, "malformed_tariff_code": 0.012, "origin_mismatch": 0.015, "implausible_unit_value": 0.015,
             "description_mismatch": 0.015, "late_filing": 0.02}
    slices, at = {}, 0
    for kind, share in sizes.items():
        slices[kind] = order[at:at + int(round(share * n))]
        at += len(slices[kind])

    for i in slices["missing_document"]:
        gone = rng.choice(DOCUMENTS, 2 if rng.random() < 0.2 else 1, replace=False, p=[0.15, 0.2, 0.2, 0.45])
        for d in gone:
            shipments.loc[i, f"has_{d}"] = False
        event("missing_document", "shipment", eid[i], ssid[i], entry_date[i], "missing: " + ", ".join(sorted(gone)),
              [("shipments", f"has_{d}", True) for d in gone], fix_date(i, 0.6))
    for i in slices["malformed_tariff_code"]:
        good = shipments.loc[i, "tariff_code"]
        how = int(rng.integers(0, 5))
        bad_code = [good.replace(".", ""), good[:-1], good.replace("0", "O", 1) if "0" in good else good + "X",
                    good.replace(".", " ", 1), ""][how]
        shipments.loc[i, "tariff_code"] = bad_code
        event("malformed_tariff_code", "shipment", eid[i], ssid[i], entry_date[i], f"code written as '{bad_code}'",
              [("shipments", "tariff_code", good)], fix_date(i, 0.7))
    for i in slices["origin_mismatch"]:
        good = shipments.loc[i, "country_of_origin"]
        allowed = {country[sup_idx[i]], other[sup_idx[i]]}
        wrong = str(rng.choice([c for c in COUNTRIES if c not in allowed]))
        shipments.loc[i, "country_of_origin"] = wrong
        event("origin_mismatch", "shipment", eid[i], ssid[i], entry_date[i], f"origin given as {wrong}",
              [("shipments", "country_of_origin", good)], fix_date(i, 0.5))
    for i in slices["implausible_unit_value"]:
        good = float(shipments.loc[i, "declared_value_usd"])
        factor = rng.uniform(6, 25) if rng.random() < 0.8 else rng.uniform(2.2, 3.5)
        factor = factor if rng.random() < 0.5 else 1 / factor
        shipments.loc[i, "declared_value_usd"] = round(good * factor, 2)
        event("implausible_unit_value", "shipment", eid[i], ssid[i], entry_date[i], f"declared value is {factor:.2f} times the true value",
              [("shipments", "declared_value_usd", good)], fix_date(i, 0.4))
    for i in slices["description_mismatch"]:
        good = shipments.loc[i, "goods_description"]
        own = products.loc[prod[i], "_chapter"]
        wrong = str(rng.choice([c for c in chapters if c != own]))
        spec = _CHAPTERS[wrong]
        shipments.loc[i, "goods_description"] = _describe(rng, spec, str(rng.choice(spec["nouns"])), str(rng.choice(spec["materials"])),
                                                           vague=rng.random() < 0.1)
        event("description_mismatch", "shipment", eid[i], ssid[i], entry_date[i],
              f"description written for chapter {wrong}, code is in chapter {own}", [("shipments", "goods_description", good)], fix_date(i, 0.4))
    for i in slices["late_filing"]:
        never = rng.random() < 0.1
        shipments.loc[i, "filed_at"] = pd.NaT if never else entry_date[i] + pd.Timedelta(days=int(rng.integers(6, 25)), hours=int(rng.integers(8, 18)))
        event("late_filing", "shipment", eid[i], ssid[i], entry_date[i] + day(6), "never filed" if never else "filed after the deadline")
    old = np.flatnonzero(entry_date <= end - day(200))
    for i in rng.choice(old, int(round(0.006 * n)), replace=False):
        purged = entry_date[i] + day(rng.integers(90, 190))
        shipments.loc[i, "records_purged_on"] = purged
        event("records_purged_early", "shipment", eid[i], ssid[i], purged, f"records purged {purged:%Y-%m-%d}",
              [("shipments", "records_purged_on", pd.NaT)], purged + day(rng.integers(20, 90)) if rng.random() < 0.3 else pd.NaT)

    products = products.drop(columns=[c for c in products.columns if c.startswith("_")])
    base = validate_records(suppliers, certificates, shipments, products)
    return SampleWorld(base=base, events=events, screening_list=sample_screening_list(), as_of=end, start=start)
