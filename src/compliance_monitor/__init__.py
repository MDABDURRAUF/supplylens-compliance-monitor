"""SupplyLens Compliance Monitor: configurable compliance checks, name screening for human review, and a findings workflow.

Decision support, not legal advice. The rules test what you configure them to test.
"""

__version__ = "0.1.0"

from . import evaluate  # noqa: E402
from .data import (DOCUMENTS, Records, SampleWorld, SchemaError, generate_sample, load_records, sample_screening_list,  # noqa: E402
                   save_records, validate_records)
from .learning import chapter_fit, declared_chapter, unit_value_outliers  # noqa: E402
from .monitor import daily_check, replay_sample  # noqa: E402
from .report import DISCLAIMER, build_audit_report, build_supplier_profile, write_html  # noqa: E402
from .rules import CATEGORIES, CHECKS, SEVERITIES, Rule, RuleSet, default_rules, run_checks  # noqa: E402
from .scores import ageing_table, headline, rule_summary, supplier_scores  # noqa: E402
from .screening import (POTENTIAL_MATCH, load_screening_list, name_similarity, normalise, screen_names,  # noqa: E402
                        validate_screening_list)
from .store import STATUSES, FindingStore  # noqa: E402

__all__ = [
    "CATEGORIES", "CHECKS", "DISCLAIMER", "DOCUMENTS", "FindingStore", "POTENTIAL_MATCH", "Records", "Rule", "RuleSet", "SEVERITIES",
    "STATUSES", "SampleWorld", "SchemaError", "ageing_table", "build_audit_report", "build_supplier_profile", "chapter_fit",
    "daily_check", "declared_chapter", "default_rules", "evaluate", "generate_sample", "headline", "load_records",
    "load_screening_list", "name_similarity", "normalise", "replay_sample", "rule_summary", "run_checks", "sample_screening_list",
    "save_records", "screen_names", "supplier_scores", "unit_value_outliers", "validate_records", "validate_screening_list", "write_html",
]
