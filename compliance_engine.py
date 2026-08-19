"""Configurable compliance rule engine for multi-domain compliance checking.

Design goals (v2):
  1. Field extraction is a separate, shared step -> fixes the ratio-rule
     context bug (rules used to get called with context={}).
  2. Extraction understands Indian currency shorthand (lakh/crore) and
     uses word-boundaried patterns.
  3. Domains (rental, hr, banking, ...) are defined in config, not code,
     so adding a new domain doesn't require editing this file.
  4. Severity-weighted compliance score, and a clear split between
     "failed a check" vs "could not run a check" in the summary.
  5. RuleResult/rule serialization doesn't silently drop subclass fields.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

try:
    from vector_db import ClauseVectorDB
except ImportError:  # optional dependency; semantic presence matching degrades gracefully
    ClauseVectorDB = None

try:
    from llm_ops import tracked_generate, MODEL_NAME
except ImportError:  # optional dependency; SemanticComplianceRule requires a client anyway
    tracked_generate = None
    MODEL_NAME = None


# ============================================================================
# FIELD EXTRACTION (shared by every rule type — this is what was missing)
# ============================================================================

# multiplier for Indian shorthand currency units
_UNIT_MULTIPLIERS = {
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
}

_NUMBER_UNIT_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|crores|cr)?",
    re.IGNORECASE,
)

# Pre-compile the keyword-based field patterns once instead of creating a new
# regex inside every clause loop.
_FIELD_PATTERN_CACHE: Dict[str, re.Pattern[str]] = {}


def _compile_field_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile a keyword-based numeric extraction pattern for a single field."""
    key = keyword.lower()
    if key not in _FIELD_PATTERN_CACHE:
        _FIELD_PATTERN_CACHE[key] = re.compile(
            rf"\b{re.escape(keyword)}\b[^\d]{{0,20}}([\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|lac|lacs|crore|crores|cr)?)",
            re.IGNORECASE,
        )
    return _FIELD_PATTERN_CACHE[key]


def parse_amount(fragment: str) -> Optional[float]:
    """Parse a numeric amount from text, handling lakh/crore shorthand."""
    m = _NUMBER_UNIT_RE.search(fragment)
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit in _UNIT_MULTIPLIERS:
        num *= _UNIT_MULTIPLIERS[unit]
    return num


@dataclass
class ExtractedField:
    """A single field value pulled from a clause, with provenance."""
    value: float
    clause_number: str
    source_snippet: str


def _normalize_extracted_field(value: Any) -> Optional[ExtractedField]:
    """Accept both ExtractedField objects and legacy dict payloads.

    This keeps older callers and tests compatible while preserving the new
    structured extraction path used by the engine.
    """
    if value is None:
        return None
    if isinstance(value, ExtractedField):
        return value
    if isinstance(value, dict):
        if "value" not in value:
            return None
        return ExtractedField(
            value=float(value["value"]),
            clause_number=str(value.get("clause_number", "N/A")),
            source_snippet=str(value.get("source_snippet", value.get("value", ""))),
        )
    return None


class FieldExtractor:
    """Extracts named fields from clause text using domain-configured patterns.

    field_specs example:
        {
            "rent": {"keywords": ["rent", "monthly rent"], "aggregation": "max"},
            "deposit": {"keywords": ["security deposit", "deposit"], "aggregation": "max"},
        }
    """

    def __init__(self, field_specs: Dict[str, Dict]):
        self.field_specs = field_specs or {}

    def extract(self, clauses: List[Dict]) -> Dict[str, ExtractedField]:
        extracted: Dict[str, ExtractedField] = {}
        for field_name, spec in self.field_specs.items():
            keywords = spec.get("keywords", [field_name])
            candidates: List[ExtractedField] = []

            for clause in clauses:
                # 1. structured field on the clause dict wins if present
                if field_name in clause and clause[field_name] is not None:
                    try:
                        candidates.append(ExtractedField(
                            value=float(clause[field_name]),
                            clause_number=clause.get("clause_number", "N/A"),
                            source_snippet=str(clause[field_name]),
                        ))
                        continue
                    except (TypeError, ValueError):
                        pass

                # 2. else regex-scan clause text for any of the keywords
                text = clause.get("clause_text", "")
                for kw in keywords:
                    pattern = _compile_field_keyword_pattern(kw)
                    m = pattern.search(text)
                    if m:
                        val = parse_amount(m.group(1))
                        if val is not None:
                            candidates.append(ExtractedField(
                                value=val,
                                clause_number=clause.get("clause_number", "N/A"),
                                source_snippet=m.group(0).strip(),
                            ))
                            break  # one match per clause per field is enough

            if not candidates:
                continue

            agg = spec.get("aggregation", "max")
            if agg == "max":
                chosen = max(candidates, key=lambda c: c.value)
            elif agg == "min":
                chosen = min(candidates, key=lambda c: c.value)
            else:  # 'first'
                chosen = candidates[0]

            extracted[field_name] = chosen

        return extracted


# ============================================================================
# RULE DEFINITIONS
# ============================================================================

class ComplianceRule(ABC):
    """Base class for compliance rules."""

    def __init__(self, rule_id: str, name: str, description: str,
                 category: str, severity: str, domain: str = "generic"):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity  # 'critical', 'warning', 'info'
        self.domain = domain

    @abstractmethod
    def check(self, clauses: List[Dict], context: Dict[str, ExtractedField], client=None) -> "RuleResult":
        """Execute rule check against clauses. `context` holds pre-extracted fields."""
        raise NotImplementedError

    def required_fields(self) -> List[str]:
        """Field names this rule needs the engine to extract. Override as needed."""
        return []

    def to_dict(self) -> Dict:
        """Serialize including subclass-specific attributes (fixes asdict() dropping them)."""
        return {k: v for k, v in vars(self).items()}


@dataclass
class RuleResult:
    """Result of a rule check."""
    rule_id: str
    passed: Optional[bool]  # True / False / None ("could not evaluate")
    message: str
    severity: str
    evidence: Optional[List[str]] = None
    recommendation: Optional[str] = None
    clause_refs: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    not_evaluated_reason: Optional[str] = None  # set when passed is None

    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


# ============================================================================
# THRESHOLD-BASED RULES
# ============================================================================

class ThresholdRule(ComplianceRule):
    """Rule that checks numeric thresholds, including cross-field ratios."""

    def __init__(self, rule_id, name, description, category, threshold_type,
                 threshold_value, field, severity="warning", domain="generic",
                 comparison_field: Optional[str] = None):
        super().__init__(rule_id, name, description, category, severity, domain)
        self.threshold_type = threshold_type  # 'max' | 'min' | 'range' | 'ratio'
        self.threshold_value = threshold_value
        self.field = field
        self.comparison_field = comparison_field

    def required_fields(self) -> List[str]:
        fields = [self.field]
        if self.comparison_field:
            fields.append(self.comparison_field)
        return fields

    def check(self, clauses: List[Dict], context: Dict[str, ExtractedField], client=None) -> RuleResult:
        primary = _normalize_extracted_field(context.get(self.field))
        if primary is None:
            return RuleResult(
                rule_id=self.rule_id, passed=None,
                message=f"No {self.field} found in contract clauses; cannot evaluate this rule",
                severity=self.severity,
                not_evaluated_reason=f"missing primary field '{self.field}'",
                recommendation=f"Ensure a clause stating {self.field} is present",
            )

        value = primary.value

        if self.threshold_type == "ratio":
            comp = _normalize_extracted_field(context.get(self.comparison_field)) if self.comparison_field else None
            if comp is None:
                return RuleResult(
                    rule_id=self.rule_id, passed=None,
                    message=f"Cannot compute ratio: {self.comparison_field} not found in contract",
                    severity=self.severity,
                    not_evaluated_reason=f"missing comparison field '{self.comparison_field}'",
                    recommendation=f"Ensure a clause stating {self.comparison_field} is present",
                )
            passed, ratio = self._evaluate_ratio(value, comp.value)
            if passed is None:
                return RuleResult(
                    rule_id=self.rule_id, passed=None,
                    message=f"Cannot compute {self.field}/{self.comparison_field} ratio because {self.comparison_field} is zero or invalid",
                    severity=self.severity,
                    not_evaluated_reason=f"comparison field '{self.comparison_field}' is zero or invalid",
                    recommendation=f"Confirm the {self.comparison_field} value is valid before evaluating this rule",
                    evidence=[
                        f"{self.field}: {value} ({primary.source_snippet})",
                        f"{self.comparison_field}: {comp.value} ({comp.source_snippet})",
                    ],
                    clause_refs=[primary.clause_number, comp.clause_number],
                    metadata={"value": value, "comparison_value": comp.value, "threshold": self.threshold_value},
                )
            evidence = [f"{self.field}: {value} ({primary.source_snippet})",
                        f"{self.comparison_field}: {comp.value} ({comp.source_snippet})",
                        f"ratio: {ratio:.2f}x"]
            clause_refs = [primary.clause_number, comp.clause_number]
            message = (f"{self.field}/{self.comparison_field} ratio of {ratio:.2f}x "
                       f"{'complies with' if passed else 'violates'} "
                       f"{self.description} (max allowed: {self.threshold_value:.1f}x)")
        else:
            passed = self._evaluate_threshold(value)
            evidence = [f"{self.field}: {value} ({primary.source_snippet})"]
            clause_refs = [primary.clause_number]
            message = f"{self.field} of {value} {'complies with' if passed else 'violates'} {self.description}"
            if not passed:
                if self.threshold_type == "max":
                    message += f" (max allowed: {self.threshold_value})"
                elif self.threshold_type == "min":
                    message += f" (min required: {self.threshold_value})"
                elif self.threshold_type == "range":
                    message += f" (allowed range: {self.threshold_value})"

        return RuleResult(
            rule_id=self.rule_id, passed=passed, message=message, severity=self.severity,
            evidence=evidence, clause_refs=clause_refs,
            recommendation=self._get_recommendation(passed),
            metadata={"value": value, "threshold": self.threshold_value},
        )

    def _evaluate_threshold(self, value: float) -> bool:
        if self.threshold_type == "max":
            return value <= self.threshold_value
        elif self.threshold_type == "min":
            return value >= self.threshold_value
        elif self.threshold_type == "range":
            low, high = self.threshold_value
            return low <= value <= high
        return False

    def _evaluate_ratio(self, value: float, comp_value: float) -> Tuple[Optional[bool], Optional[float]]:
        if comp_value is None or comp_value <= 0:
            return None, None
        ratio = value / comp_value
        return ratio <= self.threshold_value, ratio

    def _get_recommendation(self, passed: bool) -> Optional[str]:
        if passed:
            return None
        if self.threshold_type == "max":
            return f"Reduce {self.field} to be within {self.threshold_value}"
        elif self.threshold_type == "min":
            return f"Increase {self.field} to at least {self.threshold_value}"
        elif self.threshold_type == "ratio":
            return f"Ensure {self.field} does not exceed {self.threshold_value:.1f}x of {self.comparison_field}"
        return None


# ============================================================================
# PRESENCE / ABSENCE RULES
# ============================================================================

class PresenceRule(ComplianceRule):
    """Rule that checks for presence of mandatory clause language.

    match_mode:
      'substring' - plain case-insensitive substring match (fast, exact)
      'semantic'  - embedding similarity via ClauseVectorDB, falls back to
                    substring if no vector db is configured
    """

    def __init__(self, rule_id, name, description, category, required_clauses,
                 severity="warning", domain="generic", match_mode: str = "substring",
                 semantic_threshold: float = 0.75, vector_db=None, semantic_match: Optional[bool] = None):
        super().__init__(rule_id, name, description, category, severity, domain)
        self.required_clauses = required_clauses
        if semantic_match is not None:
            self.match_mode = "semantic" if semantic_match else "substring"
        else:
            self.match_mode = match_mode
        self.semantic_threshold = semantic_threshold
        self.vector_db = vector_db

    def check(self, clauses: List[Dict], context: Dict[str, ExtractedField]) -> RuleResult:
        missing, found_clauses = [], []

        for required in self.required_clauses:
            match = self._find_match(required, clauses)
            if match is None:
                missing.append(required)
            else:
                found_clauses.append(match)

        passed = len(missing) == 0
        evidence = [f"Found: {', '.join(found_clauses) if found_clauses else 'None'}"]
        message = ("All required clauses present" if passed else
                   f"Missing required clauses: {', '.join(missing)}")

        return RuleResult(
            rule_id=self.rule_id, passed=passed, message=message, severity=self.severity,
            evidence=evidence, clause_refs=found_clauses,
            recommendation=f"Add missing clauses: {', '.join(missing)}" if missing else None,
        )

    def _find_match(self, required: str, clauses: List[Dict]) -> Optional[str]:
        if self.match_mode == "semantic" and self.vector_db is not None:
            hit = self.vector_db.search(required, clauses, threshold=self.semantic_threshold)
            if hit:
                return hit.get("clause_number", "N/A")
            return None

        pattern = re.compile(rf"(?<!\w){re.escape(required.lower())}(?!\w)", re.IGNORECASE)
        for c in clauses:
            haystack = f"{c.get('clause_title', '')} {c.get('clause_text', '')}"
            if pattern.search(haystack):
                return c.get("clause_number", "N/A")
        return None


class ForbiddenClauseRule(ComplianceRule):
    """Rule that checks for forbidden/disallowed clauses."""

    def __init__(self, rule_id, name, description, category, forbidden_terms,
                 severity="critical", domain="generic", case_sensitive: bool = False):
        super().__init__(rule_id, name, description, category, severity, domain)
        self.forbidden_terms = forbidden_terms
        self.case_sensitive = case_sensitive

    def check(self, clauses: List[Dict], context: Dict[str, ExtractedField]) -> RuleResult:
        found_violations, clause_refs = [], []

        for clause in clauses:
            text = clause.get("clause_text", "")
            haystack = text if self.case_sensitive else text.lower()
            terms = self.forbidden_terms if self.case_sensitive else [t.lower() for t in self.forbidden_terms]

            for term in terms:
                if term in haystack:
                    found_violations.append(term)
                    clause_refs.append(clause.get("clause_number", "N/A"))

        passed = len(found_violations) == 0
        evidence = [f"Forbidden terms found: {', '.join(sorted(set(found_violations)))}"] if found_violations else []
        message = ("No forbidden clauses detected" if passed else
                   f"Found {len(set(found_violations))} forbidden term(s)")

        return RuleResult(
            rule_id=self.rule_id, passed=passed, message=message, severity=self.severity,
            evidence=evidence, clause_refs=clause_refs,
            recommendation="Remove or revise clauses containing forbidden terms" if not passed else None,
            metadata={"violations": sorted(set(found_violations))},
        )


# ============================================================================
# SEMANTIC COMPLIANCE RULES (LLM-based)
# ============================================================================

class SemanticComplianceRule(ComplianceRule):
    """Rule that uses an LLM for semantic compliance reasoning."""

    def __init__(self, rule_id, name, description, category, regulation_text,
                 severity="warning", domain="generic", requires_llm: bool = True,
                 max_clauses: int = 10):
        super().__init__(rule_id, name, description, category, severity, domain)
        self.regulation_text = regulation_text
        self.requires_llm = requires_llm
        self.max_clauses = max_clauses

    def check(self, clauses: List[Dict], context: Dict[str, ExtractedField], client=None) -> RuleResult:
        if self.requires_llm and client is None:
            return RuleResult(
                rule_id=self.rule_id, passed=None,
                message="LLM client required for semantic compliance check",
                severity=self.severity,
                not_evaluated_reason="no LLM client provided",
                recommendation="Provide LLM client or disable semantic checks",
            )

        relevant_clauses = clauses[: self.max_clauses]
        context_text = "\n\n".join(
            f"Clause {c.get('clause_number', 'N/A')}: {c.get('clause_text', '')}"
            for c in relevant_clauses
        )

        prompt = f"""You are a compliance auditor. Evaluate the following contract clauses against the regulation.

REGULATION:
{self.regulation_text}

CONTRACT CLAUSES:
{context_text}

TASK: Determine if the contract complies with the regulation. Consider:
1. Does the contract meet or exceed the regulatory requirements?
2. Are there any conflicts or gaps?
3. Is the language clear and unambiguous?

Provide your analysis in JSON format:
{{
    "complies": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "gaps": ["list any missing requirements"],
    "conflicts": ["list any conflicts with regulation"]
}}

Your response (JSON only):"""

        try:
            response = tracked_generate(client, prompt, label=f"compliance check: {self.rule_id}")
            try:
                result = json.loads(response.text.strip())
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse LLM response as JSON")

            passed = result.get("complies", False)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            gaps = result.get("gaps", [])
            conflicts = result.get("conflicts", [])

            evidence = [f"Confidence: {confidence:.1%}", f"Reasoning: {reasoning}"]
            if gaps:
                evidence.append(f"Gaps: {', '.join(gaps)}")
            if conflicts:
                evidence.append(f"Conflicts: {', '.join(conflicts)}")

            if not passed and confidence > 0.8:
                actual_severity = "critical"
            elif not passed and confidence > 0.5:
                actual_severity = "warning"
            else:
                actual_severity = self.severity

            return RuleResult(
                rule_id=self.rule_id, passed=passed,
                message=f"Compliance check: {'PASS' if passed else 'FAIL'} (confidence: {confidence:.1%})",
                severity=actual_severity, evidence=evidence,
                clause_refs=[c.get("clause_number") for c in relevant_clauses],
                recommendation=self._generate_recommendation(gaps, conflicts),
                metadata={"confidence": confidence, "gaps": gaps, "conflicts": conflicts},
            )
        except Exception as e:
            return RuleResult(
                rule_id=self.rule_id, passed=None,
                message=f"LLM check failed: {str(e)}", severity=self.severity,
                not_evaluated_reason=str(e),
                recommendation="Retry with a valid LLM client",
            )

    def _generate_recommendation(self, gaps, conflicts) -> Optional[str]:
        if not gaps and not conflicts:
            return None
        suggestions = []
        if gaps:
            suggestions.append(f"Add: {', '.join(gaps[:3])}")
        if conflicts:
            suggestions.append(f"Revise: {', '.join(conflicts[:3])}")
        return " | ".join(suggestions)


# ============================================================================
# RULE ENGINE
# ============================================================================

_RULE_CLASSES = {
    "threshold": ThresholdRule,
    "presence": PresenceRule,
    "forbidden": ForbiddenClauseRule,
    "semantic": SemanticComplianceRule,
}

# Internal score weights used to combine rule outcomes into a single index.
# These are NOT ISO/IEC rating levels; they are only a project-local weighting
# scheme to emphasize critical issues while still preserving warning/info signals.
# critical: 3.0, warning: 1.0, info: 0.5
_SEVERITY_WEIGHTS = {"critical": 3.0, "warning": 1.0, "info": 0.5}


class ComplianceEngine:
    """Configurable, domain-pluggable compliance rule engine."""

    def __init__(self, rules_file: Optional[Union[str, Path]] = None, vector_db=None):
        self.rules: List[ComplianceRule] = []
        self.field_specs: Dict[str, Dict] = {}  # merged across loaded domain configs
        self.vector_db = vector_db

        if rules_file:
            self.load_rules_from_file(rules_file)
        else:
            self._load_default_rules()

    # -- loading -------------------------------------------------------

    def _load_default_rules(self):
        """Built-in rental-domain rules (used when no config file is supplied)."""
        self.field_specs.update({
            "rent": {"keywords": ["rent", "monthly rent"], "aggregation": "max"},
            "deposit": {"keywords": ["security deposit", "deposit"], "aggregation": "max"},
            "interest_rate": {"keywords": ["interest rate", "interest"], "aggregation": "max"},
        })

        self.add_rule(ThresholdRule(
            rule_id="legal_deposit_ratio", name="Security Deposit Ratio",
            description="Security deposit should not exceed 3 months rent",
            category="financial", threshold_type="ratio", threshold_value=3.0,
            field="deposit", severity="warning", domain="legal", comparison_field="rent",
        ))
        self.add_rule(ThresholdRule(
            rule_id="legal_rent_limit", name="Rent Reasonableness",
            description="Monthly rent should be reasonable (not excessive)",
            category="financial", threshold_type="max", threshold_value=200000,
            field="rent", severity="info", domain="legal",
        ))
        self.add_rule(PresenceRule(
            rule_id="legal_mandatory_1", name="Governing Law Clause",
            description="Contract must specify governing law/jurisdiction",
            category="mandatory", required_clauses=["jurisdiction", "governing law", "applicable law"],
            severity="critical", domain="legal",
        ))
        self.add_rule(PresenceRule(
            rule_id="legal_mandatory_2", name="Termination Clause",
            description="Contract must include termination conditions",
            category="mandatory", required_clauses=["termination", "notice period", "vacate"],
            severity="critical", domain="legal",
        ))
        self.add_rule(PresenceRule(
            rule_id="legal_deposit_clause", name="Security Deposit Clause",
            description="Contract must specify deposit amount and terms",
            category="mandatory", required_clauses=["security deposit", "deposit"],
            severity="critical", domain="legal",
        ))
        self.add_rule(ForbiddenClauseRule(
            rule_id="generic_penalties", name="Excessive Penalties",
            description="Avoid punitive penalty clauses", category="fairness",
            forbidden_terms=["penalty of rs", "forfeit entire", "liquidated damages 2x"],
            severity="warning", domain="generic",
        ))

    def load_rules_from_file(self, filepath: Union[str, Path]):
        """Load one domain's rules + field_specs from a JSON config file."""
        with open(filepath, "r") as f:
            config = json.load(f)

        self.field_specs.update(config.get("field_specs", {}))

        for rule_config in config.get("rules", []):
            rule_config = dict(rule_config)
            rule_type = rule_config.pop("type")
            rule_cls = _RULE_CLASSES.get(rule_type)
            if rule_cls is None:
                raise ValueError(f"Unknown rule type '{rule_type}' in {filepath}")
            if rule_cls is PresenceRule and self.vector_db is not None:
                rule_config.setdefault("vector_db", self.vector_db)
            if rule_cls is PresenceRule and "semantic_match" in rule_config:
                rule_config.setdefault("match_mode", "semantic" if rule_config["semantic_match"] else "substring")
            self.add_rule(rule_cls(**rule_config))

    def add_rule(self, rule: ComplianceRule):
        self.rules.append(rule)

    def get_rules_by_domain(self, domain: str) -> List[ComplianceRule]:
        return [r for r in self.rules if r.domain == domain or r.domain == "generic"]

    def get_rules_by_category(self, category: str) -> List[ComplianceRule]:
        return [r for r in self.rules if r.category == category]

    # -- evaluation ------------------------------------------------------

    def evaluate(self, clauses: List[Dict], domain: Optional[str] = None, client=None) -> Dict[str, Any]:
        applicable_rules = self.get_rules_by_domain(domain) if domain else self.rules

        # 1. Extract every field any applicable rule needs, ONCE, up front.
        needed_fields = {f for r in applicable_rules for f in r.required_fields()}
        specs_needed = {f: self.field_specs.get(f, {"keywords": [f], "aggregation": "max"})
                         for f in needed_fields}
        context = FieldExtractor(specs_needed).extract(clauses)

        # 2. Run every rule against the SAME context (this is the fix —
        #    previously non-LLM rules always received {}).
        results: List[RuleResult] = []
        for rule in applicable_rules:
            if isinstance(rule, SemanticComplianceRule):
                result = rule.check(clauses, context, client)
            else:
                result = rule.check(clauses, context)
            results.append(result)

        # 3. Summarize.
        passed = sum(1 for r in results if r.passed is True)
        failed = [r for r in results if r.passed is False]
        not_evaluated = [r for r in results if r.passed is None]
        critical_failed = [r for r in failed if r.severity == "critical"]

        evaluated = [r for r in results if r.passed is not None]
        weighted_total = sum(_SEVERITY_WEIGHTS.get(r.severity, 1.0) for r in evaluated)
        weighted_passed = sum(_SEVERITY_WEIGHTS.get(r.severity, 1.0) for r in evaluated if r.passed is True)
        compliance_score = (weighted_passed / weighted_total * 100) if weighted_total > 0 else None

        if critical_failed:
            overall_status = "FAIL"
        elif failed:
            overall_status = "WARNING"
        elif not_evaluated and not any(r.passed is True for r in results):
            overall_status = "UNKNOWN"
        elif not_evaluated:
            overall_status = "WARNING"
        elif compliance_score is None:
            overall_status = "UNKNOWN"
        else:
            overall_status = "PASS"

        return {
            "domain": domain or "generic",
            "total_rules": len(results),
            "passed": passed,
            "failed": len(failed),
            "not_evaluated": len(not_evaluated),
            "compliance_score": round(compliance_score, 1) if compliance_score is not None else None,
            "critical_issues": [r.message for r in critical_failed],
            "not_evaluated_reasons": [
                {"rule_id": r.rule_id, "reason": r.not_evaluated_reason} for r in not_evaluated
            ],
            "overall_status": overall_status,
            "extracted_context": {k: v.value for k, v in context.items()},
            "rule_results": [r.to_dict() for r in results],
        }


def load_rules(rules_file: str) -> ComplianceEngine:
    engine = ComplianceEngine()
    engine.load_rules_from_file(rules_file)
    return engine


default_engine = ComplianceEngine()