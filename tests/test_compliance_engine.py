import unittest

from compliance_engine import ComplianceEngine, ThresholdRule, PresenceRule, ForbiddenClauseRule


class ComplianceEngineTests(unittest.TestCase):
    def test_missing_primary_field_is_not_evaluated(self):
        rule = ThresholdRule(
            rule_id="ratio_test",
            name="Deposit ratio",
            description="Security deposit should not exceed 3x rent",
            category="financial",
            threshold_type="ratio",
            threshold_value=3.0,
            field="deposit",
            severity="warning",
            domain="legal",
            comparison_field="rent",
        )

        result = rule.check([], {"rent": {"value": 1000, "source_snippet": "Rent 1000", "clause_number": "3"}})

        self.assertIsNone(result.passed)
        self.assertIn("cannot evaluate this rule", result.message.lower())

    def test_zero_comparison_value_is_not_evaluated(self):
        rule = ThresholdRule(
            rule_id="ratio_test",
            name="Deposit ratio",
            description="Security deposit should not exceed 3x rent",
            category="financial",
            threshold_type="ratio",
            threshold_value=3.0,
            field="deposit",
            severity="warning",
            domain="legal",
            comparison_field="rent",
        )

        clauses = [
            {"clause_number": "4", "clause_text": "Deposit is Rs. 1000"},
            {"clause_number": "3", "clause_text": "Rent is Rs. 0"},
        ]
        context = {
            "deposit": {"value": 1000, "source_snippet": "Deposit is Rs. 1000", "clause_number": "4"},
            "rent": {"value": 0, "source_snippet": "Rent is Rs. 0", "clause_number": "3"},
        }

        result = rule.check(clauses, context)

        self.assertIsNone(result.passed)
        self.assertIn("zero or invalid", result.message.lower())

    def test_all_unevaluated_rules_report_no_numeric_score(self):
        engine = ComplianceEngine()
        engine.rules = [
            ThresholdRule(
                rule_id="deposit_ratio",
                name="Deposit ratio",
                description="Security deposit should not exceed 3x rent",
                category="financial",
                threshold_type="ratio",
                threshold_value=3.0,
                field="deposit",
                severity="warning",
                domain="legal",
                comparison_field="rent",
            ),
            ThresholdRule(
                rule_id="rent_limit",
                name="Rent limit",
                description="Monthly rent should be reasonable",
                category="financial",
                threshold_type="max",
                threshold_value=200000,
                field="rent",
                severity="info",
                domain="legal",
            ),
        ]

        result = engine.evaluate([], domain="legal")

        self.assertIsNone(result["compliance_score"])
        self.assertEqual(result["overall_status"], "UNKNOWN")
        self.assertEqual(result["not_evaluated"], 2)

    def test_presence_rule_avoids_weak_substring_matches(self):
        rule = PresenceRule(
            rule_id="law_clause",
            name="Governing law",
            description="Requires governing law clause",
            category="mandatory",
            required_clauses=["law"],
            severity="critical",
            domain="legal",
        )

        clauses = [{"clause_number": "1", "clause_text": "This lawyer is not a legal authority."}]
        result = rule.check(clauses, {})

        self.assertFalse(result.passed)

    def test_forbidden_rule_collects_multiple_matches_in_same_clause(self):
        rule = ForbiddenClauseRule(
            rule_id="forbidden_terms",
            name="Forbidden terms",
            description="Avoid prohibited legal wording",
            category="fairness",
            forbidden_terms=["forfeit entire", "liquidated damages 2x"],
            severity="warning",
            domain="generic",
        )

        clauses = [{
            "clause_number": "7",
            "clause_text": "The tenant shall forfeit entire security deposit and liquidated damages 2x the amount."
        }]
        result = rule.check(clauses, {})

        self.assertFalse(result.passed)
        self.assertEqual(sorted(set(result.metadata["violations"])), ["forfeit entire", "liquidated damages 2x"])


if __name__ == "__main__":
    unittest.main()
