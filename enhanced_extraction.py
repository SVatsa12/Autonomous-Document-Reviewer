"""Enhanced rent/deposit extraction with better pattern matching."""

import re


def extract_rent_enhanced(clauses):
    """
    Extract rent amount from all clauses using multiple patterns.
    Returns (rent_amount, rent_clause) or (None, None).
    """
    rent_patterns = [
        # Pattern: monthly rent of Rs. 1,00,000
        r"(?:monthly\s+rent|rent\s+per\s+month|rent\s+payable)[^\d]{0,20}(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        # Pattern: rent of Rs. 1,00,000 per month
        r"(?:rent\s+of|rent\s+is)[^\d]{0,15}(?:rs\.?|inr|₹)?\s*([\d,]{4,12})[^\d]{0,10}(?:per\s+month|monthly)?",
        # Pattern: Rs. 1,00,000 as monthly rent
        r"(?:rs\.?|inr|₹)\s*([\d,]{4,12})[^\d]{0,20}(?:as\s+monthly\s+rent|per\s+month)",
        # Simple: rent = Rs. 100000
        r"rent\s*[=:]\s*(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        # Pattern: pay rent of Rs. 100000
        r"(?:pay|payable)\s+(?:a\s+)?(?:monthly\s+)?rent\s+of\s+(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        # Pattern: rent shall be Rs.
        r"rent\s+shall\s+be\s+(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
    ]

    best_rent = None
    best_clause = None

    for clause in clauses:
        text = clause.get("clause_text", "").lower()
        full_text = clause.get("clause_text", "")

        # Skip if not a rent-related clause
        title = (clause.get("clause_title") or "").lower()
        if "rent" not in title and "rent" not in text:
            continue

        # Try each pattern
        for pattern in rent_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the largest value (usually the main rent)
                for m in matches:
                    try:
                        val = int(m.replace(",", "").replace(" ", ""))
                        # Sanity check: rent should be between 1,000 and 10,000,000
                        if 1000 <= val <= 10000000:
                            if best_rent is None or val > best_rent:
                                best_rent = val
                                best_clause = clause
                    except ValueError:
                        continue
                if best_rent:
                    break
        if best_rent:
            break

    return best_rent, best_clause


def extract_deposit_enhanced(clauses):
    """
    Extract security deposit from all clauses.
    Returns (deposit_amount, deposit_clause) or (None, None).
    """
    deposit_patterns = [
        r"security\s+deposit[^\d]{0,50}(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        r"(?:rs\.?|inr|₹)\s*([\d,]{4,12})[^\d]{0,30}(?:as\s+security\s+deposit|security\s+deposit)",
        r"deposit\s+of\s+(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        r"pay\s+(?:a\s+)?(?:security\s+)?deposit\s+of\s+(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
        r"deposit\s+[=:]\s*(?:rs\.?|inr|₹)?\s*([\d,]{4,12})",
    ]

    best_deposit = None
    best_clause = None

    for clause in clauses:
        text = clause.get("clause_text", "").lower()
        full_text = clause.get("clause_text", "")

        # Check if this is a security deposit clause
        if "security deposit" not in text and "deposit" not in text:
            continue

        for pattern in deposit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for m in matches:
                    try:
                        val = int(m.replace(",", "").replace(" ", ""))
                        # Sanity check: deposit typically 10,000 - 50,000,000
                        if 10000 <= val <= 50000000:
                            if best_deposit is None or val > best_deposit:
                                best_deposit = val
                                best_clause = clause
                    except ValueError:
                        continue
                if best_deposit:
                    break
        if best_deposit:
            break

    return best_deposit, best_clause


def analyze_deposit_fairness_enhanced(rent, deposit):
    """Enhanced deposit fairness analysis."""
    if rent is None or deposit is None:
        return {
            "status": "UNKNOWN",
            "message": "Insufficient data to evaluate - need both rent and deposit amounts",
        }

    ratio = deposit / rent

    # More detailed thresholds
    if ratio > 6:
        return {
            "status": "HIGHLY UNFAIR",
            "message": f"Deposit is {ratio:.1f}x rent (exceeds 6x - likely illegal in most jurisdictions)",
            "legal_note": "Most rent control acts limit deposits to 2-3 months rent"
        }
    if ratio > 3:
        return {
            "status": "UNFAIR",
            "message": f"Deposit is {ratio:.1f}x rent (exceeds typical 3x limit)",
            "legal_note": "May violate rent control regulations"
        }
    if ratio >= 2 and ratio <= 3:
        return {
            "status": "ACCEPTABLE/NORMAL",
            "message": f"Deposit is {ratio:.1f}x rent (within standard 2-3 month range)",
            "legal_note": "Common and generally legal"
        }
    if ratio >= 1 and ratio < 2:
        return {
            "status": "REASONABLE",
            "message": f"Deposit is {ratio:.1f}x rent (within 1-2 month range)",
            "legal_note": "Fair and reasonable"
        }
    if ratio < 1:
        return {
            "status": "SUSPICIOUS/LOW",
            "message": f"Deposit is unusually low at {ratio:.1f}x rent",
            "legal_note": "Verify if all deposit components are captured"
        }
    return {
        "status": "BORDERLINE",
        "message": f"Deposit is {ratio:.1f}x rent"
    }
