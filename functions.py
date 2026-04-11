"""Pure helpers and business logic for PDF text, clauses, rent, and deposit checks."""

import json
import re

from pypdf import PdfReader


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text


def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, max_chars=20000):
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def clean_llm_output(text):
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def extract_clauses_locally(chunk):
    clauses = []
    pattern = re.compile(
        r"(?<!\d)(\d{1,2})\.\s*([A-Z][A-Z\s&/\-]{2,80}?)(?::|\()",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(chunk or ""))

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
        body = (chunk[start:end] or "").strip()

        if len(body) < 40:
            continue

        title = re.sub(r"\s+", " ", (m.group(2) or "").strip())[:60]
        clauses.append(
            {
                "clause_number": str((m.group(1) or "").strip()),
                "clause_title": title if title else "Clause",
                "clause_text": body,
            }
        )

    return clauses


def safe_parse(text):
    text = clean_llm_output(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clean_clauses(clauses):
    unique = {}

    for c in clauses:
        if not isinstance(c, dict):
            continue

        text = str(c.get("clause_text") or "").strip()
        num = str(c.get("clause_number") or "").strip()

        if len(text) < 40:
            continue

        if num not in unique or len(text) > len(unique[num]["clause_text"]):
            unique[num] = c

    return list(unique.values())


def check_rent_limit(rent_amount):
    limit = 80000

    if rent_amount > limit:
        return {
            "status": "VIOLATION",
            "message": f"Rent {rent_amount} exceeds limit {limit}",
        }
    return {
        "status": "OK",
        "message": f"Rent {rent_amount} is within limit",
    }


def is_rent_clause(clause):
    title = (clause.get("clause_title") or "").lower()
    text = (clause.get("clause_text") or "").lower()

    return "rent" in title or "rent" in text


def extract_rent_with_regex(text):
    pattern = r"(?i)(?:rent|monthly rent)[^\d]{0,15}(?:rs\.?|inr|₹)?\s*([\d,]{4,})"

    matches = re.findall(pattern, text or "")

    values = []
    for m in matches:
        try:
            values.append(int(m.replace(",", "")))
        except ValueError:
            continue

    if values:
        return max(values)

    return None


def analyze_rent_clause(clause):
    rent = clause.get("rent")

    if rent is None:
        return None

    result = check_rent_limit(rent)

    return {
        "clause_number": clause["clause_number"],
        "rent": rent,
        "status": result["status"],
        "message": result["message"],
    }


def check_deposit_fairness(rent, deposit):
    if rent is None or deposit is None:
        return {
            "status": "UNKNOWN",
            "message": "Insufficient data to evaluate",
        }

    ratio = deposit / rent

    if ratio > 3:
        return {
            "status": "UNFAIR",
            "message": f"Deposit is {ratio:.1f}x of rent (Too high)",
        }
    if 1 <= ratio <= 3:
        return {
            "status": "FAIR",
            "message": f"Deposit is {ratio:.1f}x of rent (Acceptable)",
        }
    return {
        "status": "SUSPICIOUS",
        "message": f"Deposit is unusually low ({ratio:.1f}x)",
    }


def extract_rent_and_deposit(text):
    text = (text or "").lower()

    rent_match = re.search(r"rent[^\d]{0,15}(\d{4,})", text)

    deposit_match = None
    patterns = [
        r"security\s+deposit[^\d]{0,100}(?:rs\.?|₹|inr)?\s*([\d,]{3,})",
        r"(?:security|deposit)[^\d]{0,100}(?:rs\.?|₹|inr)\s*([\d,]{3,})",
        r"(?:rs\.?|₹)\s*([\d,]{3,}).*?(?:security|deposit)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            deposit_match = match
            break

    rent = int(rent_match.group(1)) if rent_match else None
    deposit = (
        int(deposit_match.group(1).replace(",", "")) if deposit_match else None
    )

    return rent, deposit


def analyze_deposit_clause(clause):
    text = clause.get("clause_text")

    rent, deposit = extract_rent_and_deposit(text)

    result = check_deposit_fairness(rent, deposit)

    return {
        "clause_number": clause["clause_number"],
        "rent": rent,
        "deposit": deposit,
        "status": result["status"],
        "message": result["message"],
    }
