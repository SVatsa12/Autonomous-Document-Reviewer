"""Offline query mode - works without LLM API by using template-based answers."""

import sys
from pathlib import Path
from vector_db import ClauseVectorDB
from enhanced_extraction import extract_rent_enhanced, extract_deposit_enhanced

try:
    from google import genai
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


def offline_query(vector_db, query):
    """
    Answer queries without LLM using retrieved documents + templates.
    Returns formatted answer based on search results.
    """
    query_lower = query.lower()

    # Extract rent and deposit from contract
    all_clauses = list(vector_db.documents.values())
    contract_clauses = [d for d in all_clauses if d['metadata'].get('source') == 'contract_clause']
    contract_clauses_data = [{'clause_number': d['metadata'].get('clause_number', 'N/A'),
                              'clause_title': d['metadata'].get('clause_title', 'N/A'),
                              'clause_text': d['text']} for d in contract_clauses]

    rent, rent_clause = extract_rent_enhanced(contract_clauses_data)
    deposit, deposit_clause = extract_deposit_enhanced(contract_clauses_data)

    # Search for relevant documents
    # We'll simulate search by keyword matching since no LLM client
    results = vector_db.search(None, query, top_k=8)

    # Separate by source
    contract_results = [r for r in results if r['metadata'].get('source') == 'contract_clause']
    reg_results = [r for r in results if r['metadata'].get('source') == 'regulation']

    # Build answer based on query type
    if any(word in query_lower for word in ['security deposit', 'deposit amount', 'deposit refund']):
        return answer_security_deposit(query, deposit, deposit_clause, contract_results, reg_results, rent)
    elif any(word in query_lower for word in ['rent', 'monthly rent', 'rent amount']):
        return answer_rent(query, rent, rent_clause, contract_results, reg_results)
    elif any(word in query_lower for word in ['maintenance', 'repair', 'responsibility', 'who pays']):
        return answer_maintenance(query, contract_results, reg_results)
    elif any(word in query_lower for word in ['sublet', 'sublease', 'subletting']):
        return answer_subletting(query, contract_results, reg_results)
    elif any(word in query_lower for word in ['termination', 'notice period', 'vacate', 'early termination']):
        return answer_termination(query, contract_results, reg_results)
    elif any(word in query_lower for word in ['rent increase', 'increase rent', 'hike']):
        return answer_rent_increase(query, contract_results, reg_results)
    else:
        return answer_general(query, contract_results, reg_results)


def format_clause_ref(clause):
    """Format clause reference with number and title."""
    num = clause.get('clause_number', 'N/A')
    title = clause.get('clause_title') or clause.get('title', '')
    if title and title != num:
        return f"Clause {num} ({title})"
    return f"Clause {num}"


def answer_security_deposit(query, deposit, deposit_clause, contract_results, reg_results, rent):
    lines = []
    lines.append("SECURITY DEPOSIT ANALYSIS")
    lines.append("=" * 60)

    if deposit:
        lines.append(f"\nDEPOSIT AMOUNT: Rs {deposit:,}")
        if deposit_clause:
            lines.append(f"Source: {format_clause_ref(deposit_clause)}")
            lines.append(f"Text: {deposit_clause.get('clause_text','')[:150]}...")

        lines.append(f"\nLEGAL ASSESSMENT:")
        if rent:
            ratio = deposit / rent
            lines.append(f"  Deposit-to-rent ratio: {ratio:.1f}x")
            if ratio > 3:
                lines.append(f"  Status: EXCESSIVE (exceeds typical 2-3x limit)")
                lines.append(f"  Legal note: Rent Control Acts often limit deposits to 2-3 months' rent")
            elif 1 <= ratio <= 3:
                lines.append(f"  Status: NORMAL (within 1-3x standard range)")
            else:
                lines.append(f"  Status: UNUSUALLY LOW")

        # Find refund terms
        refund_clauses = [c for c in contract_results if 'refund' in c.get('text','').lower()]
        if refund_clauses:
            lines.append("\nREFUND TERMS:")
            for c in refund_clauses[:2]:
                lines.append(f"  • {format_clause_ref(c)}: {c.get('text','')[:120]}...")

        # Check regulations
        if reg_results:
            lines.append("\nLEGAL REGULATIONS:")
            for r in reg_results[:3]:
                reg_id = r['metadata'].get('regulation_id', 'N/A')
                title = r['metadata'].get('title', 'N/A')
                lines.append(f"  • [{reg_id}] {title}:")
                lines.append(f"    {r.get('text','')[:150]}...")
    else:
        lines.append("\n✗ No specific deposit amount found in contract")
        lines.append("Searching for any deposit references...")
        for c in contract_results[:3]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:150]}...")

    return "\n".join(lines)


def answer_rent(query, rent, rent_clause, contract_results, reg_results):
    lines = []
    lines.append("RENT ANALYSIS")
    lines.append("=" * 60)

    if rent:
        lines.append(f"\nMONTHLY RENT: Rs {rent:,}")
        if rent_clause:
            lines.append(f"Source: {format_clause_ref(rent_clause)}")
            lines.append(f"Text: {rent_clause.get('clause_text','')[:150]}...")

        # Check for increase clauses
        increase_clauses = [c for c in contract_results if any(w in c.get('text','').lower() for w in ['increase','enhance','hike'])]
        if increase_clauses:
            lines.append("\nRENT INCREASE PROVISIONS:")
            for c in increase_clauses[:2]:
                lines.append(f"  • {format_clause_ref(c)}: {c.get('text','')[:150]}...")

        # Regulations on rent
        if reg_results:
            lines.append("\nLEGAL REGULATIONS ON RENT:")
            for r in reg_results[:2]:
                lines.append(f"  [{r['metadata'].get('regulation_id')}] {r['metadata'].get('title')}")
                lines.append(f"  {r.get('text','')[:150]}...")
    else:
        lines.append("\n✗ No rent amount found")
        lines.append("Searching for rent-related clauses...")
        for c in contract_results[:3]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:150]}...")

    return "\n".join(lines)


def answer_maintenance(query, contract_results, reg_results):
    lines = []
    lines.append("MAINTENANCE RESPONSIBILITIES")
    lines.append("=" * 60)

    lines.append("\nCONTRACT TERMS:")
    maintenance_keywords = ['maintain', 'repair', 'damage', 'electrical', 'plumbing', 'clean']
    relevant = [c for c in contract_results if any(k in c.get('text','').lower() for k in maintenance_keywords)]

    if relevant:
        for c in relevant[:5]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:180]}...")
    else:
        lines.append("  No specific maintenance clauses found")

    if reg_results:
        lines.append("\nAPPLICABLE REGULATIONS:")
        for r in reg_results[:2]:
            lines.append(f"\n  [{r['metadata'].get('regulation_id')}] {r['metadata'].get('title')}")
            lines.append(f"  {r.get('text','')[:180]}...")

    return "\n".join(lines)


def answer_subletting(query, contract_results, reg_results):
    lines = []
    lines.append("SUBLETTING & ASSIGNMENT")
    lines.append("=" * 60)

    sublet_keywords = ['sublet', 'sub-let', 'sublease', 'assign', 'part with possession']
    relevant = [c for c in contract_results if any(k in c.get('text','').lower() or k in c.get('clause_title','').lower() for k in sublet_keywords)]

    if relevant:
        lines.append("\nCONTRACT CLAUSES:")
        for c in relevant[:3]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:200]}...")
    else:
        lines.append("\nNo explicit subletting clauses found in contract.")

    if reg_results:
        lines.append("\nLEGAL POSITION:")
        for r in reg_results[:1]:
            lines.append(f"\n  {r['metadata'].get('title')}:")
            lines.append(f"  {r.get('text','')[:250]}...")

    lines.append("\n\nSUMMARY:")
    lines.append("Subletting typically requires landlord's prior written consent.")
    lines.append("Unauthorized subletting may be grounds for eviction.")

    return "\n".join(lines)


def answer_termination(query, contract_results, reg_results):
    lines = []
    lines.append("TERMINATION & NOTICE PERIOD")
    lines.append("=" * 60)

    termination_keywords = ['terminat', 'notice', 'vacate', 'vacant', 'lock-in', 'renew']
    relevant = [c for c in contract_results if any(k in c.get('text','').lower() for k in termination_keywords)]

    lines.append("\nCONTRACTUAL TERMS:")
    if relevant:
        for c in relevant[:5]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:180]}...")
    else:
        lines.append("  No termination clauses found")

    if reg_results:
        lines.append("\nLEGAL REQUIREMENTS:")
        for r in reg_results[:2]:
            lines.append(f"\n  [{r['metadata'].get('regulation_id')}] {r['metadata'].get('title')}")
            lines.append(f"  {r.get('text','')[:180]}...")

    return "\n".join(lines)


def answer_rent_increase(query, contract_results, reg_results):
    lines = []
    lines.append("RENT INCREASE ANALYSIS")
    lines.append("=" * 60)

    increase_keywords = ['increase', 'enhance', 'hike', 'escalat']
    relevant = [c for c in contract_results if any(k in c.get('text','').lower() for k in increase_keywords)]

    lines.append("\nCONTRACT PROVISIONS:")
    if relevant:
        for c in relevant[:3]:
            lines.append(f"\n  {format_clause_ref(c)}:")
            lines.append(f"    {c.get('text','')[:200]}...")
    else:
        lines.append("  No rent increase clauses found")

    if reg_results:
        lines.append("\nLEGAL RESTRICTIONS:")
        for r in reg_results:
            lines.append(f"\n  [{r['metadata'].get('regulation_id')}] {r['metadata'].get('title')}")
            lines.append(f"  {r.get('text','')[:200]}...")

    lines.append("\n\nKEY TAKEAWAY:")
    lines.append("Rent can only be increased per contract terms or as allowed by law.")
    lines.append("Check contract for specific increase % and notice requirements.")

    return "\n".join(lines)


def answer_general(query, contract_results, reg_results):
    lines = []
    lines.append(f"QUERY: {query}")
    lines.append("=" * 60)

    lines.append(f"\nRELEVANT CONTRACT CLAUSES (top {min(5, len(contract_results))}):")
    for i, c in enumerate(contract_results[:5], 1):
        lines.append(f"\n  {i}. {format_clause_ref(c)}")
        lines.append(f"     {c.get('text','')[:180]}...")

    if reg_results:
        lines.append(f"\nRELEVANT REGULATIONS (top {min(3, len(reg_results))}):")
        for i, r in enumerate(reg_results[:3], 1):
            lines.append(f"\n  {i}. [{r['metadata'].get('regulation_id')}] {r['metadata'].get('title')}")
            lines.append(f"     {r.get('text','')[:180]}...")

    lines.append("\n\nNOTE: This is an offline answer based on document retrieval only.")
    lines.append("For AI-generated analysis, run: python query_contract.py")
    lines.append("(requires Google Generative AI API access)")

    return "\n".join(lines)


def main():
    if not Path('clause_vectors.json').exists():
        print("ERROR: clause_vectors.json not found.")
        print("Run 'python main.py' first to build the vector database.")
        sys.exit(1)

    print("=" * 70)
    print("OFFLINE RAG QUERY MODE (No API required)")
    print("=" * 70)
    print("\nThis mode uses vector search + templates (no LLM)")
    print("Type 'quit' to exit\n")

    # Load DB
    db = ClauseVectorDB(persist_path='clause_vectors.json')
    db.load()
    print(f"Loaded {len(db.documents)} documents")

    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print("\n" + "-" * 70)
            answer = offline_query(db, query)
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
