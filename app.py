"""
Main entry point: Unified compliance pipeline for any document, any domain.
Set GROQ_API_KEY in .env or the environment before running.

Usage:
    python app.py --file contract.pdf [--domain legal] [--rules rules_legal.json]
    python app.py --file loan_app.docx --domain banking
    python app.py --file scanned.jpg  # Auto-detects domain
"""

import argparse
import json
import sys
import os
from pathlib import Path

try:
    from groq_client import GroqClient
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

from configurable_pipeline import ConfigurablePipeline, detect_domain, create_domain_rules_file
from document_processor import DocumentProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Universal Compliance Pipeline - Process any document in any domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect domain (uses filename heuristics)
  python app.py --file rental_agreement.pdf
  
  # Specify domain explicitly
  python app.py --file loan_application.docx --domain banking --rules rules_banking.json
  
  # Process scanned image
  python app.py --file signed_contract.jpg
  
  # Generate rules template for new domain
  python app.py --create-rules insurance
  
  # Show document info without processing
  python app.py --info policy.docx
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Path to document (PDF, DOCX, PNG, JPG, TIFF)"
    )
    parser.add_argument(
        "--domain", "-d",
        choices=["legal", "banking", "hr", "generic", "insurance", "real_estate"],
        default="generic",
        help="Domain for compliance rules (default: auto-detect)"
    )
    parser.add_argument(
        "--rules", "-r",
        help="Path to rules JSON file (default: rules_<domain>.json)"
    )
    parser.add_argument(
        "--no-compliance",
        action="store_true",
        help="Skip compliance checks, only extract clauses"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG query functionality"
    )
    parser.add_argument(
        "--output", "-o",
        default="results.json",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--create-rules",
        metavar="DOMAIN",
        help="Create rules template for a new domain and exit"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show document info (format, detected domain) without processing"
    )
    
    args = parser.parse_args()
    
    # Handle --create-rules
    if args.create_rules:
        domain = args.create_rules
        filename = f"rules_{domain}.json"
        create_domain_rules_file(domain, filename)
        print(f"✓ Created {filename}")
        print(f"  Edit this file to customize rules for {domain} domain")
        return
    
    # Require --file unless using --create-rules
    if not args.file:
        parser.print_help()
        print("\nERROR: --file is required (or use --create-rules to make template)")
        sys.exit(1)
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Handle --info
    if args.info:
        show_document_info(file_path)
        return
    
    # Determine rules file
    if not args.rules:
        args.rules = f"rules_{args.domain}.json"
    
    # Check if file is supported
    processor = DocumentProcessor()
    if not processor.is_supported(file_path):
        print(f"ERROR: Unsupported file format: {file_path.suffix}")
        print(f"Supported: {', '.join(processor.get_supported_formats())}")
        sys.exit(1)
    
    # Get API key
    api_key = os.environ.get("GROQ_API_KEY")
    client = None
    if HAS_LLM and api_key:
        client = GroqClient(api_key=api_key)
    elif not args.no_compliance and not args.no_rag:
        print("[WARNING] GROQ_API_KEY not set - some features will be limited")
        print("  - LLM-based clause extraction: will use fallback")
        print("  - Semantic compliance checks: SKIP")
        print("  - RAG Q&A: OFFLINE mode only")
        print()
    
    # Initialize pipeline
    print("=" * 70)
    print("UNIVERSAL COMPLIANCE PIPELINE")
    print("=" * 70)
    print(f"Document: {file_path.name}")
    print(f"Domain: {args.domain}")
    print(f"Rules file: {args.rules}")
    print(f"Compliance checks: {'DISABLED' if args.no_compliance else 'ENABLED'}")
    print(f"RAG queries: {'DISABLED' if args.no_rag else 'ENABLED'}")
    print("=" * 70)
    
    pipeline = ConfigurablePipeline(
        domain=args.domain,
        rules_file=args.rules if Path(args.rules).exists() else None,
        persist_path=f"clause_vectors_{args.domain}.json"
    )
    
    # Process document
    print("\n[STARTING] Document processing...")
    results = pipeline.process_document(
        file_path=str(file_path),
        client=client,
        run_compliance=not args.no_compliance,
        use_rag=not args.no_rag,
    )
    
    # Add file info
    results['file_info'] = {
        'name': file_path.name,
        'size_bytes': file_path.stat().st_size,
        'format': file_path.suffix.lower(),
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[COMPLETE] Results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success: {results['success']}")
    
    if results.get('error'):
        print(f"ERROR: {results['error']}")
        sys.exit(1)
    
    stats = results['stats']
    print(f"Text extracted: {stats.get('total_text_chars', 0):,} chars")
    print(f"Chunks processed: {stats.get('num_chunks', 0)}")
    print(f"Clauses extracted: {stats.get('num_clauses', 0)}")
    print(f"Rules evaluated: {stats.get('num_rules_evaluated', 0)}")
    
    if results.get('compliance'):
        comp = results['compliance']
        print(f"\nCOMPLIANCE SCORE: {comp['compliance_score']}%")
        print(f"Status: {comp['overall_status']}")
        print(f"  Passed: {comp['passed']}/{comp['total_rules']}")
        print(f"  Failed: {comp['failed']}")
        print(f"  Not evaluated: {comp.get('not_evaluated', 0)}")
        
        if comp['critical_issues']:
            print("\nCRITICAL ISSUES:")
            for issue in comp['critical_issues']:
                print(f"  ✗ {issue}")
        
        # Show failed rules
        print("\nFAILED RULES:")
        for rule in comp['rule_results']:
            if not rule.get('passed'):
                rule_name = rule.get('name') or rule.get('rule_id') or 'Unknown rule'
                print(f"  [{str(rule.get('severity', 'info')).upper()}] {rule_name}")
                print(f"    {rule.get('message', 'No message provided')}")
                if rule.get('recommendation'):
                    print(f"    → {rule['recommendation']}")
    
    print("=" * 70)


def show_document_info(file_path: Path):
    """Show document metadata without processing."""
    processor = DocumentProcessor()
    
    print("=" * 60)
    print("DOCUMENT INFORMATION")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Size: {file_path.stat().st_size:,} bytes")
    print(f"Format: {file_path.suffix.lower()}")
    print(f"Supported: {processor.is_supported(file_path)}")
    
    if processor.is_supported(file_path):
        detected_domain = detect_domain(file_path)
        print(f"Detected domain: {detected_domain}")
        print(f"Suggested rules file: rules_{detected_domain}.json")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
