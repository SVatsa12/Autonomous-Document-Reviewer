"""Unified compliance pipeline for any document type and domain."""

import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    from google import genai
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

from functions import clean_clauses, split_text
from llm_ops import extract_clauses_for_chunk, tracked_generate, MODEL_NAME, CLAUSE_EXTRACTION_RULES
from vector_db import ClauseVectorDB
from document_processor import DocumentProcessor, detect_domain, get_domain_config
from compliance_engine import ComplianceEngine, ThresholdRule, PresenceRule, ForbiddenClauseRule, SemanticComplianceRule
from enhanced_extraction import extract_rent_enhanced, extract_deposit_enhanced


class ConfigurablePipeline:
    """
    Domain-agnostic pipeline that adds new domains by extending rule sets.
    Usage: Change config files, not code.
    """
    
    def __init__(
        self,
        domain: Optional[str] = None,
        rules_file: Optional[str] = None,
        custom_rules: Optional[List[Dict]] = None,
        chunk_size: int = 300,
        persist_path: str = "clause_vectors.json",
    ):
        """
        Initialize pipeline.
        
        Args:
            domain: Domain name ('legal', 'banking', 'hr', 'generic')
            rules_file: Path to JSON rules configuration
            custom_rules: List of rule config dictionaries
            chunk_size: Text chunk size for processing
            persist_path: Where to save vector database
        """
        self.domain = domain or 'generic'
        self.rules_file = rules_file
        self.custom_rules_config = custom_rules or []
        self.chunk_size = chunk_size
        self.persist_path = persist_path
        
        # Document processor
        self.doc_processor = DocumentProcessor()
        
        # Load compliance engine
        self.compliance_engine = ComplianceEngine()
        if rules_file:
            self.compliance_engine.load_rules_from_file(rules_file)
        
        # Apply domain-specific rules
        self._apply_domain_rules()
    
    def _apply_domain_rules(self):
        """Apply domain-specific default rules."""
        domain_config = get_domain_config(self.domain)
        
        # If using generic rules but want legal-specific, adjust extraction
        if self.domain == 'legal':
            # Legal documents need thorough clause separation
            from llm_ops import CLAUSE_EXTRACTION_RULES as base_rules
            # You could modify rules here based on domain
            pass
    
    def process_document(
        self,
        file_path: str,
        client=None,
        run_compliance: bool = True,
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a document end-to-end.
        
        Args:
            file_path: Path to document (PDF, DOCX, image)
            client: LLM client (optional if using fallback)
            run_compliance: Whether to run compliance checks
            use_rag: Whether to enable RAG queries
            
        Returns:
            Dict with results including clauses, compliance, vector_db
        """
        results = {
            'domain': self.domain,
            'file_path': file_path,
            'success': False,
            'clauses': [],
            'compliance': None,
            'vector_db_path': None,
            'stats': {},
        }
        
        try:
            # Step 1: Detect domain if not specified
            if self.domain == 'generic':
                detected = detect_domain(file_path)
                print(f"[INFO] Detected domain: {detected}")
                self.domain = detected
                results['domain'] = detected
            
            # Step 2: Extract text from document
            print(f"\n[1/4] Extracting text from {Path(file_path).name}...")
            text = self.doc_processor.process(file_path)
            print(f"      Extracted {len(text):,} characters")
            
            # Step 3: Chunk text
            print(f"\n[2/4] Splitting into chunks...")
            chunks = split_text(text, max_chars=8000)
            print(f"      {len(chunks)} chunks created")
            
            # Step 4: Extract clauses
            print(f"\n[3/4] Extracting clauses...")
            all_clauses = []
            for i, chunk in enumerate(chunks):
                parsed = extract_clauses_for_chunk(client, chunk)
                if isinstance(parsed, list):
                    all_clauses.extend(parsed)
                time.sleep(2)  # Rate limiting
            
            all_clauses = clean_clauses(all_clauses)
            print(f"      Extracted {len(all_clauses)} clauses")
            results['clauses'] = all_clauses
            
            # Step 5: Build vector database
            print(f"\n[4/4] Building vector database...")
            vector_db = ClauseVectorDB(persist_path=self.persist_path)
            vector_db.add_clauses(client, all_clauses)
            
            # Add domain-specific seed documents if available
            self._add_domain_seeds(vector_db, client)
            
            vector_db.save()
            print(f"      Vector DB: {len(vector_db.documents)} documents")
            results['vector_db_path'] = self.persist_path
            results['vector_db'] = {
                'path': self.persist_path,
                'document_count': len(vector_db.documents),
            }
            
            # Step 6: Run compliance check
            if run_compliance:
                print(f"\n[5/5] Running compliance checks...")
                compliance_results = self.compliance_engine.evaluate(
                    all_clauses,
                    domain=self.domain,
                    client=client,
                )
                results['compliance'] = compliance_results
                
                print(f"      Compliance score: {compliance_results['compliance_score']}%")
                print(f"      Status: {compliance_results['overall_status']}")
                print(f"      Passed: {compliance_results['passed']}/{compliance_results['total_rules']}")
                if compliance_results['critical_issues']:
                    print(f"      [CRITICAL] {', '.join(compliance_results['critical_issues'])}")
            
            # Stats
            results['stats'] = {
                'total_text_chars': len(text),
                'num_chunks': len(chunks),
                'num_clauses': len(all_clauses),
                'num_rules_evaluated': len(self.compliance_engine.rules),
            }
            
            results['success'] = True
            return results
            
        except Exception as e:
            results['error'] = str(e)
            print(f"[ERROR] {e}")
            return results
    
    def _add_domain_seeds(self, vector_db: ClauseVectorDB, client):
        """Add domain-specific reference documents to vector DB."""
        seeds = {
            'legal': [
                {
                    'text': 'Security deposit should typically be 2-3 months rent maximum.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'deposit'}
                },
                {
                    'text': 'Notice period for termination should be mutually agreed but minimum 30 days.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'termination'}
                },
            ],
            'banking': [
                {
                    'text': 'Interest rates must comply with state usury laws and RBI guidelines.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'interest'}
                },
                {
                    'text': 'All loan terms must be disclosed clearly: principal, rate, term, payment schedule.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'disclosure'}
                },
            ],
            'hr': [
                {
                    'text': 'Salary and benefits must be clearly specified in employment agreements.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'compensation'}
                },
                {
                    'text': 'Termination notice period should be specified and comply with labor laws.',
                    'metadata': {'source': 'compliance_guideline', 'category': 'termination'}
                },
            ],
        }
        
        if self.domain in seeds:
            docs = [{'text': s['text'], 'metadata': s['metadata']} for s in seeds[self.domain]]
            vector_db.bulk_insert(client, docs)
    
    def add_custom_rule(self, rule_config: Dict):
        """Add a custom rule at runtime."""
        rule_type = rule_config.pop('type')
        rule_cls = {
            'threshold': ThresholdRule,
            'presence': PresenceRule,
            'forbidden': ForbiddenClauseRule,
            'semantic': SemanticComplianceRule,
        }.get(rule_type)
        
        if rule_cls:
            rule = rule_cls(**rule_config)
            self.compliance_engine.add_rule(rule)
            print(f"[INFO] Added custom rule: {rule.name}")
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")
    
    def export_rules(self, filepath: str):
        """Export current rules to JSON file."""
        rules_data = {'domain': self.domain, 'rules': []}
        for rule in self.compliance_engine.rules:
            rule_dict = rule.to_dict()
            # Remove non-serializable callables
            rules_data['rules'].append(rule_dict)
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
        print(f"[INFO] Exported {len(rules_data['rules'])} rules to {filepath}")


# ============================================================================
# DOMAIN EXTENSION EXAMPLE
# ============================================================================

def create_domain_rules_file(domain: str, filepath: str):
    """
    Create a template rules file for a new domain.
    This shows how to extend the system with custom rules.
    
    Example usage:
        create_domain_rules_file('insurance', 'insurance_rules.json')
        # Edit the file to add insurance-specific rules
        pipeline = ConfigurablePipeline(rules_file='insurance_rules.json', domain='insurance')
    """
    templates = {
        'insurance': {
            'domain': 'insurance',
            'rules': [
                {
                    'type': 'presence',
                    'rule_id': 'ins_001',
                    'name': 'Policy Number',
                    'description': 'Insurance policy must have unique policy number',
                    'category': 'identification',
                    'required_clauses': ['policy number', 'policy no'],
                    'severity': 'critical',
                    'domain': 'insurance',
                },
                {
                    'type': 'threshold',
                    'rule_id': 'ins_002',
                    'name': 'Coverage Amount Limit',
                    'description': 'Coverage amount cannot exceed underwriting limits',
                    'category': 'financial',
                    'threshold_type': 'max',
                    'threshold_value': 100000000,  # 10 crore
                    'field': 'coverage_amount',
                    'severity': 'critical',
                    'domain': 'insurance',
                },
                {
                    'type': 'presence',
                    'rule_id': 'ins_003',
                    'name': 'Beneficiary Clause',
                    'description': 'Must specify beneficiary details',
                    'category': 'mandatory',
                    'required_clauses': ['beneficiary', 'nominee'],
                    'severity': 'critical',
                    'domain': 'insurance',
                },
                {
                    'type': 'forbidden',
                    'rule_id': 'ins_004',
                    'name': 'No Misrepresentation',
                    'description': 'Cannot contain false statements about coverage',
                    'category': 'truthfulness',
                    'forbidden_terms': ['guaranteed investment', 'tax free returns', 'no risk'],
                    'severity': 'critical',
                    'domain': 'insurance',
                },
            ]
        },
        'real_estate': {
            'domain': 'real_estate',
            'rules': [
                {
                    'type': 'presence',
                    'rule_id': 're_001',
                    'name': 'Property Description',
                    'description': 'Must have complete property description',
                    'category': 'mandatory',
                    'required_clauses': ['property', 'address', 'survey'],
                    'severity': 'critical',
                    'domain': 'real_estate',
                },
                {
                    'type': 'threshold',
                    'rule_id': 're_002',
                    'name': 'Earnest Money Limit',
                    'description': 'Earnest money deposit should not exceed 5% of property value',
                    'category': 'financial',
                    'threshold_type': 'ratio',
                    'threshold_value': 0.05,
                    'field': 'earnest_money',
                    'comparison_field': 'property_value',
                    'severity': 'warning',
                    'domain': 'real_estate',
                },
            ]
        },
    }
    
    if domain in templates:
        with open(filepath, 'w') as f:
            json.dump(templates[domain], f, indent=2)
        print(f"[INFO] Created {domain} rules template at {filepath}")
    else:
        # Create empty template
        template = {
            'domain': domain,
            'description': f'Custom {domain} compliance rules',
            'rules': [
                {
                    'type': 'presence',
                    'rule_id': f'{domain[:3]}_001',
                    'name': 'Sample Mandatory Clause',
                    'description': 'Description of what this checks',
                    'category': 'mandatory',
                    'required_clauses': ['mandatory clause keyword'],
                    'severity': 'critical',
                    'domain': domain,
                },
                {
                    'type': 'threshold',
                    'rule_id': f'{domain[:3]}_002',
                    'name': 'Sample Threshold',
                    'description': 'Description of numeric check',
                    'category': 'financial',
                    'threshold_type': 'max',  # or 'min', 'range', 'ratio'
                    'threshold_value': 100000,
                    'field': 'amount_field_name',
                    'severity': 'warning',
                    'domain': domain,
                },
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"[INFO] Created empty {domain} template at {filepath}")
        print(f"[INFO] Edit the file to add your custom rules")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Configurable Pipeline Examples\n")
    print("="*60)
    
    print("\n1. Create a new domain rules file:")
    print("   create_domain_rules_file('insurance', 'insurance_rules.json')")
    
    print("\n2. Use pipeline with existing domain:")
    print("   pipeline = ConfigurablePipeline(domain='legal')")
    print("   results = pipeline.process_document('contract.pdf', client=llm_client)")
    
    print("\n3. Add custom rule at runtime:")
    print("   pipeline.add_custom_rule({")
    print("       'type': 'threshold',")
    print("       'rule_id': 'custom_001',")
    print("       'name': 'Custom Check',")
    print("       ...")  # Add rest of rule config
    print("   })")
    
    print("\nAvailable domains: legal, banking, hr, generic")
    print("Extend by: editing JSON rules file or using add_custom_rule()")
