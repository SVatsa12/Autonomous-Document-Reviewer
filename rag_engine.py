"""Retrieval-Augmented Generation (RAG) engine for contract analysis with regulations."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from vector_db import ClauseVectorDB
from llm_ops import tracked_generate, MODEL_NAME


class RAGQueryEngine:
    """RAG engine that combines semantic retrieval with LLM generation."""

    def __init__(self, vector_db: ClauseVectorDB):
        self.vector_db = vector_db

    def hybrid_search(
        self,
        client,
        query: str,
        top_k_contract: int = 5,
        top_k_regulations: int = 3,
        rerank_mode: str = "none",
    ) -> Tuple[List[Dict], List[Dict]]:
        """Search both contract clauses and regulations using semantic similarity."""
        # Search contract clauses
        contract_results = self.vector_db.search(
            client,
            query,
            top_k=top_k_contract,
            metadata_filter={"source": "contract_clause"},
            rerank_mode=rerank_mode,
        )

        # Search regulations
        regulations_results = self.vector_db.search(
            client,
            query,
            top_k=top_k_regulations,
            metadata_filter={"source": "regulation"},
            rerank_mode=rerank_mode,
        )

        return contract_results, regulations_results

    def build_context(
        self,
        contract_results: List[Dict],
        regulations_results: List[Dict],
        max_context_chars: int = 8000,
    ) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        # Add regulations first (highest priority)
        if regulations_results:
            context_parts.append("=== RELEVANT LEGAL REGULATIONS ===")
            for i, r in enumerate(regulations_results, 1):
                reg = r.get("metadata", {})
                context_parts.append(
                    f"\nRegulation {i} [{reg.get('applicable_law', 'N/A')} - {reg.get('category', 'N/A')}]:\n"
                    f"Title: {reg.get('title', 'N/A')}\n"
                    f"Content: {r.get('text', '')}\n"
                    f"Regulation ID: {reg.get('regulation_id', 'N/A')}\n"
                )

        # Add contract clauses
        if contract_results:
            context_parts.append("\n=== CONTRACT CLAUSES ===")
            for i, c in enumerate(contract_results, 1):
                clause = c.get("clause", {})
                meta = c.get("metadata", {})
                context_parts.append(
                    f"\nClause {i} [{meta.get('clause_number', 'N/A')} - {meta.get('clause_title', 'N/A')}]:\n"
                    f"Text: {c.get('text', '')}\n"
                )

        full_context = "\n".join(context_parts)

        # Truncate if too long
        if len(full_context) > max_context_chars:
            full_context = full_context[:max_context_chars] + "\n... [Content truncated]"

        return full_context

    def generate_answer(
        self,
        client,
        query: str,
        context: str,
        answer_style: str = "comprehensive",
    ) -> str:
        """Generate answer using LLM based on retrieved context."""
        style_instructions = {
            "comprehensive": (
                "Give a useful answer in no more than 180 words. Include only the facts needed to answer the question."
            ),
            "concise": (
                "Give a direct answer in no more than 100 words. Mention the strongest supporting clause or regulation."
            ),
            "legal_analysis": (
                "Give a focused legal analysis in no more than 180 words. Distinguish clearly between what the contract says "
                "and what the supplied regulation says. Identify one practical implication only if it follows from the context."
            ),
        }

        instruction = style_instructions.get(answer_style, style_instructions["comprehensive"])

        prompt = f"""You are a careful contract-review assistant specialized in rental agreements.

USER QUESTION: {query}

RETRIEVED CONTEXT:
{context}

    TASK: {instruction}

    STRICT RESPONSE FORMAT:
    Answer: <one or two sentences directly answering the question>
    Contract: <one or two sentences citing the relevant clause number, or "Not stated in the contract.">
    Regulation: <one or two sentences citing the regulation ID/title, or "No relevant regulation was retrieved.">
    Caveat: <one short sentence only when jurisdiction, dates, or missing text makes the conclusion uncertain; otherwise omit this line>

    Rules:
    - Use plain text headings exactly as shown. Do not use Markdown tables, long lists, block quotes, or repeated summaries.
    - Keep the complete response within the requested word limit.
    - Use only the retrieved context. Do not invent clauses, dates, notice periods, statutory limits, or legal requirements.
    - Do not call a supplied regulation mandatory or universally applicable unless the context explicitly says so.
    - If the contract and regulation conflict or the context is incomplete, say so plainly.
    - Cite clause numbers and regulation IDs exactly as they appear in the context.

Your answer:"""

        try:
            response = tracked_generate(
                client,
                prompt,
                label=f"RAG answer generation ({answer_style})",
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def query(
        self,
        client,
        query: str,
        top_k_contract: int = 5,
        top_k_regulations: int = 3,
        rerank_mode: str = "none",
        answer_style: str = "comprehensive",
        include_sources: bool = True,
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve → generate → format response.

        Args:
            client: LLM client
            query: User question
            top_k_contract: Number of contract clauses to retrieve
            top_k_regulations: Number of regulations to retrieve
            rerank_mode: Reranking strategy (none, mmr, etc.)
            answer_style: Style of answer (comprehensive, concise, legal_analysis)
            include_sources: Include source citations in response

        Returns:
            Dict with answer, sources, and metadata
        """
        # Step 1: Hybrid retrieval
        contract_results, regulations_results = self.hybrid_search(
            client,
            query,
            top_k_contract=top_k_contract,
            top_k_regulations=top_k_regulations,
            rerank_mode=rerank_mode,
        )

        # Build context
        context = self.build_context(contract_results, regulations_results)

        # Step 2: Generate answer
        answer = self.generate_answer(client, query, context, answer_style)

        # Prepare sources
        sources = {
            "contract_clauses": [
                {
                    "clause_number": c.get("metadata", {}).get("clause_number"),
                    "clause_title": c.get("metadata", {}).get("clause_title"),
                    "score": c.get("score"),
                    "text_preview": c.get("text", "")[:200],
                }
                for c in contract_results
            ],
            "regulations": [
                {
                    "regulation_id": r.get("metadata", {}).get("regulation_id"),
                    "title": r.get("metadata", {}).get("title"),
                    "score": r.get("score"),
                    "text_preview": r.get("text", "")[:200],
                }
                for r in regulations_results
            ],
        }

        result = {
            "query": query,
            "answer": answer,
            "sources": sources if include_sources else None,
            "stats": {
                "contract_clauses_found": len(contract_results),
                "regulations_found": len(regulations_results),
                "total_context_chars": len(context),
            },
        }

        return result

    async def aquery(self, *args, **kwargs):
        """Async version of query."""
        import asyncio

        return await asyncio.to_thread(self.query, *args, **kwargs)
