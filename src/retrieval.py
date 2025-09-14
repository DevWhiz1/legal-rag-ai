"""
Specialized Legal Retrieval System

This module implements hybrid search combining semantic and metadata-based filtering
for precise legal document retrieval with jurisdiction awareness.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class LegalRetrievalSystem:
    """Advanced retrieval system for legal documents with jurisdiction filtering."""
    
    def __init__(self, document_processor):
        """Initialize with document processor."""
        self.doc_processor = document_processor
        self.embeddings = OpenAIEmbeddings()
    
    def extract_legal_terms(self, query: str) -> List[str]:
        """Extract legal terms and concepts from the query."""
        # Common legal terms and patterns
        legal_patterns = [
            r'\b(negligence|tort|contract|breach|liability|damages|precedent|statute|regulation)\b',
            r'\b(case law|common law|civil law|criminal law|constitutional law)\b',
            r'\b(plaintiff|defendant|appellant|respondent|petitioner)\b',
            r'\b(judgment|ruling|opinion|holding|dicta)\b',
            r'\b(appeal|motion|discovery|deposition|trial)\b'
        ]
        
        legal_terms = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            legal_terms.extend(matches)
        
        return list(set(legal_terms))
    
    def build_enhanced_query(self, original_query: str) -> str:
        """Enhance query with legal context and terms."""
        legal_terms = self.extract_legal_terms(original_query)
        
        # Add legal context to the query
        enhanced_query = original_query
        
        if legal_terms:
            legal_context = f" Legal terms: {', '.join(legal_terms)}."
            enhanced_query += legal_context
        
        # Add general legal research context
        enhanced_query += " This is a legal research query seeking relevant case law, statutes, or legal precedents."
        
        return enhanced_query
    
    def search_with_filters(self, query: str, 
                           jurisdiction: str = None,
                           court_type: str = None,
                           document_type: str = None,
                           n_results: int = 5) -> List[Dict[str, Any]]:
        """Perform filtered search with legal context awareness."""
        
        # Build enhanced query
        enhanced_query = self.build_enhanced_query(query)
        
        # Prepare filters
        filters = {}
        if jurisdiction and jurisdiction != "All":
            filters["jurisdiction"] = jurisdiction
        if court_type and court_type != "All":
            filters["court_type"] = court_type
        if document_type and document_type != "All":
            filters["document_type"] = document_type
        
        
        # Perform search
        results = self.doc_processor.search_documents(
            query=enhanced_query,
            n_results=n_results,
            filters=filters
        )
        
        return results
    
    def rerank_results(self, results: List[Dict[str, Any]], 
                      query: str) -> List[Dict[str, Any]]:
        """Re-rank results based on legal relevance and recency."""
        
        def calculate_relevance_score(result: Dict[str, Any]) -> float:
            """Calculate relevance score for a result."""
            content = result['content'].lower()
            query_lower = query.lower()
            
            # Base semantic similarity score (inverted distance)
            semantic_score = 1.0 - result['distance']
            
            # Keyword matching bonus
            keyword_matches = sum(1 for word in query_lower.split() 
                                if word in content)
            keyword_score = min(keyword_matches / len(query.split()), 1.0)
            
            # Legal term bonus
            legal_terms = self.extract_legal_terms(query)
            legal_term_matches = sum(1 for term in legal_terms 
                                   if term.lower() in content)
            legal_score = min(legal_term_matches / max(len(legal_terms), 1), 1.0)
            
            # Recency bonus (if year is available)
            recency_score = 0.5  # Default neutral score
            if 'year' in result['metadata']:
                try:
                    year = int(result['metadata']['year'])
                    current_year = datetime.now().year
                    # More recent documents get higher scores
                    recency_score = min(1.0, (year - 1900) / (current_year - 1900))
                except (ValueError, TypeError):
                    pass
            
            # Precedential value bonus
            precedential_score = 0.5  # Default neutral score
            if 'precedential_status' in result['metadata']:
                status = result['metadata']['precedential_status'].lower()
                if 'binding' in status or 'precedential' in status:
                    precedential_score = 1.0
                elif 'persuasive' in status:
                    precedential_score = 0.8
                elif 'non-precedential' in status:
                    precedential_score = 0.3
            
            # Weighted combination
            total_score = (
                0.4 * semantic_score +
                0.2 * keyword_score +
                0.2 * legal_score +
                0.1 * recency_score +
                0.1 * precedential_score
            )
            
            return total_score
        
        # Calculate scores and sort
        for result in results:
            result['relevance_score'] = calculate_relevance_score(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def get_context_for_question(self, question: str, 
                                jurisdiction: str = None,
                                court_type: str = None,
                                document_type: str = None,
                                max_context_length: int = 4000) -> Tuple[str, List[Dict[str, Any]]]:
        """Get relevant context for answering a legal question."""
        
        # Search for relevant documents
        search_results = self.search_with_filters(
            query=question,
            jurisdiction=jurisdiction,
            court_type=court_type,
            document_type=document_type,
            n_results=10  # Get more results for better context
        )
        
        # Re-rank results
        ranked_results = self.rerank_results(search_results, question)
        
        # Build context string
        context_parts = []
        used_results = []
        current_length = 0
        
        for result in ranked_results:
            content = result['content']
            metadata = result['metadata']
            
            # Add citation format
            citation = f"[{metadata.get('document_id', 'unknown')}:p{metadata.get('chunk_index', 0)}]"
            context_part = f"{citation} {content}"
            
            # Check if adding this would exceed max length
            if current_length + len(context_part) > max_context_length:
                break
            
            context_parts.append(context_part)
            used_results.append(result)
            current_length += len(context_part)
        
        context = "\n\n".join(context_parts)
        
        return context, used_results
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter options from the collection."""
        try:
            # Get all documents to extract unique values
            all_results = self.doc_processor.collection.get()
            
            filters = {
                'jurisdictions': set(),
                'court_types': set(),
                'document_types': set(),
                'years': set()
            }
            
            for metadata in all_results['metadatas']:
                if 'jurisdiction' in metadata:
                    filters['jurisdictions'].add(metadata['jurisdiction'])
                if 'court_type' in metadata:
                    filters['court_types'].add(metadata['court_type'])
                if 'document_type' in metadata:
                    filters['document_types'].add(metadata['document_type'])
                if 'year' in metadata:
                    try:
                        year = int(metadata['year'])
                        filters['years'].add(year)
                    except (ValueError, TypeError):
                        pass
            
            # Convert sets to sorted lists
            return {
                'jurisdictions': sorted(list(filters['jurisdictions'])),
                'court_types': sorted(list(filters['court_types'])),
                'document_types': sorted(list(filters['document_types'])),
                'years': sorted(list(filters['years']))
            }
            
        except Exception as e:
            st.error(f"Error getting filter options: {str(e)}")
            return {
                'jurisdictions': [],
                'court_types': [],
                'document_types': [],
                'years': []
            }
    
    def suggest_related_queries(self, query: str) -> List[str]:
        """Suggest related legal queries based on the input."""
        legal_terms = self.extract_legal_terms(query)
        
        suggestions = []
        
        # Generate suggestions based on legal terms
        if 'negligence' in query.lower():
            suggestions.extend([
                "What are the elements of negligence?",
                "Recent negligence cases in this jurisdiction",
                "Negligence defenses and exceptions"
            ])
        
        if 'contract' in query.lower():
            suggestions.extend([
                "Contract formation requirements",
                "Breach of contract remedies",
                "Contract interpretation principles"
            ])
        
        if 'tort' in query.lower():
            suggestions.extend([
                "Types of torts and their elements",
                "Tort defenses and limitations",
                "Damages in tort cases"
            ])
        
        # Add jurisdiction-specific suggestions
        if any(term in query.lower() for term in ['ny', 'new york', 'california', 'texas']):
            suggestions.append(f"State-specific laws and precedents")
        
        return suggestions[:5]  # Return top 5 suggestions
