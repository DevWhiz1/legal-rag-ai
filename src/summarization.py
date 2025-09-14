"""
Secure Legal Summarization Agent

This module implements a secure summarization system with mandatory citations
and human-in-the-loop validation for legal document analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from datetime import datetime

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('legal_rag_openai_logs.log')
    ]
)
logger = logging.getLogger(__name__)


class LegalSummarizationAgent:
    """Secure legal summarization agent with citation requirements."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize the summarization agent."""
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=2000
        )
        
        # Define the system prompt for legal analysis
        self.system_prompt = """You are a specialized legal research assistant. Your role is to:

1. ACCURATELY analyze and summarize legal documents
2. PROVIDE SPECIFIC CITATIONS for every claim you make
3. MAINTAIN PROFESSIONAL LEGAL STANDARDS
4. IDENTIFY JURISDICTIONAL CONTEXT when relevant
5. HIGHLIGHT KEY LEGAL PRINCIPLES and precedents

CRITICAL REQUIREMENTS:
- Every factual claim MUST include a citation in format [doc_id:chunk_index]
- If you cannot find sufficient evidence, state "Insufficient context to answer"
- Focus on legal precedents, statutes, and case law
- Maintain objectivity and avoid speculation
- Use proper legal terminology and formatting
- ALWAYS use the CORRECT jurisdiction name from the source documents (e.g., "Pakistan Penal Code" not "Indian Penal Code")
- Pay attention to the jurisdiction metadata in the source documents

CITATION FORMAT: [document_id:chunk_index] for each claim
RESPONSE STRUCTURE: Answer, then list all citations used"""

        # Create the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("""
Legal Question: {question}

Context Documents:
{context}

Please provide a comprehensive legal analysis with proper citations. If the context is insufficient, clearly state this limitation.
""")
        ])
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract and validate citations from the response."""
        citation_pattern = r'\[([^:]+):(\d+)\]'
        citations = []
        
        matches = re.finditer(citation_pattern, text)
        for match in matches:
            doc_id = match.group(1)
            chunk_index = int(match.group(2))
            citations.append({
                'document_id': doc_id,
                'chunk_index': chunk_index,
                'full_citation': match.group(0),
                'position': match.start()
            })
        
        return citations
    
    def validate_citations(self, citations: List[Dict[str, Any]], 
                          source_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that citations reference actual source documents."""
        valid_citations = []
        source_doc_ids = {doc['metadata'].get('document_id') for doc in source_documents}
        
        for citation in citations:
            doc_id = citation['document_id']
            if doc_id in source_doc_ids:
                # Find the specific chunk
                for doc in source_documents:
                    if (doc['metadata'].get('document_id') == doc_id and 
                        doc['metadata'].get('chunk_index') == citation['chunk_index']):
                        citation['validated'] = True
                        citation['source_content'] = doc['content']
                        citation['source_metadata'] = doc['metadata']
                        break
                else:
                    citation['validated'] = False
            else:
                citation['validated'] = False
            
            valid_citations.append(citation)
        
        return valid_citations
    
    def generate_legal_analysis(self, question: str, context: str, 
                               source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive legal analysis with citations."""
        
        logger.info(f"Starting legal analysis generation for question: '{question[:100]}...'")
        
        try:
            # Format the prompt
            messages = self.prompt_template.format_messages(
                question=question,
                context=context
            )
            
            # Log the prompt and context being sent to OpenAI
            logger.info("=" * 80)
            logger.info("OPENAI PROMPT AND CONTEXT LOGGING")
            logger.info("=" * 80)
            logger.info(f"Question: {question}")
            logger.info(f"Context Length: {len(context)} characters")
            logger.info(f"Number of Source Documents: {len(source_documents)}")
            logger.info("-" * 40)
            logger.info("SYSTEM PROMPT:")
            logger.info(self.system_prompt)
            logger.info("-" * 40)
            logger.info("CONTEXT BEING SENT:")
            logger.info(context)
            logger.info("-" * 40)
            logger.info("SOURCE DOCUMENTS METADATA:")
            for i, doc in enumerate(source_documents):
                metadata = doc.get('metadata', {})
                filename = metadata.get('filename', 'Unknown')
                jurisdiction = metadata.get('jurisdiction', 'Unknown')
                document_type = metadata.get('document_type', 'Unknown')
                logger.info(f"Document {i+1}: {filename} - Jurisdiction: {jurisdiction} - Type: {document_type}")
            logger.info("=" * 80)
            
            # Generate response
            response = self.llm(messages)
            analysis_text = response.content
            
            # Log the response
            logger.info("OPENAI RESPONSE:")
            logger.info(analysis_text)
            logger.info("=" * 80)
            logger.info("SUMMARY: Legal analysis completed successfully")
            logger.info(f"Response length: {len(analysis_text)} characters")
            logger.info("=" * 80)
            
            # Extract citations
            citations = self.extract_citations(analysis_text)
            
            # Validate citations
            validated_citations = self.validate_citations(citations, source_documents)
            
            # Calculate citation statistics
            total_citations = len(citations)
            valid_citations = len([c for c in validated_citations if c['validated']])
            citation_ratio = valid_citations / total_citations if total_citations > 0 else 0
            
            # Check if response meets quality standards
            quality_issues = []
            if citation_ratio < 0.8:  # Less than 80% valid citations
                quality_issues.append("Low citation validity")
            
            if "insufficient context" in analysis_text.lower():
                quality_issues.append("Insufficient context identified")
            
            if len(analysis_text) < 100:
                quality_issues.append("Response too brief")
            
            return {
                'analysis': analysis_text,
                'citations': validated_citations,
                'citation_stats': {
                    'total': total_citations,
                    'valid': valid_citations,
                    'ratio': citation_ratio
                },
                'quality_issues': quality_issues,
                'requires_review': len(quality_issues) > 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating legal analysis: {str(e)}")
            logger.error(f"Question was: {question}")
            logger.error(f"Context length was: {len(context)} characters")
            return {
                'analysis': f"Error generating analysis: {str(e)}",
                'citations': [],
                'citation_stats': {'total': 0, 'valid': 0, 'ratio': 0},
                'quality_issues': ['Generation error'],
                'requires_review': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def format_analysis_for_display(self, analysis_result: Dict[str, Any]) -> str:
        """Format the analysis result for display in the UI."""
        analysis = analysis_result['analysis']
        citations = analysis_result['citations']
        
        # Add citation details at the end
        if citations:
            analysis += "\n\n--- CITATIONS ---\n"
            for i, citation in enumerate(citations, 1):
                status = "✓" if citation['validated'] else "✗"
                analysis += f"{i}. {status} {citation['full_citation']}"
                if citation['validated'] and 'source_metadata' in citation:
                    metadata = citation['source_metadata']
                    filename = metadata.get('filename', 'Unknown')
                    jurisdiction = metadata.get('jurisdiction', 'Unknown')
                    analysis += f" ({filename}, {jurisdiction})"
                analysis += "\n"
        
        return analysis
    
    def generate_summary_report(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report of the analysis."""
        stats = analysis_result['citation_stats']
        issues = analysis_result['quality_issues']
        
        report = {
            'summary': {
                'total_citations': stats['total'],
                'valid_citations': stats['valid'],
                'citation_accuracy': f"{stats['ratio']:.1%}",
                'quality_issues': len(issues),
                'requires_human_review': analysis_result['requires_review']
            },
            'recommendations': []
        }
        
        # Add recommendations based on quality issues
        if stats['ratio'] < 0.8:
            report['recommendations'].append(
                "Consider adding more specific legal context or refining the search query"
            )
        
        if 'Insufficient context' in issues:
            report['recommendations'].append(
                "Upload more relevant legal documents or expand the search criteria"
            )
        
        if 'Response too brief' in issues:
            report['recommendations'].append(
                "The analysis may be incomplete - consider providing more context"
            )
        
        if not issues:
            report['recommendations'].append(
                "Analysis meets quality standards and is ready for use"
            )
        
        return report
    
    def suggest_follow_up_questions(self, question: str, 
                                   analysis_result: Dict[str, Any]) -> List[str]:
        """Suggest follow-up questions based on the analysis."""
        suggestions = []
        
        # Extract key legal concepts from the question
        legal_concepts = re.findall(r'\b(negligence|contract|tort|liability|damages|precedent|statute)\b', 
                                  question.lower())
        
        # Generate suggestions based on concepts
        if 'negligence' in legal_concepts:
            suggestions.extend([
                "What are the specific elements of negligence in this case?",
                "Are there any defenses to negligence claims?",
                "What damages are typically awarded in negligence cases?"
            ])
        
        if 'contract' in legal_concepts:
            suggestions.extend([
                "What constitutes a valid contract formation?",
                "What remedies are available for breach of contract?",
                "Are there any contract defenses or exceptions?"
            ])
        
        # Add general legal research suggestions
        suggestions.extend([
            "What are the recent developments in this area of law?",
            "How do other jurisdictions handle similar cases?",
            "What are the practical implications of this legal principle?"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def create_legal_memo_format(self, analysis_result: Dict[str, Any], 
                                question: str) -> str:
        """Format the analysis as a legal memo."""
        analysis = analysis_result['analysis']
        citations = analysis_result['citations']
        
        memo = f"""
LEGAL RESEARCH MEMORANDUM

TO: Legal Research Team
FROM: AI Legal Assistant
DATE: {datetime.now().strftime('%Y-%m-%d')}
RE: {question}

EXECUTIVE SUMMARY:
{analysis.split('.')[0] if '.' in analysis else analysis}

DETAILED ANALYSIS:
{analysis}

CITATIONS:
"""
        
        for i, citation in enumerate(citations, 1):
            if citation['validated']:
                memo += f"{i}. {citation['full_citation']}"
                if 'source_metadata' in citation:
                    metadata = citation['source_metadata']
                    filename = metadata.get('filename', 'Unknown')
                    jurisdiction = metadata.get('jurisdiction', 'Unknown')
                    memo += f" - {filename} ({jurisdiction})"
                memo += "\n"
        
        memo += f"""
RECOMMENDATION:
Based on the available legal authority, {'proceed with caution' if analysis_result['requires_review'] else 'proceed with confidence'}.

QUALITY ASSURANCE:
- Total Citations: {analysis_result['citation_stats']['total']}
- Valid Citations: {analysis_result['citation_stats']['valid']}
- Citation Accuracy: {analysis_result['citation_stats']['ratio']:.1%}
- Human Review Required: {'Yes' if analysis_result['requires_review'] else 'No'}
"""
        
        return memo
