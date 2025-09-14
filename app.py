"""
Legal RAG Assistant - Streamlit Application

A comprehensive legal research and summarization tool using RAG technology.
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from src.ingestion import LegalDocumentProcessor
from src.retrieval import LegalRetrievalSystem
from src.summarization import LegalSummarizationAgent

# Page configuration
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .citation {
        background-color: #e9ecef;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = None
if 'retrieval_system' not in st.session_state:
    st.session_state.retrieval_system = None
if 'summarization_agent' not in st.session_state:
    st.session_state.summarization_agent = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def initialize_components():
    """Initialize the RAG components."""
    try:
        if st.session_state.doc_processor is None:
            with st.spinner("Initializing Legal RAG System..."):
                st.session_state.doc_processor = LegalDocumentProcessor()
                st.session_state.retrieval_system = LegalRetrievalSystem(st.session_state.doc_processor)
                st.session_state.summarization_agent = LegalSummarizationAgent()
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def display_sidebar():
    """Display the sidebar with system information and controls."""
    with st.sidebar:
        st.markdown("## ⚖️ Legal RAG Assistant")
        st.markdown("---")
        
        # System status
        if st.session_state.doc_processor:
            stats = st.session_state.doc_processor.get_collection_stats()
            st.markdown(f"**Documents in Database:** {stats.get('total_documents', 0)}")
        else:
            st.markdown("**Status:** Not initialized")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh System"):
            st.session_state.doc_processor = None
            st.session_state.retrieval_system = None
            st.session_state.summarization_agent = None
            st.rerun()
        
        if st.button("📊 View Statistics"):
            if st.session_state.doc_processor:
                stats = st.session_state.doc_processor.get_collection_stats()
                st.json(stats)
        
        st.markdown("---")
        
        # Available filters
        if st.session_state.retrieval_system:
            st.markdown("### Available Filters")
            filters = st.session_state.retrieval_system.get_available_filters()
            
            if filters['jurisdictions']:
                st.markdown("**Jurisdictions:**")
                for jurisdiction in filters['jurisdictions'][:5]:
                    st.markdown(f"• {jurisdiction}")
                if len(filters['jurisdictions']) > 5:
                    st.markdown(f"... and {len(filters['jurisdictions']) - 5} more")
        
        st.markdown("---")
        
        # Help section
        st.markdown("### Help")
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload Documents**: Go to the Upload tab and add legal PDFs
            2. **Ask Questions**: Use the Research tab to ask legal questions
            3. **Review Results**: Check citations and analysis quality
            4. **Export Results**: Download analysis as legal memos
            """)
        
        with st.expander("Citation Format"):
            st.markdown("""
            Citations appear as: `[document_id:chunk_index]`
            
            Example: `[case_123:5]` refers to chunk 5 of document case_123
            """)

def upload_tab():
    """Handle document upload and processing."""
    st.markdown('<div class="section-header">📥 Document Upload</div>', unsafe_allow_html=True)
    
    # Initialize components if needed
    if not initialize_components():
        return
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload Legal Documents (PDF format)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files containing legal documents, case law, statutes, etc."
    )
    
    if uploaded_files:
        st.markdown(f"**Selected Files:** {len(uploaded_files)}")
        
        # Display file details
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"{i+1}. {file.name} ({file.size:,} bytes)")
            with col2:
                st.write(f"Type: PDF")
            with col3:
                st.write(f"Status: Ready")
        
        # Metadata input section
        st.markdown("### Document Metadata")
        st.markdown("Provide additional context for better retrieval:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jurisdiction = st.selectbox(
                "Jurisdiction",
                ["All", "New York", "California", "Texas", "Florida", "Federal", "Other"],
                help="Select the legal jurisdiction for these documents"
            )
            
            court_type = st.selectbox(
                "Court Type",
                ["All", "Supreme Court", "Appellate Court", "District Court", "Circuit Court", "Other"],
                help="Type of court that issued these documents"
            )
        
        with col2:
            document_type = st.selectbox(
                "Document Type",
                ["All", "Case Law", "Statute", "Regulation", "Memo", "Brief", "Other"],
                help="Type of legal document"
            )
            
            year = st.number_input(
                "Year",
                min_value=1900,
                max_value=datetime.now().year,
                value=datetime.now().year,
                help="Year the document was issued"
            )
        
        # Additional metadata
        client_id = st.text_input(
            "Client/Matter ID (Optional)",
            help="Internal reference for client or matter"
        )
        
        precedential_status = st.selectbox(
            "Precedential Status",
            ["Unknown", "Binding Precedent", "Persuasive Authority", "Non-Precedential"],
            help="Legal precedential value of the document"
        )
        
        # Process files button
        if st.button("🚀 Process Documents", type="primary"):
            # Create metadata dictionary, excluding None values
            metadata = {
                'year': year,
                'client_id': client_id,
                'precedential_status': precedential_status
            }
            
            # Only add non-"All" values to avoid None values
            if jurisdiction != "All":
                metadata['jurisdiction'] = jurisdiction
            if court_type != "All":
                metadata['court_type'] = court_type
            if document_type != "All":
                metadata['document_type'] = document_type
            
            process_uploaded_files(uploaded_files, metadata)

def process_uploaded_files(uploaded_files, metadata):
    """Process the uploaded files and store them in the database."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    processed_files = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Process the document
            chunks = st.session_state.doc_processor.process_pdf(tmp_file_path, metadata)
            
            if chunks:
                # Store in database
                success = st.session_state.doc_processor.store_documents(chunks)
                
                if success:
                    processed_files += 1
                    st.session_state.uploaded_files.append({
                        'name': uploaded_file.name,
                        'chunks': len(chunks),
                        'metadata': metadata,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    st.error(f"Failed to store {uploaded_file.name}")
            else:
                st.error(f"Failed to process {uploaded_file.name}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Final status
    progress_bar.progress(1.0)
    if processed_files == total_files:
        st.success(f"✅ Successfully processed {processed_files} files!")
        st.markdown('<div class="success-box">All documents have been processed and stored in the database.</div>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Processed {processed_files} out of {total_files} files successfully.")
    
    status_text.empty()

def research_tab():
    """Handle legal research queries and analysis."""
    st.markdown('<div class="section-header">🔍 Legal Research</div>', unsafe_allow_html=True)
    
    # Initialize components if needed
    if not initialize_components():
        return
    
    # Check if we have documents
    if st.session_state.doc_processor:
        stats = st.session_state.doc_processor.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            st.warning("⚠️ No documents found. Please upload some legal documents first.")
            return
    
    # Query input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Legal Question",
            placeholder="e.g., What is the precedent for negligence in New York tort law?",
            height=100,
            help="Enter your legal research question here"
        )
    
    with col2:
        st.markdown("### Search Options")
        n_results = st.slider("Max Results", 3, 15, 5)
        max_context = st.slider("Max Context (chars)", 2000, 8000, 4000)
        
        # Logging info
        st.info("📝 OpenAI prompts and responses are logged to `legal_rag_openai_logs.log`")
    
    # Filter options
    st.markdown("### Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        jurisdiction_filter = st.selectbox(
            "Jurisdiction",
            ["All"] + st.session_state.retrieval_system.get_available_filters()['jurisdictions'],
            help="Filter by legal jurisdiction"
        )
    
    with col2:
        court_type_filter = st.selectbox(
            "Court Type",
            ["All"] + st.session_state.retrieval_system.get_available_filters()['court_types'],
            help="Filter by court type"
        )
    
    with col3:
        document_type_filter = st.selectbox(
            "Document Type",
            ["All"] + st.session_state.retrieval_system.get_available_filters()['document_types'],
            help="Filter by document type"
        )
    
    
    # Search button
    if st.button("🔍 Search Legal Documents", type="primary"):
        if query.strip():
            perform_legal_search(query, {
                'jurisdiction': jurisdiction_filter,
                'court_type': court_type_filter,
                'document_type': document_type_filter
            }, n_results, max_context)
        else:
            st.warning("Please enter a legal question.")

def perform_legal_search(query, filters, n_results, max_context):
    """Perform the legal search and analysis."""
    with st.spinner("Searching legal documents..."):
        # Get context from retrieval system
        context, source_documents = st.session_state.retrieval_system.get_context_for_question(
            query, 
            jurisdiction=filters['jurisdiction'] if filters['jurisdiction'] != "All" else None,
            court_type=filters['court_type'] if filters['court_type'] != "All" else None,
            document_type=filters['document_type'] if filters['document_type'] != "All" else None,
            max_context_length=max_context
        )
    
    if not context:
        st.warning("No relevant documents found. Try adjusting your search criteria.")
        return
    
    # Display search results
    st.markdown("### 📋 Search Results")
    st.markdown(f"Found {len(source_documents)} relevant document chunks")
    
    # Show source documents
    with st.expander("View Source Documents"):
        for i, doc in enumerate(source_documents, 1):
            st.markdown(f"**Document {i}:**")
            st.markdown(f"**Source:** {doc['metadata'].get('filename', 'Unknown')}")
            st.markdown(f"**Jurisdiction:** {doc['metadata'].get('jurisdiction', 'Unknown')}")
            st.markdown(f"**Relevance Score:** {doc.get('relevance_score', 0):.3f}")
            st.markdown(f"**Content:** {doc['content'][:200]}...")
            st.markdown("---")
    
    # Generate analysis
    with st.spinner("Generating legal analysis..."):
        analysis_result = st.session_state.summarization_agent.generate_legal_analysis(
            query, context, source_documents
        )
    
    # Display analysis
    st.markdown("### 📝 Legal Analysis")
    
    # Analysis text
    formatted_analysis = st.session_state.summarization_agent.format_analysis_for_display(analysis_result)
    st.markdown(formatted_analysis)
    
    # Quality indicators
    st.markdown("### 📊 Analysis Quality")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Citations", analysis_result['citation_stats']['total'])
    with col2:
        st.metric("Valid Citations", analysis_result['citation_stats']['valid'])
    with col3:
        st.metric("Citation Accuracy", f"{analysis_result['citation_stats']['ratio']:.1%}")
    
    # Quality issues
    if analysis_result['quality_issues']:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**Quality Issues:**")
        for issue in analysis_result['quality_issues']:
            st.markdown(f"• {issue}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Legal Memo"):
            memo = st.session_state.summarization_agent.create_legal_memo_format(analysis_result, query)
            st.download_button(
                label="Download Legal Memo",
                data=memo,
                file_name=f"legal_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("💾 Save Analysis"):
            st.session_state.analysis_history.append({
                'query': query,
                'analysis': analysis_result,
                'timestamp': datetime.now().isoformat()
            })
            st.success("Analysis saved to history!")
    
    with col3:
        if st.button("🔄 New Search"):
            st.rerun()
    
    # Follow-up suggestions
    suggestions = st.session_state.summarization_agent.suggest_follow_up_questions(query, analysis_result)
    if suggestions:
        st.markdown("### 💡 Suggested Follow-up Questions")
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
                st.session_state.suggested_query = suggestion
                st.rerun()

def history_tab():
    """Display analysis history."""
    st.markdown('<div class="section-header">📚 Analysis History</div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("No analysis history found. Perform some searches to build your history.")
        return
    
    # Display history
    for i, entry in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"Query {len(st.session_state.analysis_history) - i}: {entry['query'][:50]}..."):
            st.markdown(f"**Timestamp:** {entry['timestamp']}")
            st.markdown(f"**Query:** {entry['query']}")
            
            # Show analysis summary
            analysis = entry['analysis']
            st.markdown(f"**Citations:** {analysis['citation_stats']['valid']}/{analysis['citation_stats']['total']}")
            st.markdown(f"**Quality Issues:** {len(analysis['quality_issues'])}")
            
            # Show full analysis
            if st.button(f"View Full Analysis", key=f"view_{i}"):
                st.markdown(analysis['analysis'])

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">⚖️ Legal RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("AI-powered legal research and summarization tool with secure citations")
    
    # Sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📥 Upload Documents", "🔍 Legal Research", "📚 History"])
    
    with tab1:
        upload_tab()
    
    with tab2:
        research_tab()
    
    with tab3:
        history_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Legal RAG Assistant** | Built with LangChain, ChromaDB, and OpenAI | "
        "⚠️ Always verify AI-generated legal analysis with qualified legal professionals"
    )

if __name__ == "__main__":
    main()
