"""
Legal Document Ingestion Pipeline

This module handles the secure ingestion, chunking, and embedding of legal documents
with proper metadata preservation for jurisdiction-specific retrieval.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pypdf import PdfReader
import streamlit as st


class LegalDocumentProcessor:
    """Handles legal document processing with jurisdiction-aware chunking."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the document processor with ChromaDB."""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Legal-aware text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract basic metadata from filename patterns."""
        metadata = {
            "filename": filename,
            "upload_date": datetime.now().isoformat(),
            "document_id": str(uuid.uuid4())
        }
        
        # Try to extract jurisdiction from filename
        filename_lower = filename.lower()
        if "ny" in filename_lower or "new_york" in filename_lower:
            metadata["jurisdiction"] = "New York"
        elif "ca" in filename_lower or "california" in filename_lower:
            metadata["jurisdiction"] = "California"
        elif "tx" in filename_lower or "texas" in filename_lower:
            metadata["jurisdiction"] = "Texas"
        elif "fl" in filename_lower or "florida" in filename_lower:
            metadata["jurisdiction"] = "Florida"
        elif "pakistan" in filename_lower or "pk" in filename_lower:
            metadata["jurisdiction"] = "Pakistan"
        elif "india" in filename_lower or "indian" in filename_lower:
            metadata["jurisdiction"] = "India"
        elif "uk" in filename_lower or "united_kingdom" in filename_lower or "britain" in filename_lower:
            metadata["jurisdiction"] = "United Kingdom"
        elif "canada" in filename_lower or "ca" in filename_lower:
            metadata["jurisdiction"] = "Canada"
        else:
            metadata["jurisdiction"] = "Unknown"
        
        return metadata
    
    def process_pdf(self, file_path: str, custom_metadata: Dict[str, Any] = None) -> List[Document]:
        """Process a PDF file and return chunked documents with metadata."""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract metadata
            base_metadata = self.extract_metadata_from_filename(os.path.basename(file_path))
            if custom_metadata:
                base_metadata.update(custom_metadata)
            
            # Add metadata to all documents
            for doc in documents:
                doc.metadata.update(base_metadata)
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{base_metadata['document_id']}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            return chunks
            
        except Exception as e:
            st.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata by removing None values and ensuring proper types."""
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                # Convert to string if it's not a basic type that ChromaDB accepts
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)
        return cleaned

    def store_documents(self, documents: List[Document]) -> bool:
        """Store processed documents in ChromaDB."""
        try:
            if not documents:
                return False
            
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [self.clean_metadata(doc.metadata) for doc in documents]
            ids = [doc.metadata["chunk_id"] for doc in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error storing documents: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_documents(self, query: str, n_results: int = 5, 
                        filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents with optional filters."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if value and value != "All":
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_available_jurisdictions(self) -> List[str]:
        """Get list of available jurisdictions in the collection."""
        try:
            # Get all unique jurisdictions
            results = self.collection.get()
            jurisdictions = set()
            
            for metadata in results['metadatas']:
                if 'jurisdiction' in metadata:
                    jurisdictions.add(metadata['jurisdiction'])
            
            return sorted(list(jurisdictions))
            
        except Exception as e:
            st.error(f"Error getting jurisdictions: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Find all chunks for this document
            results = self.collection.get(where={"document_id": document_id})
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
