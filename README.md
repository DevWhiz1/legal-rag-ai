# ⚖️ Legal RAG Assistant

A comprehensive **Retrieval-Augmented Generation (RAG) system** specialized for legal research and document summarization. This tool helps legal professionals quickly find relevant precedents, summarize complex legal documents, and generate properly cited legal analyses.

## 🎯 Features

- **📥 Secure Document Ingestion**: Upload and process legal PDFs with metadata preservation
- **🔍 Advanced Legal Retrieval**: Hybrid search combining semantic and metadata filtering
- **📝 AI-Powered Summarization**: Generate legal analyses with mandatory citations
- **🏛️ Jurisdiction-Aware**: Filter results by legal jurisdiction and court type
- **📊 Quality Assurance**: Built-in citation validation and quality checks
- **🖥️ User-Friendly Interface**: Clean Streamlit UI with upload and research tabs
- **💾 Persistent Storage**: ChromaDB for scalable document storage
- **🔒 Privacy-First**: Runs locally with no external data sharing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents

1. Go to the **Upload Documents** tab
2. Select PDF files containing legal documents
3. Fill in metadata:
   - **Jurisdiction**: Legal jurisdiction (e.g., New York, California)
   - **Court Type**: Type of court (Supreme Court, Appellate, etc.)
   - **Document Type**: Case law, statute, regulation, etc.
   - **Year**: Document year
   - **Client/Matter ID**: Internal reference (optional)
   - **Precedential Status**: Legal precedential value
4. Click **Process Documents**

### 2. Legal Research

1. Go to the **Legal Research** tab
2. Enter your legal question (e.g., "What is the precedent for negligence in New York tort law?")
3. Set search filters:
   - **Jurisdiction**: Filter by legal jurisdiction
   - **Court Type**: Filter by court type
   - **Document Type**: Filter by document type
   - **Date Range**: Filter by document date
4. Click **Search Legal Documents**
5. Review the AI-generated analysis with citations

### 3. Review Results

- **Analysis**: AI-generated legal analysis with proper citations
- **Citations**: Validated references to source documents
- **Quality Metrics**: Citation accuracy and analysis quality
- **Source Documents**: View the retrieved document chunks
- **Export Options**: Download as legal memo or save to history

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingestion     │    │   Retrieval     │    │ Summarization   │
│   Pipeline      │───▶│   System        │───▶│     Agent       │
│                 │    │                 │    │                 │
│ • PDF Loading   │    │ • Hybrid Search │    │ • AI Analysis   │
│ • Chunking      │    │ • Jurisdiction  │    │ • Citations     │
│ • Embeddings    │    │   Filtering     │    │ • Quality Check │
│ • Metadata      │    │ • Re-ranking    │    │ • Legal Memos   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ChromaDB Vector Store                       │
│              (Persistent Document Storage)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Python 3.8+
- **LLM**: OpenAI GPT-3.5-turbo (configurable)
- **Vector Database**: ChromaDB
- **Document Processing**: LangChain, PyPDF
- **UI**: Streamlit
- **Embeddings**: OpenAI text-embedding-ada-002

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Application Configuration
APP_TITLE="Legal RAG Assistant"
APP_DESCRIPTION="AI-powered legal research and summarization tool"
```

### Customization Options

- **Model Selection**: Change the LLM model in `src/summarization.py`
- **Chunk Size**: Adjust document chunking in `src/ingestion.py`
- **Search Parameters**: Modify retrieval settings in `src/retrieval.py`
- **UI Theme**: Customize Streamlit appearance in `app.py`

## 📊 Performance & Quality

### Citation System

- **Format**: `[document_id:chunk_index]`
- **Validation**: Automatic verification against source documents
- **Accuracy Tracking**: Real-time citation quality metrics
- **Traceability**: Full source document references

### Quality Assurance

- **Mandatory Citations**: Every claim must be backed by a citation
- **Insufficient Context Detection**: AI identifies when context is inadequate
- **Human Review Flags**: Automatic quality issue detection
- **Legal Memo Format**: Professional document generation

### Performance Metrics

- **Search Speed**: Sub-second retrieval for most queries
- **Accuracy**: 80%+ citation validity target
- **Scalability**: Handles thousands of documents
- **Memory Usage**: Optimized for 4GB+ systems

## 🛡️ Security & Privacy

### Data Protection

- **Local Processing**: All data stays on your machine
- **No External Sharing**: No data sent to third parties (except OpenAI API)
- **Secure Storage**: ChromaDB with local persistence
- **Access Control**: Built-in user session management

### Legal Compliance

- **Attorney-Client Privilege**: Maintains confidentiality
- **Data Retention**: Configurable document storage policies
- **Audit Trail**: Complete analysis history tracking
- **Citation Integrity**: Tamper-proof reference system

## 🚨 Important Disclaimers

### Legal Use

⚠️ **This tool is for research purposes only. Always verify AI-generated legal analysis with qualified legal professionals before using in practice.**

### Limitations

- **Accuracy**: AI may generate incorrect or incomplete information
- **Jurisdiction**: May not account for all jurisdictional nuances
- **Updates**: Legal precedents change; verify current law
- **Professional Judgment**: Cannot replace human legal expertise

### Best Practices

1. **Verify Citations**: Always check source documents
2. **Cross-Reference**: Use multiple sources for important matters
3. **Professional Review**: Have qualified attorneys review critical analyses
4. **Regular Updates**: Keep document database current
5. **Backup Data**: Regularly backup your ChromaDB database

## 🔧 Troubleshooting

### Common Issues

**"No documents found"**
- Ensure documents are properly uploaded and processed
- Check file format (PDF only)
- Verify metadata is correctly filled

**"Insufficient context to answer"**
- Upload more relevant documents
- Try broader search terms
- Adjust date range filters

**"OpenAI API Error"**
- Verify API key is correct
- Check API quota and billing
- Ensure internet connection

**"ChromaDB Error"**
- Check disk space
- Verify write permissions
- Try deleting and recreating the database

### Performance Issues

**Slow Search**
- Reduce max results parameter
- Use more specific filters
- Check system memory usage

**High Memory Usage**
- Reduce chunk size in ingestion settings
- Limit number of concurrent searches
- Restart application periodically

## 📈 Roadmap

### Planned Features

- [ ] **Multi-language Support**: Support for non-English legal documents
- [ ] **Advanced Analytics**: Document similarity and trend analysis
- [ ] **API Integration**: REST API for external system integration
- [ ] **Batch Processing**: Bulk document upload and processing
- [ ] **Custom Models**: Support for fine-tuned legal language models
- [ ] **Collaboration**: Multi-user access and sharing
- [ ] **Mobile App**: Mobile interface for legal research
- [ ] **Integration**: Connect with legal databases (LexisNexis, Westlaw)

### Version History

- **v1.0.0**: Initial release with core RAG functionality
- **v1.1.0**: Enhanced citation system and quality metrics
- **v1.2.0**: Improved UI and user experience
- **v2.0.0**: Advanced analytics and multi-user support (planned)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd legal-rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **ChromaDB**: For the vector database solution
- **OpenAI**: For the language models and embeddings
- **Streamlit**: For the user interface framework
- **Legal Community**: For feedback and requirements

## 📞 Support

For support, questions, or feature requests:

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainers
- **Documentation**: Check this README and inline code comments

---

**Built with ❤️ for the legal community**

*Always verify AI-generated legal analysis with qualified legal professionals.*
