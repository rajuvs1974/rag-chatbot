# Streamlit RAG Application with Google APIs

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit that leverages Google APIs to provide intelligent document search and question-answering capabilities.

## 🚀 Features

- **Document Upload & Processing**: Support for PDF, DOCX, and TXT files
- **Vector Search**: Efficient similarity search using embeddings
- **Google AI Integration**: Powered by Google's Gemini AI and Vertex AI
- **Real-time Chat Interface**: Interactive Q&A with your documents
- **Responsive UI**: Clean, modern interface built with Streamlit

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **AI/ML**: Google Vertex AI, Google Gemini Pro
- **Vector Database**: ChromaDB / Pinecone / FAISS
- **Document Processing**: PyPDF2, python-docx, langchain
- **Embeddings**: Google Universal Sentence Encoder

## 📋 Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account
- Google API credentials (Service Account or OAuth)
- Required API access:
  - Vertex AI API
  - Gemini API
  - Cloud Storage API (optional)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rajuvs1974/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Google Cloud Setup

#### Option A: Service Account (Recommended)
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing one
3. Enable required APIs:
   - Vertex AI API
   - Gemini API
4. Create a Service Account:
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Assign roles: `Vertex AI User`, `AI Platform Developer`
5. Download the JSON key file
6. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
   ```

### 5. Environment Configuration

Create a `.env` file in the project root:

```env
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Google APIs
GEMINI_API_KEY=your-gemini-api-key
VERTEX_AI_LOCATION=us-central1

# Application Settings
STREAMLIT_PORT=8501
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_STORE_TYPE=chromadb

# Optional: Vector Database Configuration
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env
```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Basic Workflow

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents**: Documents are automatically chunked and embedded
3. **Ask Questions**: Use the chat interface to query your documents
4. **View Sources**: See which document sections were used to generate answers

