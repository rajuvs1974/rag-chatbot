import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
import pickle
import os
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

class DocumentProcessor:
    """Handle document processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file):
        """Extract text from PDF file"""
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
        return text
    
    @staticmethod
    def extract_text_from_docx(file):
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = docx.Document(file)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
        return text
    
    @staticmethod
    def extract_text_from_txt(file):
        """Extract text from TXT file"""
        try:
            text = file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            text = ""
        return text

class VectorStore:
    """Handle document embeddings and similarity search"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, texts, sources):
        """Add documents to the vector store"""
        # Split texts into chunks
        chunks = []
        chunk_metadata = []
        
        for text, source in zip(texts, sources):
            text_chunks = self.split_text(text)
            chunks.extend(text_chunks)
            chunk_metadata.extend([{
                'source': source,
                'timestamp': datetime.now().isoformat()
            }] * len(text_chunks))
        
        # Generate embeddings
        new_embeddings = self.model.encode(chunks)
        
        # Add to store
        self.documents.extend(chunks)
        self.embeddings.extend(new_embeddings)
        self.metadata.extend(chunk_metadata)
    
    def split_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search(self, query, top_k=3):
        """Search for similar documents"""
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': similarities[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save_store(self, filename):
        """Save vector store to file"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_store(self, filename):
        """Load vector store from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.metadata = data['metadata']
            return True
        return False

class RAGChatbot:
    """Main RAG chatbot class"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.vector_store = VectorStore()
        self.chat_history = []
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        texts = []
        sources = []
        
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = DocumentProcessor.extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = DocumentProcessor.extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = DocumentProcessor.extract_text_from_txt(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue
            
            if text.strip():
                texts.append(text)
                sources.append(file.name)
        
        if texts:
            self.vector_store.add_documents(texts, sources)
            st.success(f"Successfully processed {len(texts)} documents!")
    
    def generate_response(self, query):
        """Generate response using RAG"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query, top_k=3)
        
        if not relevant_docs:
            return "I don't have any relevant documents to answer your question. Please upload some documents first."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Create prompt with context
        prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Add sources information
            sources = list(set([doc['metadata']['source'] for doc in relevant_docs]))
            source_info = f"\n\n**Sources:** {', '.join(sources)}"
            
            return response.text + source_info
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.title("🤖 RAG Chatbot with Google Gemini")
    st.markdown("Upload documents and ask questions about them!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("Google API Key", type="password", 
                               help="Get your API key from Google AI Studio")
        
        if not api_key:
            st.warning("Please enter your Google API key to continue")
            st.stop()
        
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = RAGChatbot(api_key)
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    st.session_state.chatbot.process_documents(uploaded_files)
        
        # Document stats
        if st.session_state.chatbot.vector_store.documents:
            st.success(f"📄 {len(st.session_state.chatbot.vector_store.documents)} document chunks loaded")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.chat_message("user").write(message['content'])
                else:
                    st.chat_message("assistant").write(message['content'])
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.chatbot.vector_store.documents:
                st.error("Please upload and process documents first!")
            else:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': query
                })
                
                # Generate response
                with st.spinner("Generating response..."):
                    response = st.session_state.chatbot.generate_response(query)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
    
    with col2:
        st.header("Document Search")
        
        # Search interface
        search_query = st.text_input("Search documents:")
        
        if search_query and st.session_state.chatbot.vector_store.documents:
            results = st.session_state.chatbot.vector_store.search(search_query, top_k=3)
            
            st.subheader("Search Results:")
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                    st.write(f"**Source:** {result['metadata']['source']}")
                    st.write(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])

if __name__ == "__main__":
    main()