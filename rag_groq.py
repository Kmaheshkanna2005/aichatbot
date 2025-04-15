import os
import re
import random
import requests
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import uuid
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load API key from .env ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise EnvironmentError("GROQ_API_KEY not found. Please add it to your .env file")

# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "rag_uploads")
VECTOR_STORE_FOLDER = os.path.join(tempfile.gettempdir(), "rag_vectorstores")
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# Dictionary to track active sessions
active_sessions = {}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_pdf(pdf_path):
    """Load and split a PDF into chunks"""
    logger.info(f"Loading PDF from: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"PDF split into {len(docs)} chunks")
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

def initialize_vector_store(documents):
    """Initialize FAISS vector store with document embeddings"""
    logger.info("Initializing vector store...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        embedding_dim = len(embedding_model.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        vector_store.add_documents(documents)
        logger.info("Vector store initialized successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def generate_questions(documents, num_questions=5):
    """Generate questions from the PDF content"""
    logger.info(f"Generating {num_questions} questions from the document...")
    
    try:
        # Combine some random chunks to send to the API
        selected_docs = random.sample(documents, min(10, len(documents)))
        combined_text = "\n\n".join([doc.page_content for doc in selected_docs])
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Given the following content from a document, generate {num_questions} diverse and insightful questions 
        that would test understanding of the material. 
        Create questions that require reasoning and comprehension, not just factual recall.
        Format your response as a Python list of strings, with each question as an element.
        Do not include any explanations, just the list of questions.
        
        Document content:
        {combined_text}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at generating insightful questions from educational content."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            questions_text = response.json()['choices'][0]['message']['content'].strip()
            
            # Extract questions as a list from the response
            try:
                # Try to extract a Python-formatted list from the response
                questions_match = re.search(r'\[(.*?)\]', questions_text, re.DOTALL)
                if questions_match:
                    questions_str = questions_match.group(0)
                    # Safety measure: replace any potential code execution
                    questions_str = questions_str.replace('__', '')
                    questions = eval(questions_str)
                else:
                    # Fallback: extract questions line by line
                    questions = [line.strip() for line in questions_text.split('\n') 
                               if line.strip() and '?' in line]
                    
                return questions[:num_questions]  # Ensure we return exactly num_questions
                
            except Exception as e:
                logger.error(f"Error parsing questions: {e}")
                # Fallback method
                questions = [line.strip() for line in questions_text.split('\n') 
                           if line.strip() and '?' in line]
                return questions[:num_questions]
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to generate questions: {response.status_code}")
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise

def answer_question(question, vector_store):
    """Answer a question using RAG approach with the PDF content"""
    logger.info(f"Answering: {question}")
    
    try:
        # Get the most relevant documents for the question
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Answer the following question using only the provided context. 
        If the context doesn't contain the information needed to answer the question, 
        state that you don't have enough information.
        
        Context:
        {context}
        
        Question: {question}
        """
        
        messages = [
            {"role": "system", "content": "You are an educational assistant providing clear, accurate answers based on provided context."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            return answer
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"Error generating answer: {response.status_code}")
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise

# API Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'pdf-question-generator.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing"""
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create a unique session ID
            session_id = str(uuid.uuid4())
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
            file.save(file_path)
            logger.info(f"File saved to {file_path}")
            
            # Process the PDF
            documents = load_pdf(file_path)
            vector_store = initialize_vector_store(documents)
            
            # Save the session data
            active_sessions[session_id] = {
                'file_path': file_path,
                'filename': filename,
                'documents': documents,
                'vector_store': vector_store
            }
            
            # Save vector store
            vector_store_path = os.path.join(VECTOR_STORE_FOLDER, session_id)
            vector_store.save_local(vector_store_path)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'PDF processed successfully',
                'filename': filename,
                'chunks': len(documents)
            })
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/generate-questions', methods=['POST'])
def api_generate_questions():
    """Generate questions from the PDF content"""
    data = request.json
    session_id = data.get('session_id')
    num_questions = data.get('numQuestions', 5)
    
    if not session_id or session_id not in active_sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    
    try:
        # Generate questions
        documents = active_sessions[session_id]['documents']
        questions = generate_questions(documents, num_questions)
        
        # Generate answers for each question
        vector_store = active_sessions[session_id]['vector_store']
        answers = [answer_question(q, vector_store) for q in questions]
        
        return jsonify({
            'success': True,
            'questions': questions,
            'answers': answers
        })
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer-question', methods=['POST'])
def api_answer_question():
    """Answer a specific question about the PDF content"""
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question')
    
    if not session_id or session_id not in active_sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Answer the question
        vector_store = active_sessions[session_id]['vector_store']
        answer = answer_question(question, vector_store)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Check if a session exists and return its info"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': session['filename'],
            'chunks': len(session['documents'])
        })
    else:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and its associated files"""
    if session_id in active_sessions:
        try:
            # Delete the uploaded file
            file_path = active_sessions[session_id]['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete the vector store
            vector_store_path = os.path.join(VECTOR_STORE_FOLDER, session_id)
            if os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)
            
            # Remove from active sessions
            del active_sessions[session_id]
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Session not found'}), 404

# Clean up temporary sessions periodically (run in a separate thread in production)
def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    # This would be better implemented with a proper job scheduler in production
    while True:
        time.sleep(3600)  # Run every hour
        logger.info("Running session cleanup")
        now = time.time()
        to_delete = []
        
        for session_id, session in active_sessions.items():
            # Check if the session has an expiry time, else add one
            if 'expiry' not in session:
                session['expiry'] = now + 3600  # Expire in 1 hour
            
            if session['expiry'] < now:
                to_delete.append(session_id)
        
        # Delete expired sessions
        for session_id in to_delete:
            try:
                file_path = active_sessions[session_id]['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                vector_store_path = os.path.join(VECTOR_STORE_FOLDER, session_id)
                if os.path.exists(vector_store_path):
                    import shutil
                    shutil.rmtree(vector_store_path)
                
                del active_sessions[session_id]
                logger.info(f"Deleted expired session: {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

if __name__ == "__main__":
    # For production, use a proper WSGI server and start the cleanup in a separate thread
    import threading
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)