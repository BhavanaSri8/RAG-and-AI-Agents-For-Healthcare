import streamlit as st
import warnings
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set API Key (Replace with a valid Groq API Key)
API_KEY = "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"

# Initialize the language model
llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=API_KEY)

# Load and process the PDF file
pdf_path = "health.pdf"  # Ensure this file exists
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(data)

# Set up the FAISS vector store
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = FAISS.from_documents(documents, embeddings)

# Set up the retriever
retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 4})

# Define the prompt template
prompt_template = """
You are a helpful assistant who generates answers only from the provided context.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Answer in a single line.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})

# ------------------- Streamlit UI -------------------

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="🧠 🩺 HealthQuery Chatbot", layout="centered")

# Custom CSS for a colorful background and bordered chatbot box
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
        }
        .chat-container {
            border: 2px solid #6a11cb;
            border-radius: 15px;
            background-color: white;
            padding: 20px;
            width: 80%;
            margin: auto;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-box {
            border: 1px solid #cfd8dc;
            padding: 15px;
            border-radius: 12px;
            background-color: #ffffff;
        }
        .chat-bubble {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            max-width: 75%;
            font-size: 16px;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #6a11cb;
            color: white;
            text-align: right;
            float: right;
            clear: both;
            border-top-right-radius: 0px;
        }
        .bot-bubble {
            background-color: #e0e7ff;
            color: black;
            float: left;
            clear: both;
            border-top-left-radius: 0px;
        }
        .input-box {
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-top: 15px;
        }
        .send-button {
            background: #ff6f61;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px;
            cursor: pointer;
            width: 120px;
            font-size: 16px;
            margin-left: 5px;
        }
        .send-button:hover {
            background: #e63946;
        }
        .header {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #6a11cb;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section with Stethoscope Symbol
st.markdown("<div class='header'>🧠🩺 HealthQuery Chatbot</div>", unsafe_allow_html=True)
st.write("Ask Your Health Query Below:")



# Chat history container
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Close chat container
st.markdown("</div>", unsafe_allow_html=True)

# User input field with send button
user_input = st.text_input("Type your question here...", key="user_input", help="Type your question here")

if st.button("Get Health Advice", key="send_button", help="Click to ask"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process user input and get bot response
        response = qa(user_input)
        bot_reply = response["result"] if "result" in response else "I don't know."

        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        # Rerun the app to display new messages
        st.rerun()
