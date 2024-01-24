import os, tempfile
import streamlit as st
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai

# Streamlit app
st.subheader('Generative Q&A with LangChain & Pinecone (using Sentence Transformers)')

# Get Google API key (if using Cloud TPU) and Pinecone details
with st.sidebar:
    google_api_key = st.text_input("Google API key (optional for Cloud TPU)", type="password")  # Add if needed
    pinecone_api_key = st.text_input("Pinecone API key", type="password")
    pinecone_env = st.text_input("Pinecone environment")
    pinecone_index = st.text_input("Pinecone index name")
source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
query = st.text_input("Enter your query")

if st.button("Submit"):
    # Validate inputs
    if not pinecone_api_key or not pinecone_env or not pinecone_index or not source_doc or not query:
        st.warning(f"Please upload the document and provide the missing fields.")
    else:
        try:
            # Save uploaded file temporarily, load and split, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)

            # Generate embeddings using Sentence Transformers, insert into Pinecone, create retriever
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
            embeddings = SentenceTransformerEmbeddings("all-mpnet-base-v2")  # Use a suitable model
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            retriever = vectordb.as_retriever()

            genai.configure(api_key=google_api_key)

            model = genai.GenerativeModel('gemini-pro')
            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            qa = RetrievalQA.from_chain_type(model, chain_type="stuff", retriever=retriever)
            response = qa.run(query)

            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
