import os, tempfile
import streamlit as st
import pinecone
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit app
st.subheader('Generative Q&A with LangChain & Pinecone (using Sentence Transformers)')

# Get Google API key (if using Cloud TPU) and Pinecone details
with st.sidebar:
    google_api_key = st.text_input("Google API key)", type="password")  # Add if needed
    pinecone_api_key = st.text_input("Pinecone API key", type="password")
    pinecone_env = st.text_input("Pinecone environment", placeholder="gcp-starter")
    pinecone_index = st.text_input("Pinecone index name", placeholder="apollo-chatbot")
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
            embeddings = SentenceTransformerEmbeddings()  # Use a suitable model
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            retriever = vectordb.as_retriever()

            model = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-pro", convert_system_message_to_human=True)
            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            qa = RetrievalQA.from_chain_type(model, chain_type="stuff", retriever=retriever)
            response = qa.run(query)

            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
