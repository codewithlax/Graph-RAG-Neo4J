import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Chat_GROQ with LLAMA3")

llm = ChatGroq(groq_api_key=groq_api_key,
               model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context only. Provide the most accurate answer based on the question.
context: {context}
question: {input}
                                          """)

prompt1 = st.text_input("Ask a question from the document")





def vector_embedding():
    if 'vectors' not in  st.session_state:
        
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFLoader("meditation.pdf")
        st.session_state.doc = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings)
    

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Embedding is done!")



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({"input":prompt1})
    st.write(response['answer'])