import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import utils
import streamlit as st
import chromadb
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq.chat_models import ChatGroq
from langchain_community.llms import ollama
import yaml
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import YoutubeLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

st.title("Add YouTube Video Content")


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

llm_config = load_config(f"llm_config.yaml") 

with st.sidebar:
    st.markdown("## LLM Setup")
    llm_provider = st.selectbox("Select LLM Provider", list(llm_config.keys()), key="llm_provider")
    selected_model = st.selectbox("Select Model Name", llm_config[llm_provider], key="model_name")
    api_key = st.text_input("API Key (if needed)", type="password", key="api_key")

if llm_provider == 'openai':
    if api_key:
        llm = ChatOpenAI(model=selected_model, temperature=0, api_key=api_key)
        embeddings_model = OpenAIEmbeddings(api_key=api_key)
    if not api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
elif llm_provider == 'anthropic':
    if api_key:
        llm = ChatAnthropic(model=selected_model, anthropic_api_key=api_key)
        embeddings_model = HuggingFaceEmbeddings()
    if not api_key:
        st.info("Please add your Anthropic API key to continue.")
        st.stop()
elif llm_provider == 'groq':
    if api_key:
        llm = ChatGroq(model_name=selected_model, api_key=api_key)
        embeddings_model = HuggingFaceEmbeddings()
    if not api_key:
        st.info("Please add your Groq API key to continue.")
        st.stop()
else:
    llm = ollama(model=selected_model)
    embeddings_model = OllamaEmbeddings()

collection_names = utils.get_collection_list()

# Create a Streamlit dropdown
existing_kb = st.selectbox('Select an existing knowledge base', 
                           collection_names, 
                           index=None, 
                           placeholder='Select an existing knowledge base')

new_kb = st.text_input("Create a new knowledge base", type="default", key="kb")

if existing_kb:
    selected_collection = existing_kb
else:
    selected_collection = new_kb
    
url = st.text_input("YouTube url:", type="default", key="url")
submit = st.button('Submit')

if url:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    
    if submit:
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=embeddings_model, 
                                            persist_directory='db',
                                            collection_name=selected_collection)



        st.success(f"Added {url} to knowledge base: {selected_collection}")
