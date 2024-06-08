import streamlit as st
import utils
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq.chat_models import ChatGroq
from langchain_community.llms import ollama
import yaml
from typing import List, Optional, Dict, Any
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

st.title("Chat with your Knowledge Base")


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

def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']



collection_names = utils.get_collection_list()

# Create a Streamlit dropdown
selected_collection = st.selectbox('Select an existing knowledge base', 
                           collection_names, 
                           index=None, 
                           placeholder='Select an existing collection')

if selected_collection:

    persistent_client = chromadb.PersistentClient(path='db')
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=selected_collection,
        embedding_function=embeddings_model,
    )

    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = langchain_chroma  

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
