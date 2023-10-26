import os
import streamlit as st
from streamlit.logger import get_logger
# import tkinter as tk
# from tkinter import filedialog
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    get_qa_rag_chain
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import load_tools
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from agent import get_agent_executor

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

def process_documents(language, directory) -> (str, Neo4jVector):
    print("File chunking begins...", language, directory)
    
    # Create a dictionary mapping languages to file extensions
    language_suffix_mapping = {
        Language.CPP: ".cpp",
        Language.GO: ".go",
        Language.JAVA: ".java",
        Language.KOTLIN: ".kt",
        Language.JS: ".js",
        Language.TS: ".ts",
        Language.PHP: ".php",
        Language.PROTO: ".proto",
        Language.PYTHON: ".py",
        Language.RST: ".rst",
        Language.RUBY: ".rb",
        Language.RUST: ".rs",
        Language.SCALA: ".scala",
        Language.SWIFT: ".swift",
        Language.MARKDOWN: ".md",
        Language.LATEX: ".tex",
        Language.HTML: ".html",
        Language.SOL: ".sol",
        Language.CSHARP: ".cs",
    }
    # Get the corresponding suffix based on the selected language
    suffix = language_suffix_mapping.get(language, "")
    print("language file extension:", suffix)

    loader = GenericLoader.from_filesystem(
        path=directory,
        glob="**/*",
        suffixes=[suffix],
        parser=LanguageParser(language=language, parser_threshold=500)
    )
    documents = loader.load()
    print("Total documents:", len(documents))
    if len(documents) == 0:
        return ("0 documents found", None)

    text_splitter = RecursiveCharacterTextSplitter.from_language(language=language, 
                                                               chunk_size=5000, 
                                                               chunk_overlap=500)

    chunks = text_splitter.split_documents(documents)
    print("Chunks:", len(chunks))

    hashStr = str(abs(hash(directory)))

    # Store the chunks part in db (vector)
    vectorstore = Neo4jVector.from_documents(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name=f"index_{hashStr}",
        node_label=f"node_{hashStr}",
        pre_delete_collection=True,  # Delete existing data
    )

    print("Files are now chunked up")

    return (None, vectorstore)