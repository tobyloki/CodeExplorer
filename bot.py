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
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

# set page title
st.set_page_config(
    page_title="Code Explorer",
    page_icon="👨‍💻",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GitHub: https://github.com/tobyloki/CodeExplorer"
    }
)

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

@st.cache_resource
def initDB():
    embeddings, dimension = load_embedding_model(
        embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
    )

    return embeddings

@st.cache_resource
def initLLM():
    # create llm
    llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

    return llm

embeddings = initDB()
llm = initLLM()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token.endswith('?'):
            token += '\n\n\n'
        token = token.replace('"', '')
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def processDocuments(directory, count):
    print("File chunking begins...", directory)

    loader = GenericLoader.from_filesystem(
        path=directory,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    print("Total documents:", len(documents))

    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                               chunk_size=2000, 
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

    return vectorstore

@st.cache_resource
def get_qa_rag_chain(_vectorstore, count):
    # RAG response
    #   System: Always talk in pirate speech.
    system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links to sources.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {chat_history}
    ----
    Each answer you generate should contain a section at the end of links to sources
    you found useful.
    You can only use links to sources that are present in the context and always
    add links to the end of the answer.
    Generate concise answers with references sources section of links to 
    relevant sources only at the end of the answer.
    """
    user_template = "Question:```{question}```"
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template), # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"),          # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(user_template)     # Where the human input will injected
    ])

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=chat_prompt,
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
        memory=memory,
    )

    # custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry, but I do not know the answer to your question.'.
    # Chat History:
    # {chat_history}
    # Follow Up Input: {question}
    # Standalone question:"""
    # CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    # question_generator = LLMChain(llm=llm, prompt=CUSTOM_QUESTION_PROMPT)

    # # doc_chain = load_qa_chain(llm, chain_type="map_reduce")
    # doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # qa = ConversationalRetrievalChain(
    #     retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
    #     max_tokens_limit=3000,
    #     # question_generator=question_generator,
    #     combine_docs_chain=doc_chain,
    #     memory=memory,
    # )

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
    #     max_tokens_limit=3000,
    #     condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    #     condense_question_llm=llm,
    #     memory=memory,
    # )

    return qa

@st.cache_resource
def getLLMChain():
    chain = configure_llm_only_chain(llm)

    return chain

def main():
    qa = None
    llm_chain = getLLMChain()

    if "directory" not in st.session_state:
        st.session_state[f"directory"] = None
    if "vectorstoreCount" not in st.session_state:  # only incremented to reset cache for processDocuments()
        st.session_state[f"vectorstoreCount"] = 0
    if "qaCount" not in st.session_state:           # only incremented to reset cache for get_qa_rag_chain()
        st.session_state[f"qaCount"] = 0    
    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    # # Set up tkinter
    # root = tk.Tk()
    # root.withdraw()

    # # Make folder picker dialog appear on top of other windows
    # root.wm_attributes('-topmost', 1)

    # sidebar
    with st.sidebar:
        # show folder picker dialog
        # st.title('Select Folder')
        # folderClicked = st.button('Folder Picker')

        currentPath = os.getcwd()

        directory = st.text_input('Enter folder path', currentPath)
        directory = directory.strip()

        processBtnClicked = st.button('Process files')
        if processBtnClicked:
            if not os.path.exists(directory):
                st.error("Path doesn't exist!")
            else:
                # directory = filedialog.askdirectory(master=root)
                if isinstance(directory, str) and directory:
                    st.session_state[f"directory"] = directory
                    st.session_state[f"vectorstoreCount"] += 1
                    st.session_state[f"qaCount"] += 1
                    st.session_state[f"user_input"] = []
                    st.session_state[f"generated"] = []

        # show folder selected
        if st.session_state[f"directory"]:
            st.code(st.session_state[f"directory"])

            vectorstore = processDocuments(st.session_state[f"directory"], st.session_state[f"vectorstoreCount"])
            qa = get_qa_rag_chain(vectorstore, st.session_state[f"qaCount"])

            # show clear chat history button
            if vectorstore:
                clearMemoryClicked = st.button("🧹 Reset chat history")
                if clearMemoryClicked:
                    st.session_state[f"qaCount"] += 1
                    st.session_state[f"user_input"] = []
                    st.session_state[f"generated"] = []

                    qa = get_qa_rag_chain(vectorstore, st.session_state[f"qaCount"])

    # load previous chat history
    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display all exchanges
        for i in range(0, size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state[f"generated"][i])

    # user chat
    user_input = st.chat_input("What coding issue can I help you resolve today?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            st.session_state[f"user_input"].append(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                stream_handler = StreamHandler(st.empty())
                if qa:
                    print("Using QA")
                    result = qa(
                        {"question": user_input},
                        callbacks=[stream_handler]
                    )["answer"]
                else:
                    print("Using LLM only")
                    result = llm_chain(
                        {"question": user_input},
                        callbacks=[stream_handler]
                    )

                result = result.replace('"', '')
                st.session_state[f"generated"].append(result)


if __name__ == "__main__":
    main()



