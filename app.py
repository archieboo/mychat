import os
import shutil
import sys

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from htmlTemplates import bot_template, css, user_template


def load_and_chunk(pdf_docs):

    print("loading pdf files")
    for i, pdf in enumerate(pdf_docs):
        temp_dir = os.mkdir("temp") if not os.path.exists("temp") else "temp"
        filename = pdf.name
        save_path = os.path.join(temp_dir, f"{filename}.pdf")
        with open(save_path, "wb") as out_file:
            out_file.write(pdf.read())

    # Load the saved PDF file
    loader = PyPDFDirectoryLoader(temp_dir)
    docs = loader.load_and_split()

    # split documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # remove tem directory
    shutil.rmtree(temp_dir)

    return all_splits


def getVectorstore(all_splits):

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings()
    )

    # retrieve data
    # retriever = vectorstore.as_retriever(search_type="similarity",
    #                                     search_kwargs={"k": 6})

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        ),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_prompt(question, retriever):
    template = """context:

    {context}

    Question: {question}

    Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | custom_rag_prompt
    return rag_chain


def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        # user_prompt = get_prompt(user_question, retriever=st.session_state.conversation.retriever)
        # handle_userinput(user_prompt)
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                text_chunks = load_and_chunk(pdf_docs)

                # create vector store
                vectorstore = getVectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":

    main()

