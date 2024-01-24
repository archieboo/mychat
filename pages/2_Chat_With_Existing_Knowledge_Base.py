import os
import sys
import streamlit as st
import shutil
from dotenv import load_dotenv
    
from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


def load_vectorstore(path):
    vectorstore = Chroma(persist_directory=path,
                         embedding_function=OpenAIEmbeddings())
    return vectorstore

def get_conversation_chain(vectorstore, model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model_name=model_name, temperature=0.4)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(search_type="similarity", 
                                           search_kwargs={"k": 6}),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def get_prompt(question, retriever):
    template = """context:

    {context}

    Question: {question}

    Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
    )
    return rag_chain

def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with prepared vector stores",
                       page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with prepared vector stores :computer:")
    user_question = st.text_input("Ask a question about your knowdge database:")
    if user_question:
        # user_prompt = get_prompt(user_question, retriever=st.session_state.conversation.retriever)
        # handle_userinput(user_prompt)
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your vectorstore, e.g. Chroma")
        vectorstore_path = st.text_input("Path to vectorstore")
        select_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
        if st.button("Process") and select_model:
            st.write(f'Processing {vectorstore_path}:')
            with st.spinner("Processing"):
                # create vector store
                vectorstore = load_vectorstore(vectorstore_path)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, model_name=select_model)
                
                st.success(f"Done! Running with {select_model} API.")


if __name__ == "__main__":

    
    main()