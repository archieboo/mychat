#!/usr/bin/env python3

import os
import sys
import argparse
from dotenv import load_dotenv


def main(args):

    load_dotenv()

    # directory loader
    from langchain_community.document_loaders.directory import DirectoryLoader

    loaderdir = DirectoryLoader(args.f)
    docs = loaderdir.load_and_split()

    # split documents
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # vectorize and index
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    vectorstore = Chroma.from_documents(documents=all_splits, 
                                        embedding=OpenAIEmbeddings())

    # retrieve data
    retriever = vectorstore.as_retriever(search_type="similarity", 
                                        search_kwargs={"k": 6})


    # language model
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # set up prompt
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be precise and concise with your answers, but do not omit any important information.

    {context}

    Question: {question}

    Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # ask question
    question = args.q
    print(rag_chain.invoke(f"{question}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask a question')
    parser.add_argument('-q', type=str, help='the question to ask')
    parser.add_argument('-f', type=str, help='the directory containing the contextual data')
    args = parser.parse_args()
    main(args)
