#!/usr/bin/env python3
import os
import sys
import argparse


def main(args):

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'donotsync'))
    import authentications
    os.environ["OPENAI_API_KEY"] = authentications.APIKEY

    # Load Chroma db
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    vectorstore = Chroma(persist_directory=args.d,
                         embedding_function=OpenAIEmbeddings())

    # retrieve data
    retriever = vectorstore.as_retriever(search_type="similarity", 
                                        search_kwargs={"k": 8})


    # language model
    from langchain_openai import ChatOpenAI

    model_name = {'A':'gpt-3.5-turbo','B':'gpt-4'}
    llm = ChatOpenAI(model_name=model_name[args.m], temperature=0.3)

    # set up prompt
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """context:

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
    parser = argparse.ArgumentParser(description='This script is used to ask a question where a knowledge base is provided (chromadb)')
    parser.add_argument('-q', type=str, help='the question to ask')
    parser.add_argument('-d', type=str, help='path to persisted Chroma db')
    # choose gpt model ,default is gpt-3.5-turbo
    parser.add_argument('-m', type=str, default='A',help='gpt model name, choose A: gpt-3.5-turbo, B: gpt-4')
    args = parser.parse_args()
    main(args)
