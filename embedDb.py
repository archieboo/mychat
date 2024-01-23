#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime



def main(args):

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'donotsync'))
    import authentications
    os.environ["OPENAI_API_KEY"] = authentications.APIKEY


    # directory of pdf files
    from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader 

    # if arg.s is a .pdf file, load it as a single document
    if args.f.endswith('.pdf'):
        print('loading single pdf file')
        loader = PyPDFLoader(args.f)
    else:
        print('loading directory of pdf files')
        loader = PyPDFDirectoryLoader(args.f)
    docs = loader.load_and_split()

    # split documents
    from langchain.text_splitter import RecursiveCharacterTextSplitter


    print('Load and split documents into chunks.')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # vectorize, embed, and index
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    print(f'Embedding knowledge database and persist to: {args.to}')
    if not os.path.exists(args.to):
        os.makedirs(args.to)
    vectorstore = Chroma.from_documents(documents=all_splits, 
                                        embedding=OpenAIEmbeddings(),
                                        persist_directory=args.to)
    print("Done at ", datetime.now())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        This script create a knowledge database from pdf files.
        It's a text embedding Chromadb. It's used to provide context to questions.
        """)
    parser.add_argument('-f', type=str, help='path to the pdf file(s)')
    parser.add_argument('-to', type=str, help='directory to store embedding')
    args = parser.parse_args()
    main(args)
