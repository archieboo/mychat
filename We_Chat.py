import streamlit as st

st.set_page_config(
    page_title="Landing page",
    page_icon=":robot:"
)

st.markdown("""
# A chatbot for your own knowledge base

            
Select an option on the left to ask questions about with your own knowledge base.       
- PDFs: Upload a single or multiple PDFs and ask questions about it.
- Your own knowledge base: Enter full path to your Chroma vectorstore and ask questions about it.
            
Future directions:
- Option to ask pubmed questions
- Option to ask arxiv questions
- Add local llm model so that it can be used offline and free!


            

            
*Currently, all queries use OPENAI's API, so it's NOT FREE!*

"""
)

# Change the item names in the sidebar