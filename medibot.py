import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load database as cache
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vectorstore: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512
        )
        # st.write("LLM initialized:", str(llm))  # Debugging step for Streamlit
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def main():
    st.title("ASK FROM MEDCON CHATBOT!")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display chat history
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Pass your Question Here")

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state['messages'].append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return

            llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
            if llm is None:
                return

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Combine result and source documents for display
            result_to_show = f"{result}\n\n\n\nSource Documents:\n{str(source_documents)}"

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result_to_show)
            st.session_state['messages'].append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == '__main__':
    main()