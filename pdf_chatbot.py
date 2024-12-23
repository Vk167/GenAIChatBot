import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

from htmlTemplates import css, bot_template, user_template


load_dotenv()
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

## PDF to TEXT Extraction ##
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


## Get Chunks of text ##
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks


## Get Vectorize the Chunk text ##
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")


# Get conversational chain ##
def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context, make sure to provide all the details, if the answer is not in the provided
    context then just say, "answer is not available in the context", dont provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
"""

    model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature = 0.3)

    prompt = PromptTemplate(template= prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type = "stuff",prompt = prompt)
    return chain


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#
#     chain = get_conversational_chain()
#
#     response = chain(
#         {"input_documents": docs, "question": user_question}
#         , return_only_outputs=True)
#
#     print(response)
#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config(page_title="Chat PDF", page_icon=":file_pdf:")  # Set title and icon
#
#     # Use st.markdown for HTML content
#     st.markdown("""
#     <style>
#         .custom-text-input input {
#             font-size: 18px;
#             padding: 10px;
#             border: 1px solid #ddd;
#             border-radius: 5px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     user_question = st.text_input("Ask a Question from the PDF Files", key="question_input")
#
#     if user_question:
#         user_input(user_question)
#
#     with st.sidebar:
#         st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)
#
#         # File uploader for PDF files
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
#                                     accept_multiple_files=True)
#
#         # if pdf_docs:
#         #     st.write("Uploaded files:", [doc.name for doc in pdf_docs])  # Debugging output
#
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 try:
#                     if pdf_docs:
#                         raw_text = get_pdf_text(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("Done!")
#                     else:
#                         st.warning("Please upload PDF files.")
#                 except Exception as e:
#                     st.error(f"Error occurred: {e}")


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Check if the response is None
    return response["output_text"] if response["output_text"] else "Sorry, I couldn't find an answer to your question."

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")  # Set title and icon

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Use imported CSS for styling
    st.write(css, unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files", key="question_input")

    if st.button("Submit"):
        if user_question:
            answer = user_input(user_question)

            # Append user question and bot response to chat history
            st.session_state.chat_history.append(("User", user_question))
            st.session_state.chat_history.append(("Bot", answer))


    # Display chat history with HTML templates
    for role, message in st.session_state.chat_history:
        if message:  # Check if message is not None
            if role == "User":
                st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)



    with st.sidebar:
        st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)

        # File uploader for PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    if pdf_docs:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done!")
                    else:
                        st.warning("Please upload PDF files.")
                except Exception as e:
                    st.error(f"Error occurred: {e}")





if __name__ == "__main__":
    main()