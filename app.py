import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def get_vectorstore(chunks, model_name="llama2"):
    embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def get_conversation_chain(vectorstore, model_name="llama2"):
    prompt = PromptTemplate(
        template="""You answer questions based ONLY on this PDF content:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""",
        input_variables=["context", "chat_history", "question"]
    )
    llm = Ollama(model=model_name, base_url="http://localhost:11434", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )


def main():
    st.title("📄 DocuMind AI")
    
    # Sidebar
    st.sidebar.header("Upload PDF Documents")
    pdf_docs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    selected_model = st.sidebar.selectbox("Select Model", ["llama2", "mistral", "mixtral"])
    
    if st.sidebar.button("Process PDFs") and pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(chunks, selected_model)
        st.session_state.conversation = get_conversation_chain(vectorstore, selected_model)
        st.success("Documents processed! You can now ask questions.")

    # Chat interface
    if "conversation" in st.session_state and st.session_state.conversation:
        user_input = st.text_input("Ask a question about your PDFs:")
        if st.button("Send") and user_input:
            response = st.session_state.conversation({"question": user_input})
            st.write("**Answer:**", response["answer"])


if __name__ == "__main__":
    main()
