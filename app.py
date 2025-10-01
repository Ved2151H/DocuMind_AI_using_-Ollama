import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Import templates
import htmlTemplates as ht


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
    # Page config
    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="🧠",  # only 1 emoji kept
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    st.markdown(ht.css, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("DocuMind AI 🧠")   # simple, one emoji
        st.markdown("#### Your Intelligent PDF Assistant")
        st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📂 Document Upload")  # one emoji for sidebar
        pdf_docs = st.file_uploader(
            "Choose PDF files", accept_multiple_files=True, type="pdf"
        )

        st.markdown("---")
        st.header("⚙️ Settings")
        selected_model = st.selectbox("AI Model", ["llama2", "mistral", "mixtral"])

        if pdf_docs:
            st.info(f"{len(pdf_docs)} file(s) uploaded")
            if st.button("Process Documents"):
                with st.spinner("Processing PDFs... Please wait"):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(chunks, selected_model)
                    st.session_state.conversation = get_conversation_chain(vectorstore, selected_model)
                st.success("Ready to chat!")

    # Main chat interface
    if "conversation" in st.session_state and st.session_state.conversation:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(ht.user_message(message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(ht.bot_message(message["content"]), unsafe_allow_html=True)

        user_input = st.text_input("Ask a question", placeholder="Type your question...")

        if st.button("Send") and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_input})
                answer = response["answer"]
            st.session_state.messages.append({"role": "bot", "content": answer})
            st.rerun()
    else:
        st.markdown(ht.welcome_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
