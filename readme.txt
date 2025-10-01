
DocuMind AI - Workflow

DocuMind AI is a local AI-powered PDF document assistant. 
It allows users to upload PDFs, process them, 
and interactively ask questions based only on the PDF content.


\[Upload PDFs] 
       ↓
[Read PDFs & Extract Text] 
       ↓
[Split Text into Chunks] 
       ↓
[Generate Embeddings for Chunks] 
       ↓
[Store Embeddings in FAISS Vector Store] 
       ↓
[Build Conversational Retrieval Chain with LLM] 
       ↓
[User Asks Questions in Chat] 
       ↓
[AI Retrieves Relevant Chunks & Generates Answer]
       ↓
[Display Answer to User]



**must download to run**

# DocuMind AI - Setup Guide
# Save this as setup_documind.py if you want to run as a script or just follow the commands in terminal

"""
1. Create Virtual Environment
"""
# Windows
# python -m venv documind_env
# documind_env\Scripts\activate

# Linux/Mac
# python3 -m venv documind_env
# source documind_env/bin/activate

"""
2. Upgrade pip
"""
# pip install --upgrade pip

"""
3. Install Streamlit
"""
# pip install streamlit

"""
4. Install PDF Reading Library
"""
# pip install PyPDF2

"""
5. Install LangChain and LangChain Community Modules
"""
# pip install langchain
# pip install langchain-community

"""
6. Install FAISS (Vector Store)
"""
# Windows
# pip install faiss-cpu

# Linux/Mac
# pip install faiss-cpu
# or GPU support: pip install faiss-gpu

"""
7. Install Ollama LLM Client
"""
# pip install ollama

"""
8. Install missing dependencies
"""
# pip install typing-extensions requests numpy

"""
***for first time***
       python -m venv venv
 
9. Run Ollama Server (Before running the app)
"""
# ollama serve

"""
10. Run Streamlit App
"""
# streamlit run app.py

"""
Optional Commands:
"""
# Check Python version
# python --version

# Check installed packages
# pip list

# Deactivate virtual environment
# deactivate