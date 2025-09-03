# --- MCL Knowledge Base (Backup/Alternative Implementation) ---
# This file provides an alternative LangChain-based implementation
# The primary implementation uses OpenAI vector stores in services.py

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from pathlib import Path
import os

def initialize_mcl_knowledge_base():
    """
    Alternative implementation using LangChain and ChromaDB.
    Loads MCL documents, creates embeddings, and stores them in ChromaDB.
    """
    print("Initializing MCL knowledge base with ChromaDB...")
    
    documents = []
    documents_path = Path("app/documents")
    
    # Load markdown files
    for md_file in documents_path.glob("*.md"):
        try:
            loader = TextLoader(str(md_file), encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {md_file.name}")
        except Exception as e:
            print(f"Error loading {md_file.name}: {e}")
    
    # Load PDF files
    for pdf_file in documents_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {pdf_file.name}")
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    if not documents:
        print("No documents loaded for knowledge base")
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        texts, 
        embeddings, 
        collection_name="mcl-knowledge-base"
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
    print("MCL knowledge base initialization complete.")
    return retriever

def query_mcl_knowledge_base(query: str, retriever):
    """
    Query the MCL knowledge base using LangChain.
    Returns a comprehensive answer about MCL topics.
    """
    if not retriever:
        return "Sorry, the MCL knowledge base is not available at the moment."

    # Create QA chain with MCL-specific prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    # MCL-specific prompt template
    qa_chain.combine_documents_chain.llm_chain.prompt.template = """
    You are an expert assistant for the MCL (Mobile Checklist) application. 
    Use the provided context to answer questions about MCL features, usage, troubleshooting, and best practices.
    
    Provide clear, detailed, and step-by-step instructions when relevant.
    If the question is about a specific feature, explain how to use it.
    If it's about troubleshooting, provide practical solutions.
    Always be helpful and reference the documentation when possible.
    
    Context: {context}
    Question: {question}
    
    Detailed Answer:
    """

    print(f"Querying MCL knowledge base with: {query}")
    try:
        result = qa_chain.invoke({"query": query})
        return result.get("result", "I could not find specific information about that topic in the MCL knowledge base.")
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return "I encountered an error while searching the knowledge base. Please try rephrasing your question."

