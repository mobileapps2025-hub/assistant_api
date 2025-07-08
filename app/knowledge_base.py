from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# This function will set up your knowledge base at startup
def initialize_knowledge_base():
    """
    Loads the guide, splits it into chunks, creates embeddings,
    and stores them in a Chroma vector database. Returns a retriever object.
    """
    print("Initializing knowledge base with ChromaDB...")
    loader = TextLoader('spotplan_guide.md', encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embeddings, collection_name="spotplan-guide")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # k=3 retrieves the top 3 most relevant chunks
    print("Knowledge base initialization complete.")
    return retriever

# This function will query the knowledge base
def query_knowledge_base(query: str, retriever):
    """
    Queries the knowledge base and returns a direct answer for "how-to" questions.
    """
    if not retriever:
        return "Sorry, the knowledge base is not available at the moment."

    # We create a QA chain to ask the retriever for information
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False # We don't need to see the source docs in the output
    )

    # The prompt for the QA chain itself
    qa_chain.combine_documents_chain.llm_chain.prompt.template = """
    You are an expert on the Spotplan application. Use the following context to answer the user's question precisely and helpfully.
    Provide clear, step-by-step instructions. Do not mention your context or that you are an AI.
    Context: {context}
    Question: {question}
    Helpful Answer:
    """

    print(f"Querying knowledge base with: {query}")
    result = qa_chain.invoke({"query": query})
    return result.get("result", "I could not find an answer in the knowledge base.")

