import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Load environment variables from .env file
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
def create_rag_pipeline(llm_model_name="tiiuae/falcon-7b-instruct", faiss_index=None):
    """Creates the RAG pipeline."""

    # Check if the API token is available
    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found in environment variables. "
            "Please set it in your .env file."
        )

    # Initialize the language model
    llm = HuggingFaceHub(
        repo_id=llm_model_name,
        model_kwargs={"temperature": 0.1, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
        retriever=faiss_index.as_retriever() if faiss_index else None,
        return_source_documents=True
    )

    return qa_chain

def run_rag_pipeline(query, rag_pipeline):
    """Runs the RAG pipeline and returns the result."""
    result = rag_pipeline({"query": query})
    return result

if __name__ == '__main__':
    # Example usage
    from data_loader import load_data
    from embedding import create_embeddings, chunk_data
    from vector_database import create_faiss_index, load_faiss_index

    # Load data, create embeddings, and FAISS index
    data = load_data('data/data.json')
    chunks = chunk_data(data)
    embeddings = create_embeddings()
    faiss_index = create_faiss_index(chunks, embeddings, "my_faiss_index")
    #faiss_index = load_faiss_index("my_faiss_index", embeddings) # Load existing index

    # Create the RAG pipeline
    try:
        rag_pipeline = create_rag_pipeline(faiss_index=faiss_index)
    except ValueError as e:
        print(f"Error creating RAG pipeline: {e}")
        exit()

    # Run the pipeline
    query = "What is the capital of France?"
    result = run_rag_pipeline(query, rag_pipeline)

    print("Question:", query)
    print("Answer:", result["result"])
    print("Source Documents:")
    for doc in result["source_documents"]:
        print(doc.page_content)