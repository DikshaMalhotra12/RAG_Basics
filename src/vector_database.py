from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def create_faiss_index(chunks, embeddings, index_name="my_faiss_index"):
    """Creates a FAISS index from the text chunks and embeddings."""
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_name)
    return db

def load_faiss_index(index_name, embeddings):
    """Loads a FAISS index from disk."""
    db = FAISS.load_local(index_name, embeddings)
    return db

if __name__ == '__main__':
    # Example usage
    from data_loader import load_data
    from embedding import create_embeddings, chunk_data

    data = load_data('data/data.json')
    chunks = chunk_data(data)
    embeddings = create_embeddings()
    db = create_faiss_index(chunks, embeddings, "my_faiss_index")

    # Example query
    query = "What is the capital of France?"
    results = db.similarity_search(query)
    print(f"Found {len(results)} results.")
    print(results[0].page_content)