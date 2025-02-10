import chromadb
from chromadb.utils import embedding_functions

def create_chroma_client(persist_directory="chroma_db"):
    """Creates and returns a ChromaDB client."""
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def create_collection(client, collection_name="my_collection"):
    """Creates a ChromaDB collection."""
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def add_data_to_collection(collection, ids, embeddings, metadatas=None, documents=None):
    """Adds data to a ChromaDB collection."""
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def query_collection(collection, query_embedding, n_results=5):
    """Queries a ChromaDB collection with a given embedding."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

if __name__ == '__main__':
    # Example usage:
    client = create_chroma_client()
    collection = create_collection(client, "my_test_collection")

    # Sample data
    sample_ids = ["doc1", "doc2"]
    sample_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] # Replace with actual embeddings
    sample_metadatas = [{"author": "John"}, {"author": "Jane"}]
    sample_documents = ["This is document 1.", "This is document 2."]

    add_data_to_collection(collection, sample_ids, sample_embeddings, sample_metadatas, sample_documents)

    query_embedding = [0.2, 0.3, 0.4]  # Replace with actual query embedding
    results = query_collection(collection, query_embedding)
    print("Query results:")
    print(results)