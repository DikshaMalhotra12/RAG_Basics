from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def create_embeddings(model_name="all-MiniLM-L6-v2"):
    """Creates a Hugging Face embeddings model."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def chunk_data(data, chunk_size=1000, chunk_overlap=0):
    """Chunks the data into smaller pieces."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [item['text'] for item in data]  # Extract text from your data structure
    chunks = text_splitter.create_documents(texts)
    return chunks

if __name__ == '__main__':
    # Example usage
    from data_loader import load_data
    data = load_data('data/data.json')
    chunks = chunk_data(data)
    print(f"Created {len(chunks)} chunks.")
    print(chunks[0].page_content)