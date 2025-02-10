
from data_loader import load_data
from embedding import create_embeddings, chunk_data
from vector_database import create_faiss_index, load_faiss_index
from rag_pipeline import create_rag_pipeline, run_rag_pipeline
import os

def main():
    # 1. Load Data
    data = load_data('data/data.json')

    # 2. Create Chunks
    chunks = chunk_data(data)

    # 3. Create Embeddings
    embeddings = create_embeddings()

    # 4. Create or Load FAISS Index
    try:
        faiss_index = load_faiss_index("my_faiss_index", embeddings)
        print("Loaded existing FAISS index.")
    except:
        faiss_index = create_faiss_index(chunks, embeddings, "my_faiss_index")
        print("Created new FAISS index.")

    # 5. Create RAG Pipeline
    rag_pipeline = create_rag_pipeline(faiss_index=faiss_index)  # Pass the token here

    # 6. Get User Query
    query = input("Enter your question: ")

    # 7. Run RAG Pipeline
    result = run_rag_pipeline(query, rag_pipeline)

    # 8. Print Results
    print("Question:", query)
    print("Answer:", result["result"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(doc.page_content)

if __name__ == "__main__":
    main()


### USE for OPEN SOURCE DATABASE FROM HUGGINGFACE

# from datasets import load_dataset
# from embedding import create_embeddings, chunk_data
# from vector_database import create_faiss_index, load_faiss_index
# from rag_pipeline import create_rag_pipeline, run_rag_pipeline

# def load_data_from_huggingface(dataset_name, text_column="text", split="train"):
#     """Loads data from a Hugging Face dataset."""
#     try:
#         dataset = load_dataset(dataset_name, split=split)
#         data = [{"text": row[text_column]} for row in dataset]
#         return data
#     except Exception as e:
#         print(f"Error loading dataset {dataset_name}: {e}")
#         return None

# def main():
#     # 1. Load Data from Hugging Face Datasets
#     dataset_name = "rajpurkar/squad_v2"  # Replace with your desired dataset
#     text_column = "context"  # Replace with the name of the text column in your dataset
#     split = "validation" # Replace with the split you want to use (e.g., "train", "validation", "test")

#     data = load_data_from_huggingface(dataset_name, text_column, split)

#     if data is None:
#         print("Failed to load data. Exiting.")
#         return

#     # 2. Create Chunks
#     chunks = chunk_data(data)


#     # 3. Create Embeddings
#     embeddings = create_embeddings()

#     # 4. Create or Load FAISS Index
#     try:
#         faiss_index = load_faiss_index("my_faiss_index", embeddings)
#         print("Loaded existing FAISS index.")
#     except:
#         faiss_index = create_faiss_index(chunks, embeddings, "my_faiss_index")
#         print("Created new FAISS index.")

#     # 5. Create RAG Pipeline
#     rag_pipeline = create_rag_pipeline(faiss_index=faiss_index)

#     # 6. Get User Query
#     query = input("Enter your question: ")

#     # 7. Run RAG Pipeline
#     result = run_rag_pipeline(query, rag_pipeline)

#     # 8. Print Results
#     print("Question:", query)
#     print("Answer:", result["result"])
#     print("\nSource Documents:")
#     for doc in result["source_documents"]:
#         print(doc.page_content)

# if __name__ == "__main__":
#     main()