# RAG with Gemini, ChromaDB, and Hugging Face Datasets

This project demonstrates a basic Retrieval Augmented Generation (RAG) pipeline using:

*   **LLM:** Google Gemini API
*   **Vector Database:** ChromaDB
*   **Dataset:** [Name of your Hugging Face dataset] (Link to the dataset)
*   **Language:** Python

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [your repository URL]
    cd rag-gemini-chromadb
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key:**
    *   Obtain an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
    *   Set the environment variable `GOOGLE_API_KEY` with your key.
        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY"  # Linux/macOS
        set GOOGLE_API_KEY=YOUR_API_KEY      # Windows
        ```

## Usage

1.  **Run the `main.py` script:**
    ```bash
    python src/main.py
    ```

    This script will:
    *   Load the dataset.
    *   Embed the text data using the Gemini API.
    *   Create a ChromaDB database.
    *   Index the embeddings.
    *   Run a sample query.

## Example Query
The code includes a sample query.  Feel free to modify it in `src/main.py`.

## Contributing

Contributions are welcome! Please submit a pull request.

## License

[Choose a license, e.g., MIT License]