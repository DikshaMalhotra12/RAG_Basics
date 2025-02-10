import json

def load_data(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_data_from_txt(file_path):
    """Loads data from a text file, treating each line as a document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [{"text": line.strip()} for line in f]  # Create a list of dictionaries
    return data

def load_data_from_csv(file_path, text_column="text"):
    """Loads data from a CSV file, specifying the column containing the text."""
    import pandas as pd
    df = pd.read_csv(file_path)
    data = [{"text": row[text_column]} for _, row in df.iterrows()]
    return data


if __name__ == '__main__':
    # Example usage
    file_path = 'data/data.json'  # Replace with your actual file path
    data = load_data(file_path)
    print(f"Loaded {len(data)} documents.")
    print(data[0]) # Print the first document