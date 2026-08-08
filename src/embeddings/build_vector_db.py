import json
import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# Define Paths based on your existing structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PARSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'full_parsed')
DB_DIR = os.path.join(BASE_DIR, 'data', 'vector_db')

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Initialize the embedding model (downloads automatically the first time)
print("Loading all-MiniLM-L6-v2 embedding model...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or reset the collection
collection_name = "apollo_agc_codebase"
try:
    chroma_client.delete_collection(name=collection_name)
except Exception: # Broadened to catch Chroma's specific NotFoundError
    pass # Collection doesn't exist yet

collection = chroma_client.create_collection(
    name=collection_name, 
    embedding_function=sentence_transformer_ef,
    metadata={"description": "Apollo 11 AGC source code embeddings"}
)

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def format_routine_for_embedding(routine):
    """
    Combines the routine metadata, comments, and code into a single block of text.
    This ensures the embedding model captures the English context alongside the assembly.
    """
    # Extract comments if they exist; Module 1 might format this differently, adjust keys as needed
    comments = "\n".join(routine.get('comments', []))
    code = "\n".join(routine.get('instructions', []))
    
    formatted_text = f"ROUTINE NAME: {routine.get('name', 'UNKNOWN')}\n"
    if comments:
        formatted_text += f"COMMENTS:\n{comments}\n"
    formatted_text += f"CODE:\n{code}"
    
    # MiniLM max token limit is 256. If a routine is massive, we truncate the raw string.
    # A more advanced chunker could split it, but for V1, truncation keeps it simple.
    # ~4 chars per token, so 1000 chars is a safe rough limit for the text string.
    return formatted_text[:1000]

def process_and_embed(filename, module_name):
    filepath = os.path.join(PARSED_DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\nProcessing {module_name} ({filename})...")
    routines = load_json_data(filepath)
    
    ids = []
    documents = []
    metadatas = []

    # Prepare batches
    for idx, routine in enumerate(tqdm(routines)):
        routine_name = routine.get('name', f'unnamed_routine_{idx}')
        
        # 1. Format text chunk
        chunk_text = format_routine_for_embedding(routine)
        
        # 2. Extract Metadata (Crucial for RAG citations later)
        metadata = {
            "module": module_name,
            "routine_name": routine_name,
            "file_source": routine.get('filename', 'Unknown'),
            # Include cross-references if you extracted them in Module 1
            "cross_references": ",".join(routine.get('calls', [])) 
        }

        ids.append(f"{module_name}_{routine_name}_{idx}")
        documents.append(chunk_text)
        metadatas.append(metadata)

    # Insert into ChromaDB in batches to prevent memory crashes
    batch_size = 500
    print(f"Embedding and storing {len(documents)} chunks in ChromaDB...")
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

if __name__ == "__main__":
    # Process both the Command Module and Lunar Module parsed JSONs
    process_and_embed('comanche.json', 'Comanche055')
    process_and_embed('luminary.json', 'Luminary099')
    
    print(f"\nSuccess! Vector database populated at: {DB_DIR}")
    print(f"Total documents embedded: {collection.count()}")