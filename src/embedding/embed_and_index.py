"""
embed_and_index.py
--------------------
Module 2 (part 2): Embeddings + Vector Store.

Embeds chunks (from chunk_builder.py) using a local sentence-transformers
model and stores them in a persistent Chroma collection for retrieval in
Module 3 (RAG assistant).

Why all-MiniLM-L6-v2 by default: it's a small, fast, general-purpose text
embedding model. Since most of the retrievable signal in this dataset
lives in the English comments rather than AGC-specific assembly syntax
(which no pretrained code model has ever seen), a general text embedder
is a better fit here than a code-specific one -- see project docs for
the full reasoning.

Install (on your machine, not in a network-isolated sandbox):
    pip install sentence-transformers chromadb

Usage:
    python3 src/embedding/embed_and_index.py data/chunks/chunks.jsonl \
        --persist-dir data/vector_store \
        --collection agc_routines \
        --model all-MiniLM-L6-v2
"""

import json
import argparse
import os


def load_chunks(path: str) -> list:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def flatten_metadata(metadata: dict) -> dict:
    """
    Chroma metadata values must be str/int/float/bool -- no lists/dicts.
    'references' is a list, so we join it into a comma-separated string,
    and keep an integer count alongside for filtering/sorting.
    """
    flat = dict(metadata)
    refs = flat.get("references", [])
    flat["references"] = ", ".join(refs) if refs else ""
    flat["n_references"] = len(refs)
    return flat


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and index them in a persistent Chroma vector store.")
    parser.add_argument("chunks_path", help="Path to chunks.jsonl from chunk_builder.py")
    parser.add_argument("--persist-dir", default="data/vector_store", help="Directory for the persistent Chroma DB")
    parser.add_argument("--collection", default="agc_routines", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    # Imported here (not at module top) so this file can still be inspected/tested
    # in environments where these packages aren't installed yet.
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}.\n"
            f"Run: pip install sentence-transformers chromadb"
        )

    chunks = load_chunks(args.chunks_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {args.chunks_path}")

    print(f"Loaded {len(chunks)} chunks from {args.chunks_path}")
    print(f"Loading embedding model: {args.model} (first run will download it)...")
    model = SentenceTransformer(args.model)

    os.makedirs(args.persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=args.persist_dir)

    # Recreate the collection fresh each run so re-indexing doesn't duplicate entries
    existing = [c.name for c in client.list_collections()]
    if args.collection in existing:
        client.delete_collection(args.collection)
    collection = client.create_collection(args.collection)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [flatten_metadata(c["metadata"]) for c in chunks]

    total = len(texts)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_texts = texts[start:end]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids[start:end],
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end}/{total} chunks...", end="\r")

    print(f"\nDone. {total} chunks indexed into collection '{args.collection}' at {args.persist_dir}")

    # Quick sanity-check query so you know retrieval actually works before Module 3
    sample_query = "landing guidance"
    results = collection.query(
        query_texts=[sample_query],
        n_results=3,
    )
    print(f"\nSanity check -- top 3 results for query '{sample_query}':")
    for doc_id, doc, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
        first_line = doc.strip().split("\n")[1]  # the 'Routine: X' line
        print(f"  [{dist:.3f}] {doc_id}  -> {first_line}")


if __name__ == "__main__":
    main()