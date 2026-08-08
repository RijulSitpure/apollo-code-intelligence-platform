"""
rag_assistant.py
-------------------
Module 3: RAG-Powered Code Q&A Assistant.

Retrieves relevant AGC code chunks from the Chroma vector store (Module 2)
and uses Gemini (free tier, via the Google GenAI SDK) to answer questions
about the Apollo Guidance Computer source, with citations back to the
specific file/routine/line range each part of the answer came from.

Design decisions:

1. GROUNDING VIA PROMPT CONSTRAINTS. The system prompt explicitly forbids
   the model from answering using anything outside the retrieved context,
   and requires it to say so when the context is insufficient. This is
   the main lever against hallucination in a RAG system -- retrieval
   quality helps, but the generation step still needs its own guardrail.

2. A RETRIEVAL-CONFIDENCE GATE BEFORE THE LLM IS EVEN CALLED. If the best
   match Chroma returns is still a poor match (distance above threshold),
   we skip the LLM call entirely and tell the user directly rather than
   risk the model rationalizing an answer from weak context. This also
   saves API quota on the free tier.

3. EVERY ANSWER IS CITED. Retrieved chunks carry file/routine/line-range
   metadata (from Module 1 + 2); the prompt requires the model to
   reference these explicitly, and we also print them independently of
   what the model says, so the user always has a way to verify the answer
   against the actual source.

Install (on your machine):
    pip install google-genai chromadb sentence-transformers

Setup:
    1. Get a free API key: https://aistudio.google.com/apikey
    2. export GEMINI_API_KEY="your-key-here"   (or put it in a .env file)

Usage (single question):
    python3 src/rag/rag_assistant.py --query "What does the P63 landing routine do?"

Usage (interactive):
    python3 src/rag/rag_assistant.py
"""

import os
import re
import argparse

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Model name churn is fast and expected here -- Google frequently retires/replaces
# Flash-generation models (this project alone saw gemini-2.5-flash retired for new
# users within weeks). Rather than hardcode a specific versioned model, we default
# to Google's documented evergreen alias, which auto-updates to the current Flash
# release. Override with --model if you want to pin a specific version, or if this
# alias itself changes behavior -- run `client.models.list()` to see what's live.
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_PERSIST_DIR = "data/vector_store"
DEFAULT_COLLECTION = "agc_routines"
TOP_K = 5

# Chroma's default distance metric here is squared L2 over normalized
# embeddings. Empirically (see Module 2's sanity check), well-matched
# results sit under ~1.0; this threshold is a starting point -- tune it
# against your own query set once you've run real queries.
DISTANCE_THRESHOLD = 1.3

SYSTEM_INSTRUCTIONS = """You are a code documentation assistant for the Apollo Guidance Computer (AGC) source code -- the real flight software that flew Apollo 11.

Rules you must follow:
1. Answer ONLY using the provided source code excerpts below. Do not use outside knowledge about Apollo 11 or the AGC.
2. If the excerpts don't contain enough information to answer the question, say so plainly instead of guessing.
3. When you reference a routine, name it explicitly (e.g. "the RGVGCALC routine").
4. Keep answers concise and technically precise -- this is for an engineer trying to understand legacy code, not a general audience.
5. Do not invent line numbers, file names, or routine names that are not present in the excerpts.
"""


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Hybrid retrieval: exact identifier matching + semantic search
# ----------------------------------------------------------------------
#
# Dense embeddings are good at matching MEANING ("landing guidance" ->
# routines about landing guidance) but bad at matching arbitrary
# identifiers ("RGVGCALC" isn't a word -- the model has no real notion
# of what it "means", so semantic distance to the right document can
# stay high even for an exact-name question). We fix this by detecting
# AGC-style identifiers in the query and looking them up directly in
# Chroma's metadata index (an exact match, not a similarity search)
# alongside the normal semantic search, then merging both result sets.

# AGC routine/label names are conventionally uppercase, 3+ chars, and may
# start with a digit or contain digits/slashes/question marks/hyphens
# (e.g. RGVGCALC, TTF/8CL, P67NOW?, 1406ALM, -DEC103). The lookahead
# requires at least one uppercase letter so we don't treat plain numbers
# as candidates; no trailing \b, since AGC names can end in punctuation
# like "?" that a word boundary would otherwise strip.
IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?=[A-Z0-9/\?\-\+\.]*[A-Z])[A-Z0-9\-\+][A-Z0-9/\?\-\+\.]{2,}")


def extract_identifier_candidates(query: str) -> list:
    """Pull out tokens from the query that look like AGC routine names."""
    return list(set(IDENTIFIER_PATTERN.findall(query)))


def exact_match_retrieve(collection, candidates: list) -> list:
    """Direct metadata lookup for candidate routine names -- no embedding involved."""
    if not candidates:
        return []
    results = collection.get(
        where={"routine": {"$in": candidates}},
        include=["documents", "metadatas"],
    )
    matches = []
    for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        matches.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": 0.0,  # exact match -- treated as maximally confident
        })
    return matches


def retrieve(collection, query: str, k: int = TOP_K):
    """Query the Chroma collection and return results in a simplified form."""
    results = collection.query(query_texts=[query], n_results=k)
    retrieved = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ):
        retrieved.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })
    return retrieved


def hybrid_retrieve(collection, query: str, k: int = TOP_K):
    """
    Combines exact identifier lookup with semantic search. Exact matches
    (distance=0.0) are always placed first, since they're more trustworthy
    than any semantic match. Deduplicates by chunk id.
    """
    candidates = extract_identifier_candidates(query)
    exact_matches = exact_match_retrieve(collection, candidates)
    semantic_matches = retrieve(collection, query, k=k)

    seen_ids = set()
    merged = []
    for chunk in exact_matches + semantic_matches:
        if chunk["id"] not in seen_ids:
            merged.append(chunk)
            seen_ids.add(chunk["id"])

    return merged[:max(k, len(exact_matches))]


def should_answer(retrieved: list, threshold: float = DISTANCE_THRESHOLD) -> bool:
    """Confidence gate: don't even call the LLM if the best match is weak."""
    if not retrieved:
        return False
    return retrieved[0]["distance"] <= threshold


# ----------------------------------------------------------------------
# Prompt construction (pure logic -- no external dependencies, testable)
# ----------------------------------------------------------------------

def format_citation(chunk: dict) -> str:
    m = chunk["metadata"]
    return f"[{m['file']} :: {m['routine']}, lines {m['start_line']}-{m['end_line']}]"


def build_prompt(query: str, retrieved: list) -> str:
    context_blocks = []
    for chunk in retrieved:
        citation = format_citation(chunk)
        context_blocks.append(f"--- Source {citation} ---\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"SOURCE CODE EXCERPTS:\n{context_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite routines by name, and mention which excerpt(s) you used):"
    )
    return prompt


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def generate_answer(client, model: str, prompt: str) -> str:
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


# ----------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------

def answer_question(collection, client, model: str, query: str, k: int = TOP_K, verbose: bool = True):
    retrieved = hybrid_retrieve(collection, query, k=k)

    if not should_answer(retrieved):
        return {
            "answer": (
                "I couldn't find sufficiently relevant code in the indexed Apollo "
                "Guidance Computer source to answer that confidently. Try rephrasing, "
                "or this may genuinely be outside what's in the dataset."
            ),
            "citations": [],
            "retrieved": retrieved,
        }

    prompt = build_prompt(query, retrieved)
    answer_text = generate_answer(client, model, prompt)
    citations = [format_citation(c) for c in retrieved]

    if verbose:
        print(f"\n[retrieved {len(retrieved)} chunks, best distance={retrieved[0]['distance']:.3f}]")

    return {
        "answer": answer_text,
        "citations": citations,
        "retrieved": retrieved,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Q&A assistant over the Apollo Guidance Computer source.")
    parser.add_argument("--query", default=None, help="Ask a single question and exit. Omit for interactive mode.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--list-models", action="store_true",
                         help="List models your API key can currently call, then exit.")
    args = parser.parse_args()

    try:
        import chromadb
        from google import genai
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}.\n"
            f"Run: pip install google-genai chromadb sentence-transformers"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
            "then run: export GEMINI_API_KEY='your-key-here'"
        )

    client = genai.Client(api_key=api_key)

    if args.list_models:
        print("Models available to your API key (that support generateContent):\n")
        for m in client.models.list():
            if "generateContent" in getattr(m, "supported_actions", []):
                print(f"  {m.name}")
        return

    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.get_collection(args.collection)

    def ask(query):
        result = answer_question(collection, client, args.model, query, k=args.k)
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(result["answer"])
        if result["citations"]:
            print("\nSOURCES:")
            for c in result["citations"]:
                print(f"  {c}")
        print("=" * 60)

    if args.query:
        ask(args.query)
    else:
        print("Apollo Code Intelligence Platform -- interactive Q&A. Type 'exit' to quit.\n")
        while True:
            query = input("\nAsk about the AGC source > ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if query:
                ask(query)


if __name__ == "__main__":
    main()