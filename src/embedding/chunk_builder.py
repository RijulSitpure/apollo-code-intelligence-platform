"""
chunk_builder.py
------------------
Module 2 (part 1): Chunking.

Converts Module 1's parsed routine JSON into embedding-ready text chunks.

Design decisions (documented here since they matter for your report):

1. UNIT OF CHUNKING = ROUTINE, not fixed-length text windows.
   Routines are already semantically coherent (one label = one logical
   subroutine), so naive fixed-length chunking would slice through the
   middle of a routine and destroy that structure for no benefit.

2. LARGE ROUTINES ARE SPLIT. A few routines run 100-180 instructions
   (e.g. IDADDTAB, REMDIST) -- too long for a single embedding to
   represent well, and too long for a RAG answer to cite precisely.
   These are split into overlapping sub-chunks (default: 40 instructions
   per chunk, 8-instruction overlap) so context isn't lost at the seams.

3. CHUNK TEXT PRIORITIZES COMMENTS. Since retrieval signal lives mostly
   in the English comments (see the general vs. code-specific embedding
   model discussion), the rendered chunk text keeps comments inline next
   to their instructions rather than stripping them.

4. DATA-TABLE ROUTINES ARE FLAGGED, NOT DROPPED. Routines like PREAMBLE
   blocks or constant tables (all-DEC/OCT, no branch opcodes) are tagged
   chunk_type="data" instead of "code" -- kept in the index (someone
   might legitimately ask "what are the star vector constants?") but
   distinguishable so Module 5's complexity scoring can exclude them.

Usage:
    python3 src/embedding/chunk_builder.py data/full_parsed/luminary.json data/full_parsed/comanche.json --out data/chunks/chunks.jsonl
"""

import json
import argparse
import os

MAX_INSTRUCTIONS_PER_CHUNK = 40
CHUNK_OVERLAP = 8

# Opcodes that indicate this routine is executable control flow, not just data
CONTROL_FLOW_OPCODES = {
    "TC", "TCF", "BZF", "BZMF", "BMN", "BPL", "CCS", "GOTO", "CALL",
    "INDEX", "EXTEND", "TS", "CA", "CS", "AD", "MASK",
}
# Opcodes that indicate this routine is just data/constants
DATA_OPCODES = {"DEC", "OCT", "2DEC", "=", "ADRES", "EQUALS"}


def classify_routine(routine: dict) -> str:
    """Heuristic: is this routine executable code, or a data/constant table?"""
    if routine["name"] == "PREAMBLE":
        return "data"
    opcodes = [i["opcode"] for i in routine["instructions"] if i.get("opcode")]
    if not opcodes:
        return "data"
    control_count = sum(1 for op in opcodes if op in CONTROL_FLOW_OPCODES)
    data_count = sum(1 for op in opcodes if op in DATA_OPCODES)
    if control_count == 0 and data_count > 0:
        return "data"
    return "code"


def render_instruction(instr: dict) -> str:
    """Render one instruction back into readable pseudo-assembly with its comment."""
    parts = []
    if instr.get("label"):
        parts.append(instr["label"])
    if instr.get("opcode"):
        parts.append(instr["opcode"])
    if instr.get("operands"):
        parts.append(" ".join(instr["operands"]))
    line = "  ".join(parts)
    if instr.get("comment"):
        line += f"   # {instr['comment']}"
    return line


def render_chunk_text(file_name: str, routine_name: str, banner: str, instructions: list) -> str:
    """Compose the final text that gets embedded, comments-forward."""
    header_lines = [f"File: {file_name}", f"Routine: {routine_name}"]
    if banner:
        header_lines.append(f"Section: {banner}")
    body = "\n".join(render_instruction(i) for i in instructions)
    return "\n".join(header_lines) + "\n\n" + body


def build_chunks_for_routine(file_name: str, routine: dict) -> list:
    chunks = []
    instructions = routine["instructions"]
    chunk_type = classify_routine(routine)
    n = len(instructions)

    if n <= MAX_INSTRUCTIONS_PER_CHUNK:
        windows = [(0, n)]
    else:
        windows = []
        start = 0
        step = MAX_INSTRUCTIONS_PER_CHUNK - CHUNK_OVERLAP
        while start < n:
            end = min(start + MAX_INSTRUCTIONS_PER_CHUNK, n)
            windows.append((start, end))
            if end == n:
                break
            start += step

    for idx, (start, end) in enumerate(windows):
        window_instructions = instructions[start:end]
        text = render_chunk_text(file_name, routine["name"], routine.get("banner_comment"), window_instructions)
        chunk_id = f"{file_name}::{routine['name']}::{idx}"
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "file": file_name,
                "routine": routine["name"],
                "chunk_index": idx,
                "n_chunks_in_routine": len(windows),
                "chunk_type": chunk_type,
                "start_line": routine["instructions"][start]["line_no"],
                "end_line": routine["instructions"][end - 1]["line_no"],
                "n_instructions_in_chunk": end - start,
                "references": routine.get("references", []),
            },
        })
    return chunks


def build_chunks(parsed_files: list) -> list:
    all_chunks = []
    for file_data in parsed_files:
        file_name = file_data["filename"]
        for routine in file_data["routines"]:
            if not routine["instructions"]:
                continue
            all_chunks.extend(build_chunks_for_routine(file_name, routine))
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Build embedding-ready chunks from Module 1 parser output.")
    parser.add_argument("inputs", nargs="+", help="One or more parsed JSON files from agc_parser.py")
    parser.add_argument("--out", default="chunks.jsonl", help="Output JSONL path (one chunk per line)")
    args = parser.parse_args()

    parsed_files = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            parsed_files.extend(json.load(f))

    chunks = build_chunks(parsed_files)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    n_code = sum(1 for c in chunks if c["metadata"]["chunk_type"] == "code")
    n_data = sum(1 for c in chunks if c["metadata"]["chunk_type"] == "data")
    n_split = sum(1 for c in chunks if c["metadata"]["n_chunks_in_routine"] > 1)

    print(f"Built {len(chunks)} chunks ({n_code} code, {n_data} data).")
    print(f"{n_split} chunks came from routines that needed splitting (>{MAX_INSTRUCTIONS_PER_CHUNK} instructions).")
    print(f"Output written to {args.out}")


if __name__ == "__main__":
    main()