"""
agc_parser.py
--------------
Module 1 of the Apollo Code Intelligence Platform: Ingestion & Preprocessing.

Parses raw Apollo Guidance Computer (.agc) assembly source files (yaYUL format,
as used in the chrislgarry/Apollo-11 repository) into a clean, structured
dataset suitable for downstream embedding, RAG, graph, and ML modules.

Usage:
    python agc_parser.py <path_to_agc_file_or_directory> [--out output.json]

Design notes:
- yaYUL source format: a line either starts with whitespace (no label; first
  token is the opcode) or starts with a label (first token is the label,
  second token is the opcode). Everything after '#' on a line is a comment.
- Full-line comments (lines starting with '#') are collected separately and
  used to (a) parse the file header block and (b) detect section banners
  (rows of '#####' or '*****' style separators) which we use as human-authored
  section boundaries.
- Routines are reconstructed by treating each label that begins a new
  "logical routine" (i.e. is referenced elsewhere, or follows a banner/blank
  line) as a routine start, and collecting all instructions up to the next
  such label.
"""

import re
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class Instruction:
    line_no: int
    label: Optional[str]
    opcode: Optional[str]
    operands: List[str]
    comment: Optional[str]
    raw: str


@dataclass
class Routine:
    name: str
    start_line: int
    end_line: int
    instructions: List[Instruction] = field(default_factory=list)
    banner_comment: Optional[str] = None  # nearest preceding section banner text
    references: List[str] = field(default_factory=list)  # labels this routine calls/branches to


@dataclass
class AGCFile:
    filename: str
    header: dict
    routines: List[Routine]
    raw_line_count: int


# ----------------------------------------------------------------------
# Header parsing
# ----------------------------------------------------------------------

HEADER_FIELDS = [
    "Copyright", "Filename", "Purpose", "Assembler",
    "Contact", "Website", "Pages", "Mod history",
]

def parse_header(lines: List[str]) -> dict:
    """Extract the structured metadata block at the top of every .agc file."""
    header = {}
    current_field = None
    for line in lines:
        if not line.startswith("#"):
            if header:
                break  # header block ended
            continue
        content = line.lstrip("#").strip()
        matched = False
        for field_name in HEADER_FIELDS:
            if content.startswith(field_name + ":"):
                header[field_name] = content[len(field_name) + 1:].strip()
                current_field = field_name
                matched = True
                break
        if not matched and current_field and content:
            # continuation line (e.g. wrapped "Purpose" or "Mod history")
            header[current_field] += " " + content
    return header


# ----------------------------------------------------------------------
# Instruction-level parsing
# ----------------------------------------------------------------------

BANNER_RE = re.compile(r"^#\s*[\*#]{5,}\s*$")          # e.g. "# ********"
SECTION_TITLE_RE = re.compile(r"^#\s+([A-Z0-9 /\-\.,:'\(\)]+)\s*$")  # ALL-CAPS comment line
LABEL_ONLY_ASSIGN_RE = re.compile(r"^\S+\s*=\s*\S+")     # e.g. "RDG = RBRFG"

def split_code_and_comment(line: str):
    """Split a source line into (code_part, comment_part)."""
    if "#" in line:
        idx = line.index("#")
        return line[:idx].rstrip("\n"), line[idx + 1:].strip()
    return line.rstrip("\n"), None


def parse_body(lines: List[str]):
    """
    Parse all non-header lines into Instruction records, and separately
    track full-line comments (banners / section titles) with their line numbers.
    """
    instructions: List[Instruction] = []
    section_markers = {}  # line_no -> section title text
    last_banner_title = None
    in_header = True

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")

        # Skip the leading metadata header block
        if in_header:
            if line.strip().startswith("#") or line.strip() == "":
                continue
            else:
                in_header = False

        if line.strip() == "":
            continue

        # Full-line comment: could be a banner, a section title, or prose
        if line.lstrip().startswith("#"):
            if BANNER_RE.match(line.strip()):
                continue
            m = SECTION_TITLE_RE.match(line.strip())
            if m and len(m.group(1).split()) <= 8:
                last_banner_title = m.group(1).strip()
                section_markers[i] = last_banner_title
            continue

        code_part, comment = split_code_and_comment(line)
        if not code_part.strip():
            continue

        starts_with_space = code_part[0] in (" ", "\t")
        tokens = code_part.split()

        if starts_with_space:
            label = None
            opcode = tokens[0] if tokens else None
            operands = tokens[1:]
        else:
            label = tokens[0] if tokens else None
            opcode = tokens[1] if len(tokens) > 1 else None
            operands = tokens[2:] if len(tokens) > 2 else []

        instructions.append(Instruction(
            line_no=i,
            label=label,
            opcode=opcode,
            operands=operands,
            comment=comment,
            raw=raw_line.rstrip("\n"),
        ))

    return instructions, section_markers


# ----------------------------------------------------------------------
# Routine reconstruction
# ----------------------------------------------------------------------

# Opcodes that represent a branch/call -- used to build cross-references
BRANCH_OPCODES = {
    "TC", "TCF", "BZF", "BZMF", "BMN", "BPL", "CCS",
    "GOTO", "CALL", "POSTJUMP", "CADR",
}

def looks_like_label_ref(token: str) -> bool:
    """Heuristic: alphabetic-led token, not a pure number/octal constant."""
    return bool(re.match(r"^[A-Za-z\?/][A-Za-z0-9\?/\-\+]*$", token)) and not token.isdigit()


def build_routines(instructions: List[Instruction], section_markers: dict) -> List[Routine]:
    routines: List[Routine] = []
    current: Optional[Routine] = None
    sorted_markers = sorted(section_markers.items())

    def nearest_banner(line_no):
        title = None
        for ln, t in sorted_markers:
            if ln <= line_no:
                title = t
            else:
                break
        return title

    for instr in instructions:
        if instr.label is not None:
            # start a new routine at every labeled instruction
            if current:
                current.end_line = instr.line_no - 1
                routines.append(current)
            current = Routine(
                name=instr.label,
                start_line=instr.line_no,
                end_line=instr.line_no,
                banner_comment=nearest_banner(instr.line_no),
            )
        if current is None:
            # instructions before the first label -> synthetic "PREAMBLE" routine
            current = Routine(
                name="PREAMBLE",
                start_line=instr.line_no,
                end_line=instr.line_no,
                banner_comment=nearest_banner(instr.line_no),
            )
        current.instructions.append(instr)
        current.end_line = instr.line_no

        if instr.opcode in BRANCH_OPCODES:
            for op in instr.operands:
                clean = op.strip()
                if looks_like_label_ref(clean):
                    current.references.append(clean)

    if current:
        routines.append(current)

    return routines


# ----------------------------------------------------------------------
# Top-level file parsing
# ----------------------------------------------------------------------

def parse_agc_file(path: str) -> AGCFile:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header = parse_header(lines)
    instructions, section_markers = parse_body(lines)
    routines = build_routines(instructions, section_markers)

    return AGCFile(
        filename=os.path.basename(path),
        header=header,
        routines=routines,
        raw_line_count=len(lines),
    )


def agc_file_to_dict(agc_file: AGCFile) -> dict:
    d = asdict(agc_file)
    return d


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse Apollo Guidance Computer .agc source files.")
    parser.add_argument("path", help="Path to a .agc file or a directory of .agc files")
    parser.add_argument("--out", default="parsed_output.json", help="Output JSON path")
    args = parser.parse_args()

    results = []
    if os.path.isdir(args.path):
        for fname in sorted(os.listdir(args.path)):
            if fname.endswith(".agc"):
                fpath = os.path.join(args.path, fname)
                try:
                    results.append(agc_file_to_dict(parse_agc_file(fpath)))
                except Exception as e:
                    print(f"[WARN] Failed to parse {fname}: {e}")
    else:
        results.append(agc_file_to_dict(parse_agc_file(args.path)))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_routines = sum(len(r["routines"]) for r in results)
    total_instructions = sum(len(rt["instructions"]) for r in results for rt in r["routines"])
    print(f"Parsed {len(results)} file(s): {total_routines} routines, {total_instructions} instructions.")
    print(f"Output written to {args.out}")


if __name__ == "__main__":
    main()