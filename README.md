# Apollo Code Intelligence Platform

An end-to-end system for understanding, documenting, and analyzing the
open-sourced **Apollo 11 Guidance Computer (AGC)** source code —
~145,000 lines of the original flight assembly that flew the Command
Module (Comanche055) and Lunar Module (Luminary099) to the Moon.

Built as a final-year Data Science major project. Combines classic
data engineering, ML, LLM/RAG, knowledge graphs, and MLOps into one
deployable application.

Full project brief: [`docs/project_brief.md`](docs/project_brief.md)

## Status

- [x] **Module 1 — Ingestion & Preprocessing**: parser for `.agc` source files
- [ ] Module 2 — Chunking & Embeddings
- [ ] Module 3 — RAG-powered Code Q&A Assistant
- [ ] Module 4 — Dependency Knowledge Graph
- [ ] Module 5 — ML-based Complexity/Risk Scoring
- [ ] Module 6 — Auto-Documentation Generator
- [ ] Module 7 — Frontend/Dashboard
- [ ] Module 8 — MLOps, CI/CD, Deployment

## Project Structure

```
apollo-code-intelligence-platform/
├── data/
│   ├── sample/          # small sample .agc files for local dev/testing
│   └── parsed/          # sample parser output (JSON)
├── docs/
│   └── project_brief.md
├── src/
│   └── ingestion/
│       └── agc_parser.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the Full Dataset

This repo ships with one sample `.agc` file for development. To work
with the full corpus:

```bash
git clone https://github.com/chrislgarry/Apollo-11.git data/raw_repo
python src/ingestion/agc_parser.py data/raw_repo/Luminary099 --out data/full_parsed/luminary.json
python src/ingestion/agc_parser.py data/raw_repo/Comanche055 --out data/full_parsed/comanche.json
```

(`data/raw_repo/` and `data/full_parsed/` are gitignored — don't commit
the full third-party source tree or its full parse output to this repo.)

## Module 1: Running the Parser

```bash
python src/ingestion/agc_parser.py data/sample/LUNAR_LANDING_GUIDANCE_EQUATIONS.agc --out data/parsed/parsed_sample.json
```

Parses AGC assembly into structured routines: labels, opcodes,
operands, comments, section banners, and cross-references (for the
dependency graph in Module 4).

## License

Apollo Guidance Computer source code: public domain (see
[chrislgarry/Apollo-11](https://github.com/chrislgarry/Apollo-11)).
All original code in this repository: MIT (or your preferred license —
update this section).
