# Apollo Code Intelligence Platform
### A Final-Year Major Project Brief (B.E./B.Tech Data Science)

---

## 1. The Hook

In 2016, NASA/MIT open-sourced the original **Apollo Guidance Computer (AGC)** source code — ~145,000 lines of hand-annotated assembly that flew Apollo 11 to the Moon (public repo: `chrislgarry/Apollo-11` on GitHub, mirroring the MIT Museum's `virtualagc` archive). It's full of legendary engineer comments (`# BURN BABY BURN`, `# TEMPORARY, I HOPE HOPE HOPE`) and represents one of the most historically significant, safety-critical codebases ever written — with almost no modern documentation.

That's your dataset. Instead of just admiring it, you're going to build an **AI system that understands, documents, analyzes, and answers questions about it** — the same category of tool real companies build internally for legacy-code modernization (banks, aerospace, telecom all sit on decades-old codebases).

---

## 2. Problem Statement

Large legacy codebases (assembly, COBOL, old C) are expensive to understand, maintain, and onboard new engineers onto. There's no scalable way to:
- Ask natural-language questions about what a piece of legacy code does
- Automatically generate human-readable documentation
- Identify the most complex / riskiest / most-central modules
- Visualize how modules and subroutines depend on each other

**Goal:** Build an end-to-end Code Intelligence platform — using the Apollo 11 AGC source as the demonstration corpus — that ingests raw legacy code, understands it semantically, and exposes that understanding through a RAG-powered assistant, a dependency knowledge graph, and an ML-driven complexity/risk analysis, all wrapped in a deployed, production-style application.

---

## 3. Why This Has Strong Interview Merit

- It's not another "Titanic dataset" or "iris classifier" — the dataset itself is a great story, and interviewers will remember it.
- It touches **every** current hiring keyword: LLMs, RAG, embeddings, vector DBs, knowledge graphs, MLOps, CI/CD, cloud deployment — without being a shallow "I called the OpenAI API once" project.
- It's genuinely **end-to-end**: data engineering → ML/NLP → GenAI → backend → frontend → deployment → monitoring.
- It maps directly onto a real enterprise use case (legacy code modernization / internal developer tools), which is exactly what product companies pay data scientists and ML engineers to build.
- It gives you natural, meaty answers to "walk me through a project you built" — architecture decisions, trade-offs, evaluation metrics, and failure cases all exist here.

---

## 4. System Architecture (8 Modules)

```
[GitHub AGC Repo] 
      │
      ▼
[1. Ingestion & Parsing] → [2. Chunking + Embeddings] → [Vector DB]
      │                                                       │
      ▼                                                       ▼
[5. Complexity/Risk ML Model]                     [3. RAG Q&A Assistant] ← LLM
      │                                                       │
      ▼                                                       ▼
[4. Dependency Knowledge Graph] ──────────────► [7. Dashboard / Chat UI]
      │                                                       │
      ▼                                                       ▼
[6. Auto-Documentation Generator] ─────────► [8. Dockerized, CI/CD Deployed App]
```

---

## 5. Module-by-Module Breakdown

### Module 1 — Ingestion & Preprocessing
- Clone/parse the AGC repo (`.agc` assembly files + comments + block structure).
- Write a custom lightweight parser (regex/AST-like) to extract: routine names, comments, opcodes, line numbers, cross-references (`TC`, `CAF`, `TCF` calls, etc.).
- Structure everything into a clean dataset (JSON/Parquet): one record per subroutine/module.
- **Skills shown:** data engineering, parsing, working with messy unstructured real-world text.

### Module 2 — Chunking & Embeddings
- Design a chunking strategy suited to assembly code (per-subroutine, not naive fixed-length).
- Generate embeddings using an open-source code embedding model (e.g., `CodeBERT`, `bge-code`, or `sentence-transformers`) — avoid being 100% dependent on a paid API.
- Store vectors in a vector database: **Chroma** or **Qdrant** (both free/local, resume-friendly, and used in real companies) or Pinecone if you want cloud-managed.
- **Skills shown:** embeddings, vector search, representation learning.

### Module 3 — RAG-Powered Code Q&A Assistant
- Build a retrieval-augmented generation pipeline using **LangChain** or **LlamaIndex** on top of your vector store.
- User asks: *"What does the P63 landing routine do?"* → system retrieves relevant chunks → LLM answers with **cited line numbers**.
- Add guardrails: if retrieval confidence is low, say "not found" instead of hallucinating.
- **Skills shown:** GenAI application development, prompt engineering, hallucination mitigation — the single most in-demand skill set right now.

### Module 4 — Dependency Knowledge Graph
- From Module 1's parsed cross-references, build a graph of subroutine calls/dependencies.
- Store and query it in **Neo4j** (free tier / local Docker).
- Visualize: "which modules are most central / most called" using graph centrality metrics (PageRank, betweenness).
- **Skills shown:** graph databases, graph analytics — a differentiator most student projects don't have.

### Module 5 — ML-Based Code Complexity & Risk Scoring
- Engineer features per module: lines of code, comment density, cyclomatic-complexity proxy for assembly, number of jumps/branches, call-graph centrality (from Module 4).
- Apply unsupervised clustering (KMeans/HDBSCAN) to group modules by "risk profile," and/or train a regression/classification model to predict a "complexity score."
- Correlate ML-flagged high-risk modules against modules historically known to be critical (e.g., the real 1202/1201 program alarms during the actual Apollo 11 landing — great narrative hook for your demo).
- **Skills shown:** classic ML, feature engineering, unsupervised learning, model evaluation.

### Module 6 — Auto-Documentation Generator
- Use the LLM to generate plain-English docstrings/summaries for each subroutine, batch-processed and cached (not generated live) to control cost.
- Evaluate quality using **RAGAS** or a custom rubric (faithfulness, relevance) against a small hand-labeled sample.
- **Skills shown:** applied GenAI evaluation, not just generation — interviewers love this because most students skip evaluation entirely.

### Module 7 — Interactive Dashboard / Frontend
- **React** (or Streamlit if you want to move faster) frontend with:
  - Chat interface for the RAG assistant (with citations)
  - Interactive dependency graph visualization (e.g., `react-force-graph` or `pyvis`)
  - Complexity heatmap of the codebase
- **FastAPI** backend serving all the above via REST endpoints.
- **Skills shown:** full-stack capability, API design.

### Module 8 — MLOps, CI/CD & Deployment
- Containerize everything with **Docker** / `docker-compose` (vector DB + Neo4j + FastAPI + frontend).
- Set up **GitHub Actions** CI/CD: lint, test, build, deploy on push.
- Deploy on a free-tier cloud (Render/Fly.io/AWS free tier/GCP Cloud Run).
- Add basic monitoring/logging (latency of RAG responses, retrieval hit-rate over time).
- **Skills shown:** MLOps, DevOps — the thing that separates "notebook projects" from "product-ready engineers," and exactly what product companies screen for.

---

## 6. Tech Stack Summary

| Layer | Tools |
|---|---|
| Data/Parsing | Python, regex/custom parser, Pandas |
| Embeddings | CodeBERT / bge / sentence-transformers |
| Vector DB | Chroma or Qdrant |
| Knowledge Graph | Neo4j + Cypher |
| GenAI/RAG | LangChain or LlamaIndex + an LLM (open-source via Ollama, or API-based) |
| ML | scikit-learn (clustering, regression) |
| Evaluation | RAGAS / custom eval harness |
| Backend | FastAPI |
| Frontend | React (or Streamlit for speed) |
| Infra | Docker, Docker Compose, GitHub Actions |
| Deployment | Render / Fly.io / AWS / GCP free tier |
| Monitoring | Basic logging + a simple metrics dashboard (Grafana optional) |

---

## 7. Data Source

- Primary: `chrislgarry/Apollo-11` GitHub repository (Apollo 11 Guidance Computer source, Command Module + Lunar Module code, "Comanche055" and "Luminary099").
- Optional enrichment: NASA's public historical documentation on the 1202/1201 program alarms for a compelling real-world validation story in Module 5.

---

## 8. Evaluation Metrics (have these ready to present)

- **Retrieval:** Precision@k / Recall@k on a hand-built set of ~30 Q&A pairs about the codebase.
- **Generation quality:** RAGAS faithfulness & answer-relevance scores.
- **Complexity model:** Silhouette score for clustering; correlation between predicted risk and known-critical modules.
- **System:** end-to-end query latency, uptime, cost per query.

---

## 9. Suggested Timeline (~16–18 weeks)

1. **Weeks 1–2:** Repo research, parser design, dataset construction.
2. **Weeks 3–4:** Embeddings + vector DB + baseline retrieval.
3. **Weeks 5–6:** RAG assistant v1 + prompt engineering.
4. **Weeks 7–8:** Knowledge graph construction + graph analytics.
5. **Weeks 9–10:** Complexity/risk ML model + feature engineering.
6. **Weeks 11–12:** Auto-documentation module + evaluation harness.
7. **Weeks 13–14:** Frontend + FastAPI backend integration.
8. **Weeks 15–16:** Dockerize, CI/CD, deploy, monitoring.
9. **Weeks 17–18:** Buffer, polish, write report/paper, demo video.

---

## 10. Stretch Goals (if you want to go further)

- Fine-tune a small open-source LLM (e.g., via LoRA) on assembly-code Q&A pairs you generate synthetically.
- Add a multi-agent setup (one agent retrieves, one critiques/verifies the answer against source before responding).
- Build a "diff explainer": given two versions of a routine, explain what changed in plain English.
- Add voice interface as a nod to "Houston, we have a solution" (fun demo factor).

---

## 11. How to Position This on Your Resume

> **Apollo Code Intelligence Platform** — Built an end-to-end RAG-based system over the open-sourced Apollo 11 Guidance Computer codebase (145K+ lines of assembly), combining vector search, a Neo4j dependency knowledge graph, and an ML-driven code-risk model. Achieved [X]% retrieval precision and [Y] faithfulness score (RAGAS); deployed as a containerized full-stack app (FastAPI + React) with CI/CD via GitHub Actions.

Interview talking points this unlocks:
- "Tell me about a time you dealt with ambiguous/unstructured data" → parsing raw assembly.
- "How do you evaluate an LLM system?" → your RAGAS/retrieval metrics.
- "How would you reduce hallucination?" → your citation + confidence-threshold design.
- "Tell me about a system you deployed end-to-end" → your Docker/CI-CD pipeline.

---

## 12. Alternative / Companion Angle (Plan B or Bonus Module)

If you want more classic time-series/DS flavor alongside the GenAI work: simulate Apollo-style **mission telemetry** (based on publicly documented AGC parameters) and build a **real-time anomaly detection pipeline** (streaming via Kafka, LSTM/Transformer-based forecasting, alerting dashboard). This can be bolted on as an extra module if you want an even heavier "classic ML/streaming" component for interviews at data-heavy companies.
