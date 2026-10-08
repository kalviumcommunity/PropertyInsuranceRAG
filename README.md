# Property Insurance RAG Assistant

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Sprint](https://img.shields.io/badge/Sprint-2%20(5%20Weeks)-orange.svg)]()

> An enterprise Retrieval-Augmented Generation (RAG) assistant designed for property insurance adjusters to retrieve accurate, verified, and cited coverage terms from overlapping policy documents, claim guidelines, and underwriting manuals.

---

## 1. Problem Statement

> **"A property insurance provider holds policy documents, claim guidelines, and underwriting manuals, but adjusters frequently misquote coverage terms because the answers are buried across hundreds of overlapping documents."**

### Core Business Context
- **Primary Users:** Insurance claim adjusters and underwriting specialists.
- **Pain Point:** Adjusters handle time-sensitive claims under high call volume. Answering coverage questions requires cross-referencing base policy forms (e.g., HO-3, DP-3), endorsement riders, special perils, claim adjustment guidelines, and state-specific underwriting manuals. Because guidelines overlap and contradict based on effective dates and endorsements, adjusters frequently misquote coverage, resulting in delayed claims, bad faith disputes, or improper payouts.
- **Assistant Mission:** Provide immediate, high-fidelity answers with exact document citations, section references, and clear conflict-resolution explanations between general guidelines and specific policy riders. If a term is not definitively covered, the assistant strictly refuses rather than hallucinating.

---

## 2. Team Charter

### 2.1 Team Details & Roles
| Role | Name | Technical Focus / Strengths | Areas of Growth |
| :--- | :--- | :--- | :--- |
| **Project Admin & Lead** | Parv Jain | Git workflow, Project Architecture, Python Backend | Vector DB Optimization, Evaluation Frameworks |
| **Document Processing Lead** | Team Member 1 | Text extraction, Document Parsing, Cleaning | Chunking strategies for nested insurance tables |
| **RAG Pipeline Engineer** | Team Member 2 | Embeddings, Vector indexing, Retrieval logic | Re-ranking and hybrid keyword-semantic search |
| **Evaluation & QA Lead** | Team Member 3 | Grounding guardrails, Pytest, Citation validation | Synthetic query generation & Ragas/TruLens metrics |

### 2.2 Working Agreements & Team Norms
1. **Daily Standup:** Daily 10-minute sync at 09:30 AM covering:
   - What did you complete in the last 24 hours?
   - What will you complete today?
   - Any blockers?
2. **Branching & PR Rules:**
   - No direct commits to `main`. `main` is protected.
   - Branch naming: `feature/<concept-id>-<description>` (e.g., `feature/3.10-env-setup`).
   - Every PR must address a single concept, include test evidence or runnable scripts, and require at least 1 peer review approval before merging.
3. **Definition of Done (DoD) for PRs:**
   - [ ] Code is isolated and modular within `src/`.
   - [ ] No hardcoded secrets or API keys; configuration read via `src/config.py`.
   - [ ] Unit tests written and passing (`pytest`).
   - [ ] `.gitignore` rules respected (no temporary files or outputs checked in).
   - [ ] PR description includes implementation evidence, screenshots/logs, and aligns with the concept rubric.
4. **Communication & Blocker Protocol:**
   - Active discussions via team Slack/Discord channel.
   - If blocked for > 45 minutes on a technical problem, ping the team for pairing or escalate to the mentor.

---

## 3. End-to-End RAG Architecture

```
Raw Documents (PDF, DOCX, TXT)
  ├── Policy Forms (HO-3, DP-3)
  ├── Endorsement Riders
  ├── Claim Guidelines
  └── Underwriting Manuals
            │
            ▼
┌─────────────────────────────────┐
│ Ingestion & Chunking (src/)     │  -> Semantic splitting, hierarchy preservation
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Embeddings API                  │  -> e.g., text-embedding-3-small
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Vector Database (ChromaDB)      │  -> Metadata filtered by policy type, state, date
└─────────────────────────────────┘
            │
    Question from Adjuster
            │
            ▼
┌─────────────────────────────────┐
│ Top-K Semantic Retrieval        │  -> Top chunks + re-ranking
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Grounded LLM Generation         │  -> Prompt templates in prompts/
│ (with Citations & Refusal Guard)│  -> Explicit policy hierarchy resolution
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Front-End UI (Streamlit/Next.js)│  -> Streaming response + source drawer
└─────────────────────────────────┘
```

---

## 4. Scalable Folder Layout

The project separates concerns cleanly so code, prompts, configurations, and outputs do not clash:

```
PropertyInsuranceRAG/
├── data/                  # Source documents & corpus (git-ignored raw files)
│   └── .gitkeep
├── prompts/               # System and user prompt templates (isolated from code)
│   └── system_prompt.txt
├── src/                   # Core application source code
│   ├── __init__.py
│   └── config.py          # Strongly-typed environment variable loader
├── outputs/               # Logs, generated outputs, evaluations (git-ignored)
│   └── .gitkeep
├── tests/                 # Unit and integration test suite
│   ├── __init__.py
│   └── test_config.py
├── .env.example           # Committed template for required environment variables
├── .gitignore             # Strict ignore rules for .env, .venv, data, outputs
├── requirements.txt       # Frozen application dependencies
└── README.md              # Project documentation and Team Charter
```

---

## 5. Development Environment & Reproducibility Setup

Follow these steps to set up a clean, isolated environment:

### Step 1: Clone the Repository
```bash
git clone https://github.com/kalviumcommunity/PropertyInsuranceRAG.git
cd PropertyInsuranceRAG
```

### Step 2: Create and Activate an Isolated Virtual Environment
```bash
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template and fill in your development keys:
```bash
# On Windows (PowerShell):
Copy-Item .env.example .env

# On macOS / Linux:
cp .env.example .env
```
Open `.env` in your editor and provide your `OPENAI_API_KEY`:
```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-actual-api-key
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```
> **SECURITY NOTICE:** Never commit `.env` to Git. Verify using `git status` that `.env` is ignored.

### Step 5: Run Verification Tests
```bash
pytest
```

---

## 6. Day 1 Alignment & Mastery Check

### Standup Summary
1. **Team Name & Members:** Property Insurance RAG Team (Parv Jain, Project Admin)
2. **Our Problem Statement:** A property insurance provider holds hundreds of overlapping policy documents, claim guidelines, and underwriting manuals, causing adjusters to frequently misquote coverage terms.
3. **The Core Question Our Assistant Must Answer:** *"Under policy [Form X] with endorsement [Rider Y], is [specific damage/loss event] covered for [property type], what deductible applies, and what guideline document authorizes this decision?"*
4. **Our Biggest Open Question About the Corpus:** How are policy hierarchy conflicts (e.g., standard policy exclusions vs. endorsements with broader coverage riders) structured across conflicting document versions and effective dates?
5. **Plan for Tonight's Submission:** Finalize repository configuration, push Team Charter in `README.md`, submit Day 1 PR, and record the Concept 3.10 walkthrough.

### Mastery Check Q&A
- **Question 1:** *In two sentences: what is the core question your assistant must answer, who asks it, and what does a trustworthy answer look like?*
  > **Answer:** Insurance claim adjusters and underwriters ask whether specific property damages or perils are covered under given policy forms and active endorsements. A trustworthy answer must provide an unambiguous coverage determination, explicitly cite the exact policy form, section number, and endorsement rider, and resolve any overlapping guideline conflicts without hallucination.

- **Question 2:** *Show your GitHub repository. Is the team setup complete?*
  > **Answer:**
  > - [x] Repo under `kalviumcommunity/PropertyInsuranceRAG`
  > - [x] All members granted Write access
  > - [x] `main` branch protection configured
  > - [x] `.gitignore` and `.env.example` present and verified
  > - [x] Kanban project board configured with 3 columns (`To Do`, `In Progress`, `Done`)

---

## 7. 5-Week Sprint Roadmap

| Week | Phase | Milestones | Key Concepts |
| :---: | :--- | :--- | :--- |
| **Week 1** | Foundations & Workspace Setup | Environment isolation, Team Charter, Corpus discovery, PRD & Architecture draft | Concepts 1–9, 3.10 |
| **Week 2** | Corpus Ingestion & Vector Indexing | Document parsing (PDFs, tables), semantic chunking, embedding generation, ChromaDB setup | Concepts 10–22 |
| **Week 3** | Retrieval & Grounded Generation | Top-k retrieval, metadata filtering, citations engine, prompt engineering & refusal guardrails | Concepts 23–34 |
| **Week 4** | App UI & Evaluation | Streamlit/Next.js interface, source drawer, end-to-end evaluation & hallucination testing | Concepts 35–40 |
| **Week 5** | Optimization & Delivery | Performance tuning, latency reduction, final demonstration & mentor review | Final Delivery |

---

## 8. LLM Chat Completion & Error Handling (OpenAI-Compatible Client)

The module [`src/llm_client.py`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/src/llm_client.py) powers all interactions with the chat model:

### Features:
1. **Configurable Endpoint:** Adapts seamlessly to OpenAI, Azure OpenAI, Ollama, or LM Studio by passing `OPENAI_BASE_URL` and `CHAT_MODEL`.
2. **Transparent Payload Logging:** Logs outbound message payloads (`REQUEST: ...`), model responses (`RESPONSE: ...`), and token counters (`USAGE: ...` prompt, completion, total).
3. **Robust Error Handling:**
   - **HTTP 401 (`AuthenticationError`):** Intercepted and reported with clear instructions to check `OPENAI_API_KEY` in `.env`.
   - **HTTP 429 (`RateLimitError`):** Intercepted with rate-limit / quota warning advising exponential backoff.
   - **Connection & API Errors:** Catches network timeouts and service-side anomalies cleanly.

### Running the Demo & Simulations:
```bash
# Run simulation for 401 and 429 error scenarios
python src/demo_chat.py --simulate

# Run with an adjuster question (requires valid OPENAI_API_KEY in .env)
python src/demo_chat.py --question "Does policy HO-3 cover sudden water damage from a burst pipe?"
```
