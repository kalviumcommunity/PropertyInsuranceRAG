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
| **Project Admin & Lead** | Parv | Git workflow, Project Architecture, Python Backend | Vector DB Optimization, Evaluation Frameworks |
| **Document Processing Lead** | Parv | Text extraction, Document Parsing, Cleaning | Chunking strategies for nested insurance tables |
| **RAG Pipeline Engineer** | Parth | Embeddings, Vector indexing, Retrieval logic | Re-ranking and hybrid keyword-semantic search |
| **Evaluation & QA Lead** | Parth | Grounding guardrails, Pytest, Citation validation | Synthetic query generation & Ragas/TruLens metrics |

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

---

## 9. Token Counting, Cost Estimation & Corpus Scaling Analysis

The module [`src/token_counter.py`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/src/token_counter.py) handles tokenization measurement using `tiktoken` (`cl100k_base`), calculates asymmetric input vs. output pricing, projects costs across our 4,000-document policy corpus, and demonstrates the non-linear relationship between text length and token count.

### 9.1 What is a Token?
A token is the atomic unit processed by Large Language Models — typically a subword piece averaging roughly 4 characters or 0.75 words in standard English text. Models do not see characters or full words; they see numerical token identifiers. Because punctuation, morphology, coding syntax, and non-Latin alphabets tokenize differently, token count tracks text length but is **not strictly proportional**.

### 9.2 Running the Token Counter
```bash
# Run the complete analysis report and save output to outputs/token_sample_results.txt
python src/token_counter.py

# Count tokens and characters for custom text
python src/token_counter.py --text "Under policy HO-3, what is the deductible for water backup?"
```

### 9.3 Task 1 — Project Corpus Input & Output Token Counts
For an adjuster query with retrieved context and grounded response:
- **Sample Input (System Prompt + Retrieved Excerpt + Question):** `174 tokens`
  - System Prompt: `47 tokens`
  - Policy Excerpt (HO-3 water backup terms): `89 tokens`
  - Adjuster Question: `32 tokens`
- **Sample Output (Assistant Grounded Response):** `84 tokens`
- **Total Call Tokens:** `258 tokens`

### 9.4 Task 2 — Token Counts Across Samples of Varying Length
| Sample Level | Chars | Words | Tokens | Chars/Token | Tokens/Word |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short (Adjuster Question)** | 67 | 10 | 15 | 4.47 | 1.50 |
| **Medium (Endorsement HO-04-95 Paragraph)** | 494 | 81 | 120 | 4.12 | 1.48 |
| **Long (Full Policy Document: `data/sample_policy.txt`)** | 13,165 | 1,932 | 2,768 | 4.76 | 1.43 |

### 9.5 Task 3 — Cost Estimation (Asymmetric Rates & Corpus Scale)
LLM providers bill input tokens and output tokens separately (output tokens require more compute and autoregressive sampling).

$$\text{Cost} = \left(\frac{\text{Input Tokens}}{1,000} \times \text{Rate}_{\text{input}}\right) + \left(\frac{\text{Output Tokens}}{1,000} \times \text{Rate}_{\text{output}}\right)$$

#### Single RAG Call Cost:
- **Benchmark Rate (\$0.0005/1K input, \$0.0015/1K output):**
  - Input Cost: \$0.000087
  - Output Cost: \$0.000126
  - Total Call Cost: **\$0.000213** (~0.021¢)
- **Production `gpt-4o-mini` (\$0.15/1M input, \$0.60/1M output):** **\$0.000076** (\$0.076 per 1,000 queries)
- **Production `gpt-4o` (\$2.50/1M input, \$10.00/1M output):** **\$0.001275** (\$1.275 per 1,000 queries)

#### Corpus Scale Design (4,000 Policy Documents):
When indexing 4,000 property insurance documents (~2,768 tokens each = ~11.07 million total tokens):
1. **Upfront Embedding Cost:** With `text-embedding-3-small` at \$0.02 / 1M tokens, embedding all 4,000 documents costs only **\$0.2214** (~22 cents).
2. **Runtime Query Cost:** Serving 10,000 adjuster queries per month (with top-5 retrieved chunks ~2,500 input tokens + 300 output tokens) costs just **\$5.55/month** on `gpt-4o-mini`.
3. **Context Limits:** The 128k token window provides ample headroom for multi-chunk retrieval without risk of truncation.

### 9.6 Task 4 — Length-to-Token Non-Linearity Demonstration
| Category | Sample Preview | Chars | Words | Tokens | Chars/Tok | Tok/Word |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple Word** | `refund` | 6 | 1 | 1 | 6.00 | 1.00 |
| **Compound Word** | `refundable` | 10 | 1 | 2 | 5.00 | 2.00 |
| **Short Question** | `What is our refund window?` | 26 | 5 | 6 | 4.33 | 1.20 |
| **Technical Legal Term** | `Noncancellability and indemnification underwriter clause` | 56 | 5 | 10 | 5.60 | 2.00 |
| **German Compound Term** | `Wohngebäudeversicherungsgesellschaft` | 36 | 1 | 11 | 3.27 | 11.00 |
| **Spanish Claim Query** | `¿La póliza cubre daños por agua repentinos...` | 57 | 9 | 19 | 3.00 | 2.11 |
| **Hindi Insurance Term** | `संपत्ति बीमा पॉलिसी दावा निपटान` | 31 | 5 | 32 | 0.97 | 6.40 |
| **Japanese Policy Peril**| `火災水災特約保険証券` | 10 | 1 | 16 | 0.62 | 16.00 |
| **Policy ID / Numbers** | `POL-2024-HO3-998234-EX-01` | 25 | 1 | 14 | 1.79 | 14.00 |
| **JSON Data Payload** | `{"claim_id": "CLM-8812", "amount": 14500.5...}` | 66 | 6 | 28 | 2.36 | 4.67 |
| **Python Code Snippet** | `def calculate_deductible(claim: float, rat...` | 88 | 13 | 26 | 3.38 | 2.00 |

#### Why Tokens Do Not Track Linearly:
1. **Morphological Compounding:** Germanic and agglutinative words split into multiple subwords (`Wohngebäude...` takes 11 tokens for 1 word).
2. **Code and Structured Data:** Punctuation, symbols (`{`, `}`, `_`, `"`), and indentation get split into individual tokens, causing higher token density.
3. **Non-Latin Scripts:** Byte-pair encodings (such as `cl100k_base`) have fewer dedicated vocabulary slots for Devanagari or Kanji, allocating 1–3 tokens per character.

---

## 10. Extracted Text Cleaning

Run the same cleaning pipeline over every `.txt` file under `data/`. The relative
directory structure is preserved in `outputs/cleaned_data/`:

```bash
python -m src.text_cleaner
```

Custom input and output directories can be supplied:

```bash
python -m src.text_cleaner path/to/raw_text path/to/cleaned_text
```

The cleaner removes page markers and navigation lines, detects short lines
repeated at document edges across the corpus, applies Unicode NFKC normalization,
repairs common extraction artifacts, joins hard-wrapped lines, and collapses
excess whitespace. The committed [before sample](examples/raw_policy_excerpt.txt)
and [after sample](examples/cleaned_policy_excerpt.txt) demonstrate the
transformation. Tests run with `pytest`.

## 11. Chunk Provenance Metadata

[`src/chunking.py`](./src/chunking.py) splits UTF-8 text into chunks and keeps each chunk's text alongside a consistent metadata object:

| Field | Meaning |
| :--- | :--- |
| `source` | Document identifier supplied by the caller (or the file name for `chunk_file`). |
| `section` | Most recently detected all-caps or `SECTION`/`COVERAGE` heading; `null` when unavailable. |
| `page` | Most recently detected page marker; `null` when unavailable. |
| `position` | Zero-based chunk index within the document. |
| `start_char`, `end_char` | Zero-based, half-open character offsets into the original document. |

Use `chunk_corpus(Path("data"))` to process a corpus; each source identifier is its path relative to that corpus, avoiding collisions between equal file names in different folders. `chunk_document(text, "policies/homeowners.txt")` can also be used for an individual document. Each `Chunk` object's `to_dict()` representation and JSONL serialization keep text and metadata together. The offsets make the source relationship verifiable: `text[start_char:end_char] == chunk.text`. [`examples/chunk_metadata_sample.jsonl`](./examples/chunk_metadata_sample.jsonl) contains sample text/metadata pairs with a matching source excerpt in [`examples/chunk_metadata_source.txt`](./examples/chunk_metadata_source.txt). Run the focused checks with `pytest tests/test_chunking.py`.

## 12. Full-Corpus Ingestion and Reconciliation

Run `python -m src.ingestion` to load every file under `data/`, clean the text using the shared corpus boilerplate rules, split it into chunks, and attach source/section/page/position/span metadata. The pipeline writes an aggregate, per-document accounting report to `outputs/ingestion_summary.json` and a small set of traceable review chunks to `outputs/ingestion_sample_chunks.jsonl`. Samples exclude chunks matching common declaration labels and long numeric identifiers.

Every non-control file is counted as a source document. Successfully ingested documents plus explicitly recorded failures must equal that count; unsupported file types and read/clean/chunk errors are recorded as failures. The repository `.gitkeep` placeholder is explicitly reported as skipped, not counted as a document. A runtime check raises if the source count does not reconcile. `pytest tests/test_ingestion.py` covers accounting, recorded failures, sample provenance, and output safety.

## 13. Generating and Storing Chunk Embeddings

Set `OPENAI_API_KEY`, `EMBED_MODEL`, and, when needed, `OPENAI_BASE_URL` in
`.env` or the process environment. The embedding script uses the existing
chunk provenance format and writes one JSONL record per chunk with its complete
vector, source text, and source/section/page/position/offset metadata:

```bash
python -m src.embed_chunks
```

By default, it embeds the small prepared corpus in
`examples/chunk_metadata_source.txt` using `CHUNK_SIZE` and writes complete
vectors to the ignored `outputs/embedded_chunks.jsonl`. To choose input/output
paths or save a compact report with vector dimensions and the first five values
per vector:

```powershell
python -m src.embed_chunks --input examples/chunk_metadata_source.txt `
  --output outputs/embedded_chunks.jsonl `
  --sample-output outputs/embedding_sample_output.json
```

The committed [`outputs/embedding_sample_output.json`](outputs/embedding_sample_output.json)
shows the report shape with an explicitly labeled offline test fixture; it is
not a live provider response. Vector previews are intentionally trimmed while
the JSONL store retains complete vectors. Run `pytest tests/test_embed_chunks.py`
to check request construction, response validation, provenance, and output.

---

## 14. Token-Aware Chunking with Controlled Overlap

Character-based chunking cannot guarantee model token limits or semantic preservation at cut boundaries. The module [`src/token_aware_chunker.py`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/src/token_aware_chunker.py) strictly measures and sets chunk size in tokens using `tiktoken` (`cl100k_base`) and implements a sliding window with controlled token overlap.

### 13.1 Running the Token-Aware Chunker & Boundary Demo
```bash
# Run the token-aware chunker with 500-token chunks and 100-token overlap
python src/token_aware_chunker.py

# Run with custom token size and overlap parameters
python src/token_aware_chunker.py --size 400 --overlap 80
```

Artifacts produced:
- **Full Report:** [`outputs/token_overlap_demonstration.txt`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/outputs/token_overlap_demonstration.txt)
- **Serialized Token Chunks (JSON):** [`outputs/token_chunks_sample.json`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/outputs/token_chunks_sample.json)
- **Boundary Context Demonstration (Markdown):** [`examples/boundary_context_overlap_demo.md`](file:///c:/Users/parvj/OneDrive/Desktop/PropertyInsuranceRAG/examples/boundary_context_overlap_demo.md)

### 13.2 Sizing by Tokens & Controlled Overlap (Tasks 1 & 2)
- **Tokenizer:** `tiktoken` with `cl100k_base` encoding (compatible with `gpt-4o-mini` and `text-embedding-3-small`).
- **Sizing Rule:** Chunks are sliced strictly along token array indices, ensuring no chunk exceeds the model's token allocation.
- **Controlled Overlap Formula:**
  $$\text{Step Size} = \text{Chunk Size} - \text{Chunk Overlap} = 500 - 100 = 400 \text{ tokens}$$
  Each adjacent chunk $i$ repeats exactly the trailing $N=100$ tokens of chunk $i-1$ as its leading tokens.

### 13.3 Demonstrating Overlap Preserving Boundary Context (Task 3)
Inspecting the boundary cut at token index 500 on Property Insurance Policy Form HO-3:

#### A. Without Overlap (`overlap = 0 tokens`):
- **Chunk 0 End:** `"...permanently installed fixtures, and heating and cooling machinery attached to"`
- **Chunk 1 Start:** `"the dwelling. 2. We do not cover: a. Land, including land on which the dwelling is located..."`
- **Resulting Fracture:** The phrase `"attached to the dwelling"` is sliced mid-preposition. More importantly, Chunk 0 contains the coverage grant without its exclusions, while Chunk 1 contains exclusions without the parent coverage heading.

#### B. With Controlled Overlap (`overlap = 100 tokens`):
- **Chunk 1 Start:** Starts at token 400, pulling in the entire preceding clause and section header:
  `"...COVERAGE A – DWELLING 1. We cover: a. The dwelling on the 'residence premises'... and heating and cooling machinery attached to the dwelling. 2. We do not cover: a. Land..."`
- **Preserved Meaning:** When an adjuster asks about Coverage A dwelling exclusions, retrieving Chunk 1 provides both the coverage grant and its exclusions together in a single self-contained context.

### 13.4 Setting Justifications for `gpt-4o-mini` (Task 4)
| Parameter | Setting | Architectural Justification |
| :--- | :---: | :--- |
| **Chunk Size** | **500 tokens** | Standard property insurance sections (Coverage A, B, C, D) and endorsement forms (HO-04-95) average 250–450 tokens. Sizing at 500 tokens guarantees an entire policy provision fits within one chunk. Retrieving top-5 chunks consumes ~2,500 tokens (<2% of the 128k context window). |
| **Chunk Overlap** | **100 tokens (20%)** | Complex insurance bullet clauses range from 50 to 80 tokens. A 100-token overlap guarantees that any sentence severed at a boundary is completely reproduced in the adjacent chunk. |
| **Cost Overhead** | **1.22x (+22%)** | Overlap overhead ratio = $\frac{500}{400} = 1.25\text{x}$. Indexing 4,000 documents (~11M tokens) with `text-embedding-3-small` costs **\$0.27** with overlap vs. **\$0.22** without overlap—a 5-cent difference that guarantees claim fidelity. |

### 13.5 Sample Output Comparison (Task 5)
| Configuration | Chunks | Avg Tokens | Min–Max Tokens | Total Emitted Tokens | Overlap Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero Overlap (`size=500, overlap=0`)** | 6 | 461.3 | 268–500 | 2,768 | 1.00x |
| **Controlled Overlap (`size=500, overlap=100`)** | 7 | 481.1 | 368–500 | 3,368 | 1.22x |
