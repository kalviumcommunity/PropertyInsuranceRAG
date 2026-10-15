"""
Token Counter and Cost Estimation Module for Property Insurance RAG.

Provides utilities for:
- Tokenizing text using tiktoken (cl100k_base and model-specific encodings).
- Counting tokens for raw text strings and structured chat messages.
- Estimating call costs based on asymmetric input/output pricing rates.
- Analyzing corpus-level costs (e.g. 4,000 documents corpus ingestion & retrieval).
- Demonstrating the non-linear relationship between text length (chars/words) and tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tiktoken

from src.config import config

# Default fallback encoding used by OpenAI GPT-3.5 / GPT-4 / embeddings
DEFAULT_ENCODING = "cl100k_base"

# Cache tokenizers to prevent re-initialization overhead
_ENCODING_CACHE: Dict[str, tiktoken.Encoding] = {}


def get_tokenizer(encoding_name_or_model: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """
    Retrieves or loads a cached tiktoken Encoding instance.

    Args:
        encoding_name_or_model: Encoding name (e.g., 'cl100k_base', 'o200k_base')
                                 or model name (e.g., 'gpt-4o', 'gpt-4o-mini').

    Returns:
        tiktoken.Encoding: The tokenizer encoding object.
    """
    if encoding_name_or_model in _ENCODING_CACHE:
        return _ENCODING_CACHE[encoding_name_or_model]

    try:
        encoding = tiktoken.get_encoding(encoding_name_or_model)
    except ValueError:
        try:
            encoding = tiktoken.encoding_for_model(encoding_name_or_model)
        except KeyError:
            # Fallback to cl100k_base if model is unrecognized
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)

    _ENCODING_CACHE[encoding_name_or_model] = encoding
    return encoding


def count_tokens(text: Optional[str], encoding_name: str = DEFAULT_ENCODING) -> int:
    """
    Counts the number of tokens in a given text string.

    Args:
        text: Input string to tokenize.
        encoding_name: Name of encoding or model.

    Returns:
        int: Number of tokens (0 if text is empty or None).
    """
    if not text:
        return 0
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text))


def count_messages_tokens(
    messages: List[Dict[str, str]],
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """
    Counts tokens for a list of chat completion messages, accounting for
    chat format delimiters (<|im_start|>{role}\n{content}<|im_end|>).

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        encoding_name: Tokenizer encoding name.

    Returns:
        int: Total token count including protocol overhead.
    """
    enc = get_tokenizer(encoding_name)
    num_tokens = 0
    for message in messages:
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens += 3
        for key, value in message.items():
            num_tokens += len(enc.encode(str(value)))
    # Every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3
    return num_tokens


@dataclass
class CostEstimate:
    """Detailed breakdown of token counts and calculated financial costs."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_rate_per_1k: float
    output_rate_per_1k: float
    input_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_rate_per_1k: float = 0.0005,
    output_rate_per_1k: float = 0.0015,
) -> CostEstimate:
    """
    Calculates the financial cost of an LLM query from prompt and completion token counts.
    Input and output tokens are billed at different rates.

    Args:
        prompt_tokens: Total tokens in prompt/input.
        completion_tokens: Total tokens in generation/output.
        input_rate_per_1k: Dollar cost per 1,000 input tokens (default: $0.0005/1K).
        output_rate_per_1k: Dollar cost per 1,000 output tokens (default: $0.0015/1K).

    Returns:
        CostEstimate: Dataclass containing token totals and breakdown of costs.
    """
    input_cost = (prompt_tokens / 1000.0) * input_rate_per_1k
    output_cost = (completion_tokens / 1000.0) * output_rate_per_1k
    total_cost = input_cost + output_cost

    return CostEstimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_rate_per_1k=input_rate_per_1k,
        output_rate_per_1k=output_rate_per_1k,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round(total_cost, 6),
    )


@dataclass
class CorpusEstimate:
    """Cost and sizing estimate for embedding and serving an entire document corpus."""
    num_documents: int
    avg_tokens_per_doc: int
    total_corpus_tokens: int
    embed_rate_per_1m: float
    one_time_embedding_cost: float
    avg_query_input_tokens: int
    avg_query_output_tokens: int
    cost_per_query: float
    cost_per_1000_queries: float


def estimate_corpus_scale(
    num_documents: int = 4000,
    avg_tokens_per_doc: int = 1800,
    embed_rate_per_1m: float = 0.02,  # text-embedding-3-small ($0.02 / 1M tokens)
    avg_query_input_tokens: int = 2500,  # System prompt + top-k retrieved chunks + question
    avg_query_output_tokens: int = 300,  # Grounded answer with citations
    query_input_rate_per_1k: float = 0.00015,  # gpt-4o-mini ($0.15 / 1M)
    query_output_rate_per_1k: float = 0.00060,  # gpt-4o-mini ($0.60 / 1M)
) -> CorpusEstimate:
    """
    Calculates upfront embedding costs and runtime query costs for a multi-document corpus.

    Args:
        num_documents: Total number of documents in corpus (e.g., 4,000 policy documents).
        avg_tokens_per_doc: Average tokens per document.
        embed_rate_per_1m: Embedding cost per 1M tokens ($0.02 for text-embedding-3-small).
        avg_query_input_tokens: Input tokens per search query (prompt + context).
        avg_query_output_tokens: Output tokens per query response.
        query_input_rate_per_1k: LLM input cost per 1k tokens.
        query_output_rate_per_1k: LLM output cost per 1k tokens.

    Returns:
        CorpusEstimate: Upfront corpus embedding and query runtime costs.
    """
    total_corpus_tokens = num_documents * avg_tokens_per_doc
    one_time_embedding_cost = (total_corpus_tokens / 1_000_000.0) * embed_rate_per_1m

    query_estimate = estimate_cost(
        prompt_tokens=avg_query_input_tokens,
        completion_tokens=avg_query_output_tokens,
        input_rate_per_1k=query_input_rate_per_1k,
        output_rate_per_1k=query_output_rate_per_1k,
    )
    cost_per_query = query_estimate.total_cost
    cost_per_1000_queries = cost_per_query * 1000.0

    return CorpusEstimate(
        num_documents=num_documents,
        avg_tokens_per_doc=avg_tokens_per_doc,
        total_corpus_tokens=total_corpus_tokens,
        embed_rate_per_1m=embed_rate_per_1m,
        one_time_embedding_cost=round(one_time_embedding_cost, 4),
        avg_query_input_tokens=avg_query_input_tokens,
        avg_query_output_tokens=avg_query_output_tokens,
        cost_per_query=round(cost_per_query, 6),
        cost_per_1000_queries=round(cost_per_1000_queries, 4),
    )


@dataclass
class LengthTokenMetric:
    """Metrics comparing character, word, and token counts for a sample text."""
    category: str
    sample_preview: str
    char_count: int
    word_count: int
    token_count: int
    chars_per_token: float
    tokens_per_word: float


def analyze_length_to_token_relationship(
    samples: Dict[str, str],
    encoding_name: str = DEFAULT_ENCODING,
) -> List[LengthTokenMetric]:
    """
    Compares characters, words, and tokens across distinct text categories to demonstrate
    why token counts track text length but are non-linear.

    Args:
        samples: Dictionary mapping sample category name to text string.
        encoding_name: Tokenizer encoding name.

    Returns:
        List[LengthTokenMetric]: Metrics for each sample.
    """
    metrics: List[LengthTokenMetric] = []
    enc = get_tokenizer(encoding_name)

    for category, text in samples.items():
        tokens = len(enc.encode(text))
        chars = len(text)
        words = len(text.split())

        chars_per_tok = round(chars / tokens, 2) if tokens > 0 else 0.0
        tok_per_word = round(tokens / words, 2) if words > 0 else 0.0
        preview = text.strip().replace("\n", " ")
        if len(preview) > 45:
            preview = preview[:42] + "..."

        metrics.append(
            LengthTokenMetric(
                category=category,
                sample_preview=preview,
                char_count=chars,
                word_count=words,
                token_count=tokens,
                chars_per_token=chars_per_tok,
                tokens_per_word=tok_per_word,
            )
        )
    return metrics


# Convenient alias
analyze_length_to_token_ratio = analyze_length_to_token_relationship


def run_token_analysis() -> str:
    """
    Executes all tasks:
    1. Count tokens for project corpus sample inputs and outputs.
    2. Report token counts for three samples of varying length (short, paragraph, full document).
    3. Calculate cost estimates (per-call rates and corpus scale).
    4. Demonstrate length-to-token non-linearity across languages, code, and terms.
    5. Formats the full report.

    Returns:
        str: Human-readable formatted analysis report.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("PROPERTY INSURANCE RAG — TOKEN COUNTING & COST ESTIMATION REPORT")
    lines.append("=" * 80)

    # -------------------------------------------------------------------------
    # Task 1: Count tokens for sample input and output
    # -------------------------------------------------------------------------
    lines.append("\n[TASK 1] Count Tokens for Project Corpus Input & Output")
    lines.append("-" * 80)

    sample_system_prompt = (
        "You are a specialized Property Insurance Adjuster Assistant.\n"
        "Ground answers strictly in provided policy documents. Cite section numbers.\n"
        "If coverage cannot be verified, state: 'The provided policy corpus does not "
        "contain sufficient information to verify this coverage term.'"
    )
    sample_context = (
        "Under Section I – Perils Insured Against, Coverage A (Dwelling) covers accidental "
        "discharge or overflow of water from within a plumbing system. However, Section I "
        "Exclusion 1.b excludes water backup through sewers or drains unless Endorsement "
        "HO-04-95 is present. Endorsement HO-04-95 adds up to $10,000 for sewer/drain "
        "backup subject to a $1,000 special deductible."
    )
    sample_question = (
        "Does policy HO-3 cover water damage if the basement drain backs up after heavy rain, "
        "and what deductible applies if endorsement HO-04-95 is active?"
    )
    sample_prompt = f"System: {sample_system_prompt}\nContext: {sample_context}\nQuestion: {sample_question}"

    sample_output = (
        "Based on Endorsement HO-04-95 (Water Back-Up and Sump Discharge or Overflow), "
        "water backup through sewers or drains is covered up to a limit of $10,000, "
        "overriding the standard Section I Exclusion 1.b.\n\n"
        "A separate special deductible of $1,000 applies to this loss under Endorsement HO-04-95, "
        "Section 2 (Special Deductible)."
    )

    in_tokens = count_tokens(sample_prompt)
    out_tokens = count_tokens(sample_output)
    total_tokens = in_tokens + out_tokens

    lines.append(f"• Sample Input Prompt Tokens:     {in_tokens} tokens")
    lines.append(f"  - System Prompt:                {count_tokens(sample_system_prompt)} tokens")
    lines.append(f"  - Retrieved Context Excerpt:    {count_tokens(sample_context)} tokens")
    lines.append(f"  - Adjuster Question:            {count_tokens(sample_question)} tokens")
    lines.append(f"• Sample Output (LLM Answer):     {out_tokens} tokens")
    lines.append(f"• Total Call Tokens:              {total_tokens} tokens")

    # -------------------------------------------------------------------------
    # Task 2: Report counts for three samples of varying length
    # -------------------------------------------------------------------------
    lines.append("\n[TASK 2] Report Counts for Three Text Samples of Varying Length")
    lines.append("-" * 80)

    # Sample A: Short Question
    sample_short = "What is the deductible for water backup under endorsement HO-04-95?"

    # Sample B: Medium Paragraph (Endorsement excerpt)
    sample_paragraph = (
        "SPECIAL ENDORSEMENT: WATER BACK-UP AND SUMP DISCHARGE OR OVERFLOW (HO-04-95)\n"
        "For an additional premium, Section I – Exclusion 1.b. is deleted and replaced with "
        "the following: We will pay up to $10,000 for direct physical loss, not caused by the "
        "negligence of an insured, to property covered under Section I caused by water or "
        "water-borne material which backs up through sewers or drains, or which overflows from "
        "a sump pump. A separate special deductible of $1,000 applies to each covered loss."
    )

    # Sample C: Full Policy Document
    policy_path = config.DATA_DIR / "sample_policy.txt"
    if not policy_path.exists():
        policy_path = config.DATA_DIR / "policy.txt"

    if policy_path.exists():
        sample_document = policy_path.read_text(encoding="utf-8")
        doc_source = str(policy_path.name)
    else:
        sample_document = sample_paragraph * 15
        doc_source = "In-memory Synthetic Policy Document"

    three_samples = {
        "1. Short (Adjuster Question)": sample_short,
        "2. Medium (Policy Endorsement Paragraph)": sample_paragraph,
        f"3. Long (Full Policy Document: {doc_source})": sample_document,
    }

    lines.append(
        f"{'Sample Level':<35} | {'Chars':>7} | {'Words':>7} | {'Tokens':>7} | {'Chars/Tok':>9} | {'Tok/Word':>9}"
    )
    lines.append("-" * 84)

    for name, text in three_samples.items():
        c_count = len(text)
        w_count = len(text.split())
        t_count = count_tokens(text)
        c_per_t = round(c_count / t_count, 2) if t_count > 0 else 0.0
        t_per_w = round(t_count / w_count, 2) if w_count > 0 else 0.0
        lines.append(
            f"{name:<35} | {c_count:>7} | {w_count:>7} | {t_count:>7} | {c_per_t:>9.2f} | {t_per_w:>9.2f}"
        )

    # -------------------------------------------------------------------------
    # Task 3: Cost estimation (Asymmetric Input vs Output Rates)
    # -------------------------------------------------------------------------
    lines.append("\n[TASK 3] Cost Estimation from Token Counts")
    lines.append("-" * 80)

    # Rate 1: Standard benchmark rate from lesson prompt ($0.0005/1K in, $0.0015/1K out)
    cost_prompt_rate = estimate_cost(
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
        input_rate_per_1k=0.0005,
        output_rate_per_1k=0.0015,
    )

    # Rate 2: Production GPT-4o-mini rate ($0.150/1M in = $0.00015/1K, $0.600/1M out = $0.00060/1K)
    cost_gpt4o_mini = estimate_cost(
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
        input_rate_per_1k=0.00015,
        output_rate_per_1k=0.00060,
    )

    # Rate 3: Production GPT-4o rate ($2.50/1M in = $0.00250/1K, $10.00/1M out = $0.01000/1K)
    cost_gpt4o = estimate_cost(
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
        input_rate_per_1k=0.00250,
        output_rate_per_1k=0.01000,
    )

    lines.append(f"A. Single RAG Inference Call (Input={in_tokens} tokens, Output={out_tokens} tokens):")
    lines.append(
        f"   • Lesson Benchmark ($0.0005/1K in, $0.0015/1K out):\n"
        f"     Input Cost:  ${cost_prompt_rate.input_cost:.6f}\n"
        f"     Output Cost: ${cost_prompt_rate.output_cost:.6f}\n"
        f"     Total Cost:  ${cost_prompt_rate.total_cost:.6f} (~${cost_prompt_rate.total_cost:.4f})"
    )
    lines.append(
        f"   • OpenAI gpt-4o-mini ($0.15/1M in, $0.60/1M out):\n"
        f"     Total Cost:  ${cost_gpt4o_mini.total_cost:.6f} (${cost_gpt4o_mini.total_cost * 1000:.3f} per 1,000 queries)"
    )
    lines.append(
        f"   • OpenAI gpt-4o ($2.50/1M in, $10.00/1M out):\n"
        f"     Total Cost:  ${cost_gpt4o.total_cost:.6f} (${cost_gpt4o.total_cost * 1000:.3f} per 1,000 queries)"
    )

    # Corpus Scale Analysis (4,000 Documents)
    doc_tokens = count_tokens(sample_document)
    corpus = estimate_corpus_scale(
        num_documents=4000,
        avg_tokens_per_doc=doc_tokens if doc_tokens > 0 else 1800,
        embed_rate_per_1m=0.02,
    )

    lines.append("\nB. Corpus-Scale Design & Cost Projection (4,000 Property Insurance Documents):")
    lines.append(f"   • Document Corpus Size:            4,000 documents")
    lines.append(f"   • Average Tokens per Document:     {corpus.avg_tokens_per_doc:,} tokens")
    lines.append(f"   • Total Corpus Tokens:             {corpus.total_corpus_tokens:,} tokens (~{corpus.total_corpus_tokens/1e6:.2f}M tokens)")
    lines.append(f"   • Upfront Embedding Cost (Chroma): ${corpus.one_time_embedding_cost:.4f} (at $0.02 / 1M tokens with text-embedding-3-small)")
    lines.append(f"   • Runtime Query Cost (gpt-4o-mini):${corpus.cost_per_query:.6f} per adjuster query")
    lines.append(f"   • 10,000 Monthly Adjuster Queries: ${corpus.cost_per_query * 10000:.2f} / month")
    lines.append(
        f"   • Context Window Headroom:         Standard 128k context window easily holds top-5 chunks (~2,500 tokens),\n"
        f"                                      preventing truncation while keeping query costs under fractions of a cent."
    )

    # -------------------------------------------------------------------------
    # Task 4: Show the length–token relationship (Non-Linearity Demonstration)
    # -------------------------------------------------------------------------
    lines.append("\n[TASK 4] Demonstrate Text Length vs. Token Count Relationship")
    lines.append("-" * 80)
    lines.append(
        "Demonstrates why character or word counts do not map 1:1 with tokens.\n"
        "Factors like morphology, whitespace, programming syntax, and non-Latin alphabets\n"
        "alter tokenization density significantly:"
    )

    non_linear_samples = {
        "Simple Word": "refund",
        "Compound Word": "refundable",
        "Short Question": "What is our refund window?",
        "Technical Legal Term": "Noncancellability and indemnification underwriter clause",
        "German Compound Term": "Wohngebäudeversicherungsgesellschaft",
        "Spanish Claim Query": "¿La póliza cubre daños por agua repentinos e imprevistos?",
        "Hindi Insurance Term": "संपत्ति बीमा पॉलिसी दावा निपटान",
        "Japanese Policy Peril": "火災水災特約保険証券",
        "Policy ID / Numbers": "POL-2024-HO3-998234-EX-01",
        "JSON Data Payload": '{"claim_id": "CLM-8812", "amount": 14500.50, "status": "APPROVED"}',
        "Python Code Snippet": "def calculate_deductible(claim: float, rate: float = 0.02) -> float: return claim * rate",
    }

    metrics = analyze_length_to_token_ratio(non_linear_samples)

    lines.append(
        f"\n{'Category':<24} | {'Preview':<25} | {'Chars':>6} | {'Words':>5} | {'Tokens':>6} | {'Chars/Tok':>9} | {'Tok/Word':>8}"
    )
    lines.append("-" * 92)

    for m in metrics:
        lines.append(
            f"{m.category:<24} | {m.sample_preview:<25} | {m.char_count:>6} | {m.word_count:>5} | "
            f"{m.token_count:>6} | {m.chars_per_token:>9.2f} | {m.tokens_per_word:>8.2f}"
        )

    lines.append("\nKey Takeaways:")
    lines.append(" 1. English prose averages ~4 characters per token (~1.2 - 1.3 tokens per word).")
    lines.append(" 2. Long compound words and legal jargon (e.g. 'Wohngebäudeversicherungsgesellschaft')")
    lines.append("    split into multiple subwords, causing high tokens-per-word ratios.")
    lines.append(" 3. Code, JSON syntax, and alphanumeric policy IDs split into frequent single-character")
    lines.append("    or punctuation tokens, increasing token density.")
    lines.append(" 4. Non-Latin scripts (Hindi, Japanese) require multiple tokens per character/word because")
    lines.append("    standard byte-pair encodings (cl100k_base) allocate fewer dedicated vocab tokens to them.")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    return report_text


def save_sample_results(output_path: Optional[Path] = None) -> Path:
    """
    Runs the token analysis and persists the formatted results to disk.

    Args:
        output_path: Target path (defaults to outputs/token_sample_results.txt).

    Returns:
        Path: The absolute path of the generated results file.
    """
    target = output_path or (config.OUTPUTS_DIR / "token_sample_results.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    report = run_token_analysis()
    target.write_text(report, encoding="utf-8")
    return target


def main() -> None:
    """CLI entrypoint for running the token counter analysis."""
    parser = argparse.ArgumentParser(
        description="Count tokens and estimate costs for Property Insurance RAG."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to outputs/token_sample_results.txt",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional custom text string to count tokens for.",
    )
    args = parser.parse_args()

    # Safely reconfigure standard output encoding for Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.text:
        count = count_tokens(args.text)
        chars = len(args.text)
        words = len(args.text.split())
        print(f"Text: '{args.text}'")
        print(f"Characters: {chars} | Words: {words} | Tokens: {count}")
        return

    report = run_token_analysis()
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="backslashreplace").decode("ascii"))

    # Save by default or if --save flag is provided
    out_file = save_sample_results()
    print(f"\n[INFO] Sample results saved to: {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
