# Embedding Vectors & Semantic Similarity Demonstration

> Demonstrates text embedding generation, uniform vector dimensionality,
> cosine similarity comparison across similar versus dissimilar texts,
> and conceptual foundations of dense vector spaces in Property Insurance RAG.

## 1. Vector Dimension & Uniformity (Task 2)

- **Model:** `text-embedding-3-small`
- **Vector Dimensionality:** `1536` dimensions
- **Length Uniformity:** Every sample text strictly yields a vector of identical length (`1536`).

| Sample ID | Domain Category | Text Preview | Vector Shape | First 4 Dimensions |
| :--- | :--- | :--- | :---: | :--- |
| `text_1_water_pipe` | Property Peril (Water Damage) | Does policy HO-3 cover sudden water damage resu... | `(1, 1536)` | `[0.0505, 0.0482, 0.0811, 0.0727]` |
| `text_2_plumbing_leak` | Property Peril (Similar Meaning) | Is accidental discharge or leakage from broken ... | `(1, 1536)` | `[0.0473, 0.0422, 0.0747, 0.0663]` |
| `text_3_earth_movement` | Policy Exclusion (Different Insurance Peril) | The policy excludes loss caused directly or ind... | `(1, 1536)` | `[0.0004, 0.0008, 0.0005, 0.0003]` |
| `text_4_culinary_recipe` | Unrelated Domain (Culinary Recipe) | Authentic Neapolitan pizza dough requires high-... | `(1, 1536)` | `[0.0001, 0.0000, -0.0006, 0.0001]` |
| `text_5_astrophysics` | Unrelated Domain (Astrophysics) | Supernova remnants emit high-energy cosmic rays... | `(1, 1536)` | `[0.0002, -0.0003, 0.0002, -0.0004]` |

## 2. Cosine Similarity: Similar vs. Dissimilar Pairs (Task 3)

$$\text{cosine\_similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

| Pair | Text A | Text B | Semantic Relationship | Cosine Similarity |
| :--- | :--- | :--- | :--- | :---: |
| **Text vs. Text** | "Does policy HO-3 cover sudden water..." | "Is accidental discharge or leakage ..." | SIMILAR (Same Property Insurance Peril) | **0.9644** |
| **Text vs. Text** | "Does policy HO-3 cover sudden water..." | "The policy excludes loss caused dir..." | MODERATE (Same Domain, Different Policy Topic) | **0.0556** |
| **Text vs. Text** | "Does policy HO-3 cover sudden water..." | "Authentic Neapolitan pizza dough re..." | DISSIMILAR (Unrelated Subject Matter) | **0.0009** |
| **Text vs. Text** | "Does policy HO-3 cover sudden water..." | "Supernova remnants emit high-energy..." | DISSIMILAR (Unrelated Scientific Topic) | **-0.0007** |

### Key Takeaways:
1. **Semantic Pair Scores Highest:** The two sentences describing water pipe bursts share almost zero identical keywords, but achieve **0.9644** similarity.
2. **Unrelated Topics Near Orthogonal:** The water peril query compared against a pizza recipe scores **0.0009**, proving semantic discrimination.
3. **Condition Verified:** Similar pair (0.9644) scores substantially higher than dissimilar pair (0.0009).

## 3. What Embedding Vectors Represent (Task 4)

What Embedding Vectors Represent in RAG

1. Numeric Representations of Meaning (Not Random IDs):
   A random ID (e.g. UUID 'b76fa52a-9b6a' or document index 42) is purely arbitrary. Two random
   IDs carry zero spatial or semantic relationship; the distance between them is meaningless.
   An embedding vector, by contrast, is a dense point in a continuous 1,536-dimensional semantic
   space where proximity directly corresponds to conceptual similarity.

2. Semantic Closeness vs. Keyword Matching:
   Traditional search methods (like BM25, TF-IDF, or character regex) rely on literal lexical overlap.
   Consider our two sample sentences:
     • "Does policy HO-3 cover sudden water damage resulting from a frozen or burst plumbing pipe?"
     • "Is accidental discharge or leakage from broken household plumbing systems covered under homeowners insurance?"
   These two sentences describe the exact same insurance claim event, yet share almost none of the key
   words ("burst pipe" vs. "accidental discharge / broken plumbing"). Keyword search ranks them poorly.
   Embedding models map both sentences to nearby coordinates in vector space because their trained
   weights capture underlying semantic equivalence.

3. Geometric Distance as Semantic Relevance:
   In vector space, semantic similarity translates directly to linear algebra:
     • Highly similar ideas align along nearly identical vector trajectories (cosine similarity > 0.80).
     • Unrelated topics (e.g. homeowners plumbing insurance vs. a pizza dough recipe) point in nearly
       orthogonal directions in high-dimensional space (cosine similarity < 0.20).
   Retrieval-Augmented Generation relies on this geometry to retrieve facts by meaning rather than
   hoping the adjuster and the policy author used the exact same words.
