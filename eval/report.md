# Retrieval Evaluation: JA-207652

Question set: 36 natural-language questions, hand-labelled with the Addendum question numbers that answer them (`eval/questions.json`).

Metrics:
- **Recall@k**: fraction of gold question numbers covered by the top k returned chunks, averaged over questions.
- **MRR**: mean reciprocal rank of the first returned chunk containing a gold question.

## Configurations tested

| Chunking | k1 | b | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| qa | 1.2 | 0.75 | 39 | 0.889 | 0.986 | 1.000 | 0.972 |
| qa | 1.5 | 0.3 | 39 | 0.861 | 1.000 | 1.000 | 0.958 |
| window:40:20 | 1.2 | 0.75 | 93 | 0.819 | 0.986 | 0.986 | 0.926 |
| window:40:20 | 1.5 | 0.3 | 93 | 0.861 | 0.986 | 0.986 | 0.954 |
| window:80:40 | 1.2 | 0.75 | 46 | 0.903 | 0.986 | 1.000 | 0.972 |
| window:80:40 | 1.5 | 0.3 | 46 | 0.917 | 0.986 | 1.000 | 0.986 |

**Best by MRR:** chunking `window:80:40`, k1=1.5, b=0.3 (MRR 0.986, Recall@3 0.986).

## Misses for the best configuration (gold not in top 3)

None.

## Limitations
- One document (39 question/answer items) and 36 questions. Results are indicative, not statistically robust.
- Gold labels were written by the same team that built the system, so they may favour the chunking unit.
- Retrieval is lexical (BM25). Dense embeddings were not evaluated in this version.
- The best and default configurations differ by one question, so the choice of chunking is not conclusive.
