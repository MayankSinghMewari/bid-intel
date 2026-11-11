# Retrieval Evaluation: Bid1 + Bid2

20 questions (10 Bid1, 10 Bid2) in `eval/bid_questions.json`. A hit is a retrieved chunk from the gold file and page. Each search is filtered to the question's bid.

Recall@k = share of questions with at least one gold passage in the top k. MRR = mean reciprocal rank of the first gold passage.

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.65 | 0.9 | 0.9 | 0.758 | 0.9 |
| bm25 + query expansion | 0.65 | 0.9 | 0.95 | 0.777 | 0.9 |
| dense (bge-small) | 0.75 | 0.9 | 0.95 | 0.835 | 59.1 |
| hybrid RRF | 0.85 | 0.9 | 0.95 | 0.879 | 24.7 |
| hybrid RRF + cross-encoder rerank | 0.65 | 0.95 | 1.0 | 0.804 | 5235.7 |

Limitations: 20 questions written by the system's author. With this sample size a difference of one question (0.05) is not significant.
