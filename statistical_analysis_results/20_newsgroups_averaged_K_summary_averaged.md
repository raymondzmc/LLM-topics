# Statistical Analysis Summary: 20_newsgroups (Averaged across K values)
## Strategy: averaged

## Best Methods by Metric (Averaged across K ∈ {25, 50, 75, 100})

| Metric | Best Method(s) | Mean ± Std |
|--------|----------------|------------|
| llm_rating | Llama-3.2-1B-Instruct | 2.8888 ± 0.0161 |
| npmi | etm | 0.0451 ± 0.0015 |
| inverted_rbo | fastopic | 0.9966 ± 0.0005 |
| purity | Llama-3.2-1B-Instruct | 0.5612 ± 0.0024 |
| word2vec_word_embeddings | etm | 0.2008 ± 0.0020 |

## Detailed Rankings

### llm_rating

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.8888 | 0.0161 | 5 |
| 2 | Llama-3.2-3B-Instruct | 2.8628 | 0.0152 | 5 |
| 3 | fastopic | 2.5388 | 0.0187 | 5 |
| 4 | zeroshot | 2.5315 | 0.0126 | 5 |
| 5 | combined | 2.5165 | 0.0264 | 5 |
| 6 | prodlda | 2.4345 | 0.0258 | 5 |
| 7 | bertopic | 2.3237 | 0.0152 | 5 |
| 8 | etm | 2.3068 | 0.0133 | 5 |
| 9 | lda | 2.1698 | 0.0267 | 5 |

### npmi

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | etm ⭐ | 0.0451 | 0.0015 | 5 |
| 2 | zeroshot | 0.0290 | 0.0051 | 5 |
| 3 | combined | 0.0277 | 0.0007 | 5 |
| 4 | prodlda | 0.0132 | 0.0021 | 5 |
| 5 | Llama-3.2-1B-Instruct | 0.0008 | 0.0015 | 5 |
| 6 | fastopic | -0.0081 | 0.0014 | 5 |
| 7 | lda | -0.0083 | 0.0038 | 5 |
| 8 | bertopic | -0.0094 | 0.0054 | 5 |
| 9 | Llama-3.2-3B-Instruct | -0.0138 | 0.0020 | 5 |

### inverted_rbo

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | fastopic ⭐ | 0.9966 | 0.0005 | 5 |
| 2 | bertopic | 0.9913 | 0.0008 | 5 |
| 3 | Llama-3.2-3B-Instruct | 0.9909 | 0.0002 | 5 |
| 4 | zeroshot | 0.9907 | 0.0006 | 5 |
| 5 | prodlda | 0.9906 | 0.0005 | 5 |
| 6 | combined | 0.9903 | 0.0005 | 5 |
| 7 | Llama-3.2-1B-Instruct | 0.9898 | 0.0002 | 5 |
| 8 | lda | 0.9747 | 0.0008 | 5 |
| 9 | etm | 0.9328 | 0.0017 | 5 |

### purity

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.5612 | 0.0024 | 5 |
| 2 | Llama-3.2-3B-Instruct | 0.5406 | 0.0037 | 5 |
| 3 | fastopic | 0.4093 | 0.0044 | 5 |
| 4 | zeroshot | 0.3788 | 0.0027 | 5 |
| 5 | combined | 0.3695 | 0.0023 | 5 |
| 6 | etm | 0.3634 | 0.0060 | 5 |
| 7 | prodlda | 0.3405 | 0.0014 | 5 |
| 8 | bertopic | 0.3159 | 0.0073 | 5 |
| 9 | lda | 0.3046 | 0.0061 | 5 |

### word2vec_word_embeddings

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | etm ⭐ | 0.2008 | 0.0020 | 5 |
| 2 | Llama-3.2-1B-Instruct | 0.1963 | 0.0012 | 5 |
| 3 | fastopic | 0.1848 | 0.0017 | 5 |
| 4 | Llama-3.2-3B-Instruct | 0.1816 | 0.0017 | 5 |
| 5 | zeroshot | 0.1563 | 0.0013 | 5 |
| 6 | combined | 0.1537 | 0.0009 | 5 |
| 7 | prodlda | 0.1503 | 0.0009 | 5 |
| 8 | lda | 0.1468 | 0.0018 | 5 |
| 9 | bertopic | 0.1391 | 0.0009 | 5 |
