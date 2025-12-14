# Statistical Analysis Summary: stackoverflow (Averaged across K values)
## Strategy: averaged

## Best Methods by Metric (Averaged across K ∈ {25, 50, 75, 100})

| Metric | Best Method(s) | Mean ± Std |
|--------|----------------|------------|
| llm_rating | Llama-3.2-3B-Instruct (+1 tied) | 2.9628 ± 0.0136 |
| npmi | combined (+2 tied) | -0.1944 ± 0.0064 |
| inverted_rbo | bertopic | 0.9966 ± 0.0002 |
| purity | Llama-3.2-1B-Instruct | 0.6529 ± 0.0045 |
| word2vec_word_embeddings | Llama-3.2-1B-Instruct (+1 tied) | 0.1804 ± 0.0006 |

## Detailed Rankings

### llm_rating

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-3B-Instruct ⭐ | 2.9628 | 0.0136 | 5 |
| 2 | Llama-3.2-1B-Instruct ⭐ | 2.9585 | 0.0127 | 5 |
| 3 | zeroshot | 2.8225 | 0.0206 | 5 |
| 4 | combined | 2.7403 | 0.0525 | 5 |
| 5 | bertopic | 2.7337 | 0.0154 | 5 |
| 6 | prodlda | 2.5935 | 0.0326 | 5 |
| 7 | etm | 2.2723 | 0.0232 | 5 |
| 8 | fastopic | 2.0803 | 0.0265 | 5 |
| 9 | lda | 1.9545 | 0.0086 | 5 |

### npmi

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | combined ⭐ | -0.1944 | 0.0064 | 5 |
| 2 | zeroshot ⭐ | -0.1995 | 0.0031 | 5 |
| 3 | prodlda | -0.2244 | 0.0039 | 5 |
| 4 | fastopic ⭐ | -0.2727 | 0.0742 | 5 |
| 5 | etm | -0.2760 | 0.0050 | 5 |
| 6 | bertopic | -0.2898 | 0.0024 | 5 |
| 7 | Llama-3.2-1B-Instruct | -0.3573 | 0.0011 | 5 |
| 8 | lda | -0.3636 | 0.0029 | 5 |
| 9 | Llama-3.2-3B-Instruct | -0.3684 | 0.0024 | 5 |

### inverted_rbo

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | bertopic ⭐ | 0.9966 | 0.0002 | 5 |
| 2 | Llama-3.2-3B-Instruct | 0.9928 | 0.0002 | 5 |
| 3 | Llama-3.2-1B-Instruct | 0.9905 | 0.0002 | 5 |
| 4 | prodlda | 0.9837 | 0.0021 | 5 |
| 5 | zeroshot | 0.9831 | 0.0007 | 5 |
| 6 | combined | 0.9685 | 0.0129 | 5 |
| 7 | lda | 0.9647 | 0.0030 | 5 |
| 8 | etm | 0.9309 | 0.0030 | 5 |
| 9 | fastopic | 0.4789 | 0.1595 | 5 |

### purity

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.6529 | 0.0045 | 5 |
| 2 | Llama-3.2-3B-Instruct | 0.6391 | 0.0079 | 5 |
| 3 | zeroshot | 0.2727 | 0.0026 | 5 |
| 4 | bertopic | 0.2653 | 0.0049 | 5 |
| 5 | combined | 0.2407 | 0.0148 | 5 |
| 6 | prodlda | 0.2320 | 0.0032 | 5 |
| 7 | lda | 0.1649 | 0.0023 | 5 |
| 8 | etm | 0.1506 | 0.0051 | 5 |
| 9 | fastopic | 0.1326 | 0.0135 | 5 |

### word2vec_word_embeddings

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.1804 | 0.0006 | 5 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.1799 | 0.0012 | 5 |
| 3 | etm | 0.1668 | 0.0008 | 5 |
| 4 | bertopic | 0.1403 | 0.0013 | 5 |
| 5 | zeroshot | 0.1402 | 0.0007 | 5 |
| 6 | combined | 0.1327 | 0.0015 | 5 |
| 7 | prodlda | 0.1221 | 0.0012 | 5 |
| 8 | fastopic | 0.1149 | 0.0111 | 5 |
| 9 | lda | 0.1002 | 0.0008 | 5 |
