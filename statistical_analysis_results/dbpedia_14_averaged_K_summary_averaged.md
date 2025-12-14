# Statistical Analysis Summary: dbpedia_14 (Averaged across K values)
## Strategy: averaged

## Best Methods by Metric (Averaged across K ∈ {25, 50, 75, 100})

| Metric | Best Method(s) | Mean ± Std |
|--------|----------------|------------|
| llm_rating | Llama-3.2-1B-Instruct | 2.9417 ± 0.0094 |
| npmi | combined (+1 tied) | 0.1479 ± 0.0023 |
| inverted_rbo | prodlda (+1 tied) | 0.9964 ± 0.0002 |
| purity | Llama-3.2-3B-Instruct | 0.9148 ± 0.0027 |
| word2vec_word_embeddings | Llama-3.2-1B-Instruct | 0.2394 ± 0.0010 |

## Detailed Rankings

### llm_rating

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9417 | 0.0094 | 5 |
| 2 | Llama-3.2-3B-Instruct | 2.8783 | 0.0116 | 5 |
| 3 | zeroshot | 2.6515 | 0.0236 | 5 |
| 4 | combined | 2.6513 | 0.0260 | 5 |
| 5 | etm | 2.6357 | 0.0324 | 5 |
| 6 | prodlda | 2.6178 | 0.0222 | 5 |
| 7 | bertopic | 2.3278 | 0.0205 | 5 |
| 8 | lda | 2.0022 | 0.0151 | 5 |

### npmi

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | combined ⭐ | 0.1479 | 0.0023 | 5 |
| 2 | zeroshot ⭐ | 0.1469 | 0.0009 | 5 |
| 3 | prodlda | 0.1391 | 0.0045 | 5 |
| 4 | Llama-3.2-1B-Instruct | 0.1172 | 0.0016 | 5 |
| 5 | etm | 0.0797 | 0.0042 | 5 |
| 6 | Llama-3.2-3B-Instruct | 0.0781 | 0.0013 | 5 |
| 7 | bertopic | -0.0090 | 0.0028 | 5 |
| 8 | lda | -0.0410 | 0.0008 | 5 |

### inverted_rbo

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | prodlda ⭐ | 0.9964 | 0.0002 | 5 |
| 2 | combined ⭐ | 0.9962 | 0.0002 | 5 |
| 3 | zeroshot | 0.9957 | 0.0004 | 5 |
| 4 | lda | 0.9956 | 0.0002 | 5 |
| 5 | bertopic | 0.9910 | 0.0007 | 5 |
| 6 | Llama-3.2-3B-Instruct | 0.9905 | 0.0003 | 5 |
| 7 | Llama-3.2-1B-Instruct | 0.9904 | 0.0001 | 5 |
| 8 | etm | 0.9748 | 0.0012 | 5 |

### purity

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.9148 | 0.0027 | 5 |
| 2 | Llama-3.2-1B-Instruct | 0.9106 | 0.0011 | 5 |
| 3 | zeroshot | 0.8555 | 0.0051 | 5 |
| 4 | combined | 0.8481 | 0.0036 | 5 |
| 5 | prodlda | 0.8336 | 0.0024 | 5 |
| 6 | lda | 0.5301 | 0.0054 | 5 |
| 7 | etm | 0.3463 | 0.0130 | 5 |
| 8 | bertopic | 0.3164 | 0.0057 | 5 |

### word2vec_word_embeddings

| Rank | Method | Mean | Std | n |
|------|--------|------|-----|---|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.2394 | 0.0010 | 5 |
| 2 | etm | 0.2315 | 0.0037 | 5 |
| 3 | Llama-3.2-3B-Instruct | 0.2074 | 0.0014 | 5 |
| 4 | zeroshot | 0.1521 | 0.0007 | 5 |
| 5 | combined | 0.1495 | 0.0012 | 5 |
| 6 | prodlda | 0.1463 | 0.0021 | 5 |
| 7 | bertopic | 0.1393 | 0.0023 | 5 |
| 8 | lda | 0.1017 | 0.0007 | 5 |
