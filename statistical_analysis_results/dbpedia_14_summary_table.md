# Statistical Analysis Summary: dbpedia_14

## Best Methods by K and Metric

| K | llm_rating | npmi | inverted_rbo | purity | word2vec_word_embeddings | cv |
|---|---|---|---|---|---|---|
| 25 | Llama-3.2-1B-Instruct (2.936±0.022) | combined (0.150±0.006)<br>zeroshot (0.149±0.003)<br>prodlda (0.145±0.009) | lda (0.999±0.000)<br>combined (0.999±0.001)<br>prodlda (0.998±0.001)<br>... | Llama-3.2-3B-Instruct (0.918±0.007) | Llama-3.2-1B-Instruct (0.259±0.003) | combined (0.723±0.010)<br>zeroshot (0.720±0.008)<br>prodlda (0.712±0.015) |
| 50 | Llama-3.2-1B-Instruct (2.944±0.017) | combined (0.146±0.003)<br>zeroshot (0.145±0.005) | lda (1.000±0.000) | Llama-3.2-3B-Instruct (0.912±0.005)<br>Llama-3.2-1B-Instruct (0.907±0.008) | Llama-3.2-1B-Instruct (0.243±0.003) | zeroshot (0.713±0.008)<br>combined (0.711±0.005) |
| 75 | Llama-3.2-1B-Instruct (2.947±0.016) | combined (0.147±0.002)<br>zeroshot (0.147±0.002) | lda (0.998±0.000) | Llama-3.2-1B-Instruct (0.924±0.004) | etm (0.240±0.005) | combined (0.713±0.004)<br>zeroshot (0.712±0.002) |
| 100 | Llama-3.2-1B-Instruct (2.940±0.021) | combined (0.148±0.002)<br>zeroshot (0.147±0.004) | prodlda (0.995±0.000) | Llama-3.2-1B-Instruct (0.923±0.002) | etm (0.241±0.005) | combined (0.713±0.005)<br>zeroshot (0.712±0.008) |

## Detailed Statistics

### K = 25

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9360 | 0.0219 |
| 2 | Llama-3.2-3B-Instruct | 2.9040 | 0.0219 |
| 3 | combined | 2.6480 | 0.0657 |
| 4 | etm | 2.6160 | 0.0358 |
| 5 | prodlda | 2.6080 | 0.0912 |
| 6 | zeroshot | 2.5680 | 0.0867 |
| 7 | bertopic | 2.3040 | 0.0607 |
| 8 | lda | 2.0640 | 0.0669 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.1504 | 0.0063 |
| 2 | zeroshot ⭐ | 0.1486 | 0.0031 |
| 3 | prodlda ⭐ | 0.1451 | 0.0093 |
| 4 | Llama-3.2-1B-Instruct | 0.1311 | 0.0025 |
| 5 | Llama-3.2-3B-Instruct | 0.0990 | 0.0021 |
| 6 | etm | 0.0865 | 0.0151 |
| 7 | bertopic | 0.0065 | 0.0103 |
| 8 | lda | -0.0223 | 0.0022 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.9991 | 0.0003 |
| 2 | combined ⭐ | 0.9987 | 0.0009 |
| 3 | prodlda ⭐ | 0.9984 | 0.0007 |
| 4 | zeroshot ⭐ | 0.9980 | 0.0012 |
| 5 | Llama-3.2-3B-Instruct | 0.9938 | 0.0004 |
| 6 | Llama-3.2-1B-Instruct | 0.9935 | 0.0003 |
| 7 | bertopic | 0.9868 | 0.0017 |
| 8 | etm | 0.9728 | 0.0036 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.9179 | 0.0066 |
| 2 | Llama-3.2-1B-Instruct | 0.8894 | 0.0097 |
| 3 | zeroshot | 0.8434 | 0.0102 |
| 4 | combined | 0.8385 | 0.0036 |
| 5 | prodlda | 0.8257 | 0.0043 |
| 6 | lda | 0.4763 | 0.0141 |
| 7 | etm | 0.3723 | 0.0118 |
| 8 | bertopic | 0.2191 | 0.0178 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.2588 | 0.0030 |
| 2 | Llama-3.2-3B-Instruct | 0.2256 | 0.0026 |
| 3 | etm | 0.2133 | 0.0042 |
| 4 | zeroshot | 0.1530 | 0.0026 |
| 5 | combined | 0.1504 | 0.0032 |
| 6 | prodlda | 0.1479 | 0.0053 |
| 7 | bertopic | 0.1422 | 0.0041 |
| 8 | lda | 0.1092 | 0.0012 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.7227 | 0.0099 |
| 2 | zeroshot ⭐ | 0.7195 | 0.0077 |
| 3 | prodlda ⭐ | 0.7121 | 0.0147 |
| 4 | Llama-3.2-1B-Instruct | 0.6661 | 0.0015 |
| 5 | Llama-3.2-3B-Instruct | 0.6118 | 0.0045 |
| 6 | etm | 0.5643 | 0.0328 |
| 7 | bertopic | 0.4093 | 0.0176 |
| 8 | lda | 0.3415 | 0.0040 |

### K = 50

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9440 | 0.0167 |
| 2 | Llama-3.2-3B-Instruct | 2.8920 | 0.0303 |
| 3 | zeroshot | 2.6560 | 0.0410 |
| 4 | etm | 2.6280 | 0.0576 |
| 5 | combined | 2.5960 | 0.0590 |
| 6 | prodlda | 2.5720 | 0.0460 |
| 7 | bertopic | 2.3240 | 0.0623 |
| 8 | lda | 2.0080 | 0.0303 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.1464 | 0.0033 |
| 2 | zeroshot ⭐ | 0.1446 | 0.0048 |
| 3 | prodlda | 0.1335 | 0.0058 |
| 4 | Llama-3.2-1B-Instruct | 0.1262 | 0.0031 |
| 5 | Llama-3.2-3B-Instruct | 0.0865 | 0.0043 |
| 6 | etm | 0.0823 | 0.0021 |
| 7 | bertopic | -0.0100 | 0.0047 |
| 8 | lda | -0.0422 | 0.0027 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.9999 | 0.0000 |
| 2 | prodlda | 0.9968 | 0.0004 |
| 3 | zeroshot | 0.9966 | 0.0007 |
| 4 | combined | 0.9966 | 0.0006 |
| 5 | bertopic | 0.9909 | 0.0008 |
| 6 | Llama-3.2-1B-Instruct | 0.9908 | 0.0005 |
| 7 | Llama-3.2-3B-Instruct | 0.9894 | 0.0006 |
| 8 | etm | 0.9733 | 0.0025 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.9121 | 0.0054 |
| 2 | Llama-3.2-1B-Instruct ⭐ | 0.9067 | 0.0084 |
| 3 | zeroshot | 0.8498 | 0.0095 |
| 4 | combined | 0.8472 | 0.0059 |
| 5 | prodlda | 0.8222 | 0.0062 |
| 6 | lda | 0.5248 | 0.0214 |
| 7 | etm | 0.3627 | 0.0174 |
| 8 | bertopic | 0.2774 | 0.0260 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.2433 | 0.0028 |
| 2 | etm | 0.2325 | 0.0082 |
| 3 | Llama-3.2-3B-Instruct | 0.2086 | 0.0039 |
| 4 | zeroshot | 0.1480 | 0.0020 |
| 5 | combined | 0.1452 | 0.0013 |
| 6 | prodlda | 0.1421 | 0.0030 |
| 7 | bertopic | 0.1393 | 0.0024 |
| 8 | lda | 0.0995 | 0.0016 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | 0.7125 | 0.0084 |
| 2 | combined ⭐ | 0.7105 | 0.0052 |
| 3 | prodlda | 0.6875 | 0.0096 |
| 4 | Llama-3.2-1B-Instruct | 0.6664 | 0.0062 |
| 5 | Llama-3.2-3B-Instruct | 0.5907 | 0.0108 |
| 6 | etm | 0.5681 | 0.0064 |
| 7 | bertopic | 0.3980 | 0.0112 |
| 8 | lda | 0.3162 | 0.0050 |

### K = 75

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9467 | 0.0163 |
| 2 | Llama-3.2-3B-Instruct | 2.8613 | 0.0292 |
| 3 | zeroshot | 2.6720 | 0.0292 |
| 4 | etm | 2.6667 | 0.0189 |
| 5 | combined | 2.6613 | 0.0242 |
| 6 | prodlda | 2.6373 | 0.0475 |
| 7 | bertopic | 2.3413 | 0.0538 |
| 8 | lda | 1.9547 | 0.0260 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.1474 | 0.0018 |
| 2 | zeroshot ⭐ | 0.1474 | 0.0019 |
| 3 | prodlda | 0.1410 | 0.0029 |
| 4 | Llama-3.2-1B-Instruct | 0.1092 | 0.0046 |
| 5 | etm | 0.0768 | 0.0047 |
| 6 | Llama-3.2-3B-Instruct | 0.0692 | 0.0014 |
| 7 | bertopic | -0.0144 | 0.0034 |
| 8 | lda | -0.0557 | 0.0028 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.9982 | 0.0004 |
| 2 | prodlda | 0.9958 | 0.0001 |
| 3 | combined | 0.9952 | 0.0005 |
| 4 | zeroshot | 0.9949 | 0.0002 |
| 5 | bertopic | 0.9927 | 0.0003 |
| 6 | Llama-3.2-3B-Instruct | 0.9892 | 0.0004 |
| 7 | Llama-3.2-1B-Instruct | 0.9888 | 0.0003 |
| 8 | etm | 0.9748 | 0.0028 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.9237 | 0.0044 |
| 2 | Llama-3.2-3B-Instruct | 0.9145 | 0.0042 |
| 3 | zeroshot | 0.8642 | 0.0048 |
| 4 | combined | 0.8495 | 0.0085 |
| 5 | prodlda | 0.8387 | 0.0091 |
| 6 | lda | 0.5470 | 0.0075 |
| 7 | bertopic | 0.3593 | 0.0158 |
| 8 | etm | 0.3213 | 0.0318 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.2397 | 0.0048 |
| 2 | Llama-3.2-1B-Instruct | 0.2304 | 0.0031 |
| 3 | Llama-3.2-3B-Instruct | 0.2011 | 0.0018 |
| 4 | zeroshot | 0.1526 | 0.0021 |
| 5 | combined | 0.1495 | 0.0014 |
| 6 | prodlda | 0.1469 | 0.0033 |
| 7 | bertopic | 0.1384 | 0.0024 |
| 8 | lda | 0.0964 | 0.0017 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.7127 | 0.0035 |
| 2 | zeroshot ⭐ | 0.7120 | 0.0020 |
| 3 | prodlda | 0.6999 | 0.0034 |
| 4 | Llama-3.2-1B-Instruct | 0.6353 | 0.0088 |
| 5 | Llama-3.2-3B-Instruct | 0.5594 | 0.0038 |
| 6 | etm | 0.5544 | 0.0108 |
| 7 | bertopic | 0.3929 | 0.0099 |
| 8 | lda | 0.3033 | 0.0073 |

### K = 100

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9400 | 0.0212 |
| 2 | Llama-3.2-3B-Instruct | 2.8560 | 0.0134 |
| 3 | zeroshot | 2.7100 | 0.0265 |
| 4 | combined | 2.7000 | 0.0316 |
| 5 | prodlda | 2.6540 | 0.0288 |
| 6 | etm | 2.6320 | 0.0559 |
| 7 | bertopic | 2.3420 | 0.0449 |
| 8 | lda | 1.9820 | 0.0217 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.1476 | 0.0024 |
| 2 | zeroshot ⭐ | 0.1472 | 0.0043 |
| 3 | prodlda | 0.1368 | 0.0056 |
| 4 | Llama-3.2-1B-Instruct | 0.1024 | 0.0013 |
| 5 | etm | 0.0731 | 0.0079 |
| 6 | Llama-3.2-3B-Instruct | 0.0579 | 0.0013 |
| 7 | bertopic | -0.0180 | 0.0068 |
| 8 | lda | -0.0440 | 0.0010 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | prodlda ⭐ | 0.9946 | 0.0002 |
| 2 | combined | 0.9941 | 0.0003 |
| 3 | bertopic | 0.9935 | 0.0003 |
| 4 | zeroshot | 0.9934 | 0.0005 |
| 5 | Llama-3.2-3B-Instruct | 0.9896 | 0.0003 |
| 6 | Llama-3.2-1B-Instruct | 0.9886 | 0.0002 |
| 7 | lda | 0.9852 | 0.0007 |
| 8 | etm | 0.9783 | 0.0010 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.9226 | 0.0023 |
| 2 | Llama-3.2-3B-Instruct | 0.9148 | 0.0032 |
| 3 | zeroshot | 0.8646 | 0.0078 |
| 4 | combined | 0.8573 | 0.0107 |
| 5 | prodlda | 0.8480 | 0.0073 |
| 6 | lda | 0.5722 | 0.0165 |
| 7 | bertopic | 0.4098 | 0.0114 |
| 8 | etm | 0.3288 | 0.0330 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.2406 | 0.0046 |
| 2 | Llama-3.2-1B-Instruct | 0.2253 | 0.0022 |
| 3 | Llama-3.2-3B-Instruct | 0.1942 | 0.0013 |
| 4 | zeroshot | 0.1550 | 0.0018 |
| 5 | combined | 0.1528 | 0.0020 |
| 6 | prodlda | 0.1483 | 0.0024 |
| 7 | bertopic | 0.1373 | 0.0020 |
| 8 | lda | 0.1014 | 0.0013 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | 0.7129 | 0.0052 |
| 2 | zeroshot ⭐ | 0.7121 | 0.0078 |
| 3 | prodlda | 0.6924 | 0.0097 |
| 4 | Llama-3.2-1B-Instruct | 0.6221 | 0.0034 |
| 5 | etm | 0.5493 | 0.0167 |
| 6 | Llama-3.2-3B-Instruct | 0.5391 | 0.0027 |
| 7 | bertopic | 0.3947 | 0.0099 |
| 8 | lda | 0.3283 | 0.0059 |
