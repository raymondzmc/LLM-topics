# Statistical Analysis Summary: stackoverflow

## Best Methods by K and Metric

| K | llm_rating | npmi | inverted_rbo | purity | word2vec_word_embeddings | cv |
|---|---|---|---|---|---|---|
| 25 | Llama-3.2-1B-Instruct (2.968±0.033)<br>Llama-3.2-3B-Instruct (2.952±0.044) | zeroshot (-0.197±0.007)<br>combined (-0.202±0.008) | lda (0.997±0.001)<br>bertopic (0.996±0.001) | Llama-3.2-1B-Instruct (0.615±0.020)<br>Llama-3.2-3B-Instruct (0.584±0.026) | Llama-3.2-1B-Instruct (0.182±0.004)<br>Llama-3.2-3B-Instruct (0.181±0.003) | lda (0.621±0.010) |
| 50 | Llama-3.2-3B-Instruct (2.976±0.009) | combined (-0.198±0.007)<br>zeroshot (-0.200±0.012) | lda (0.999±0.001) | Llama-3.2-1B-Instruct (0.655±0.007)<br>Llama-3.2-3B-Instruct (0.653±0.009) | Llama-3.2-3B-Instruct (0.183±0.001)<br>Llama-3.2-1B-Instruct (0.182±0.001) | lda (0.644±0.007) |
| 75 | Llama-3.2-3B-Instruct (2.965±0.015)<br>Llama-3.2-1B-Instruct (2.952±0.020) | combined (-0.184±0.013)<br>fastopic (-0.260±0.177) | bertopic (0.997±0.000) | Llama-3.2-1B-Instruct (0.667±0.004)<br>Llama-3.2-3B-Instruct (0.657±0.010) | Llama-3.2-1B-Instruct (0.180±0.002)<br>Llama-3.2-3B-Instruct (0.178±0.001) | Llama-3.2-3B-Instruct (0.614±0.004)<br>fastopic (0.433±0.231) |
| 100 | Llama-3.2-3B-Instruct (2.958±0.004)<br>Llama-3.2-1B-Instruct (2.958±0.013) | fastopic (-0.152±0.121)<br>combined (-0.193±0.007)<br>zeroshot (-0.199±0.003)<br>... | bertopic (0.997±0.000) | Llama-3.2-1B-Instruct (0.675±0.005) | Llama-3.2-1B-Instruct (0.178±0.002)<br>Llama-3.2-3B-Instruct (0.177±0.003) | Llama-3.2-3B-Instruct (0.621±0.006) |

## Detailed Statistics

### K = 25

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.9680 | 0.0335 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 2.9520 | 0.0438 |
| 3 | bertopic | 2.8800 | 0.0283 |
| 4 | zeroshot | 2.8160 | 0.0607 |
| 5 | combined | 2.7680 | 0.0657 |
| 6 | prodlda | 2.6240 | 0.0829 |
| 7 | etm | 2.4160 | 0.0829 |
| 8 | fastopic | 2.0640 | 0.0219 |
| 9 | lda | 2.0160 | 0.0219 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | -0.1968 | 0.0069 |
| 2 | combined ⭐ | -0.2024 | 0.0081 |
| 3 | prodlda | -0.2251 | 0.0076 |
| 4 | etm | -0.2453 | 0.0144 |
| 5 | bertopic | -0.2723 | 0.0108 |
| 6 | fastopic | -0.2990 | 0.0774 |
| 7 | Llama-3.2-1B-Instruct | -0.3486 | 0.0038 |
| 8 | Llama-3.2-3B-Instruct | -0.3613 | 0.0059 |
| 9 | lda | -0.3940 | 0.0089 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.9974 | 0.0012 |
| 2 | bertopic ⭐ | 0.9965 | 0.0005 |
| 3 | Llama-3.2-3B-Instruct | 0.9956 | 0.0007 |
| 4 | combined | 0.9941 | 0.0009 |
| 5 | Llama-3.2-1B-Instruct | 0.9939 | 0.0004 |
| 6 | prodlda | 0.9934 | 0.0006 |
| 7 | zeroshot | 0.9931 | 0.0010 |
| 8 | etm | 0.9426 | 0.0082 |
| 9 | fastopic | 0.5640 | 0.0586 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.6150 | 0.0199 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.5844 | 0.0260 |
| 3 | zeroshot | 0.2576 | 0.0021 |
| 4 | combined | 0.2498 | 0.0094 |
| 5 | bertopic | 0.2242 | 0.0066 |
| 6 | prodlda | 0.2197 | 0.0070 |
| 7 | etm | 0.1547 | 0.0127 |
| 8 | lda | 0.1313 | 0.0068 |
| 9 | fastopic | 0.1092 | 0.0052 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.1818 | 0.0040 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.1808 | 0.0035 |
| 3 | etm | 0.1625 | 0.0030 |
| 4 | bertopic | 0.1457 | 0.0042 |
| 5 | zeroshot | 0.1416 | 0.0018 |
| 6 | combined | 0.1327 | 0.0056 |
| 7 | prodlda | 0.1237 | 0.0030 |
| 8 | fastopic | 0.1081 | 0.0094 |
| 9 | lda | 0.1040 | 0.0012 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.6209 | 0.0097 |
| 2 | Llama-3.2-3B-Instruct | 0.6054 | 0.0068 |
| 3 | Llama-3.2-1B-Instruct | 0.5899 | 0.0049 |
| 4 | fastopic | 0.4916 | 0.0832 |
| 5 | bertopic | 0.4792 | 0.0111 |
| 6 | etm | 0.4444 | 0.0156 |
| 7 | prodlda | 0.4205 | 0.0079 |
| 8 | combined | 0.3983 | 0.0097 |
| 9 | zeroshot | 0.3895 | 0.0072 |

### K = 50

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 2.9760 | 0.0089 |
| 2 | Llama-3.2-1B-Instruct | 2.9560 | 0.0089 |
| 3 | zeroshot | 2.8240 | 0.0261 |
| 4 | combined | 2.7920 | 0.0593 |
| 5 | bertopic | 2.7600 | 0.0510 |
| 6 | prodlda | 2.6200 | 0.0316 |
| 7 | etm | 2.2440 | 0.0219 |
| 8 | fastopic | 2.1120 | 0.0179 |
| 9 | lda | 1.9920 | 0.0179 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | -0.1984 | 0.0073 |
| 2 | zeroshot ⭐ | -0.2004 | 0.0116 |
| 3 | prodlda | -0.2300 | 0.0077 |
| 4 | etm | -0.2661 | 0.0081 |
| 5 | bertopic | -0.2911 | 0.0031 |
| 6 | Llama-3.2-1B-Instruct | -0.3562 | 0.0018 |
| 7 | Llama-3.2-3B-Instruct | -0.3666 | 0.0025 |
| 8 | fastopic | -0.3799 | 0.0114 |
| 9 | lda | -0.4084 | 0.0054 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.9986 | 0.0006 |
| 2 | bertopic | 0.9966 | 0.0004 |
| 3 | Llama-3.2-3B-Instruct | 0.9932 | 0.0003 |
| 4 | Llama-3.2-1B-Instruct | 0.9904 | 0.0005 |
| 5 | prodlda | 0.9859 | 0.0015 |
| 6 | zeroshot | 0.9838 | 0.0025 |
| 7 | combined | 0.9827 | 0.0011 |
| 8 | etm | 0.9227 | 0.0032 |
| 9 | fastopic | 0.6152 | 0.0247 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.6548 | 0.0072 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.6534 | 0.0086 |
| 3 | zeroshot | 0.2775 | 0.0057 |
| 4 | combined | 0.2588 | 0.0036 |
| 5 | bertopic | 0.2547 | 0.0135 |
| 6 | prodlda | 0.2321 | 0.0105 |
| 7 | fastopic | 0.1710 | 0.0064 |
| 8 | lda | 0.1645 | 0.0034 |
| 9 | etm | 0.1497 | 0.0097 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.1832 | 0.0013 |
| 2 | Llama-3.2-1B-Instruct ⭐ | 0.1821 | 0.0013 |
| 3 | etm | 0.1667 | 0.0009 |
| 4 | bertopic | 0.1457 | 0.0025 |
| 5 | zeroshot | 0.1403 | 0.0010 |
| 6 | combined | 0.1339 | 0.0019 |
| 7 | prodlda | 0.1209 | 0.0006 |
| 8 | fastopic | 0.1018 | 0.0033 |
| 9 | lda | 0.0979 | 0.0014 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | lda ⭐ | 0.6442 | 0.0074 |
| 2 | Llama-3.2-3B-Instruct | 0.6095 | 0.0045 |
| 3 | Llama-3.2-1B-Instruct | 0.5979 | 0.0016 |
| 4 | fastopic | 0.5927 | 0.0155 |
| 5 | bertopic | 0.5015 | 0.0025 |
| 6 | etm | 0.4678 | 0.0104 |
| 7 | prodlda | 0.4244 | 0.0092 |
| 8 | zeroshot | 0.4019 | 0.0098 |
| 9 | combined | 0.3936 | 0.0077 |

### K = 75

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 2.9653 | 0.0152 |
| 2 | Llama-3.2-1B-Instruct ⭐ | 2.9520 | 0.0202 |
| 3 | zeroshot | 2.8320 | 0.0223 |
| 4 | bertopic | 2.6667 | 0.0211 |
| 5 | combined | 2.6453 | 0.1349 |
| 6 | prodlda | 2.5760 | 0.0751 |
| 7 | etm | 2.2053 | 0.0307 |
| 8 | fastopic | 2.1093 | 0.0916 |
| 9 | lda | 2.0320 | 0.0242 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | combined ⭐ | -0.1840 | 0.0130 |
| 2 | zeroshot | -0.2012 | 0.0019 |
| 3 | prodlda | -0.2249 | 0.0075 |
| 4 | fastopic ⭐ | -0.2600 | 0.1772 |
| 5 | etm | -0.2907 | 0.0031 |
| 6 | bertopic | -0.2960 | 0.0030 |
| 7 | lda | -0.2961 | 0.0073 |
| 8 | Llama-3.2-1B-Instruct | -0.3605 | 0.0021 |
| 9 | Llama-3.2-3B-Instruct | -0.3705 | 0.0024 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | bertopic ⭐ | 0.9966 | 0.0002 |
| 2 | Llama-3.2-3B-Instruct | 0.9914 | 0.0002 |
| 3 | lda | 0.9907 | 0.0011 |
| 4 | Llama-3.2-1B-Instruct | 0.9890 | 0.0006 |
| 5 | prodlda | 0.9804 | 0.0037 |
| 6 | zeroshot | 0.9797 | 0.0005 |
| 7 | combined | 0.9338 | 0.0513 |
| 8 | etm | 0.9262 | 0.0044 |
| 9 | fastopic | 0.4170 | 0.3149 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.6667 | 0.0039 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.6568 | 0.0098 |
| 3 | bertopic | 0.2849 | 0.0078 |
| 4 | zeroshot | 0.2766 | 0.0041 |
| 5 | prodlda | 0.2352 | 0.0057 |
| 6 | combined | 0.2081 | 0.0621 |
| 7 | lda | 0.1872 | 0.0049 |
| 8 | etm | 0.1463 | 0.0114 |
| 9 | fastopic | 0.1434 | 0.0463 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.1795 | 0.0020 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.1781 | 0.0009 |
| 3 | etm | 0.1690 | 0.0022 |
| 4 | zeroshot | 0.1398 | 0.0017 |
| 5 | bertopic | 0.1373 | 0.0021 |
| 6 | combined | 0.1310 | 0.0030 |
| 7 | prodlda | 0.1219 | 0.0018 |
| 8 | fastopic | 0.1177 | 0.0261 |
| 9 | lda | 0.1021 | 0.0019 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.6143 | 0.0042 |
| 2 | Llama-3.2-1B-Instruct | 0.6027 | 0.0031 |
| 3 | bertopic | 0.5067 | 0.0059 |
| 4 | etm | 0.4956 | 0.0038 |
| 5 | lda | 0.4859 | 0.0108 |
| 6 | fastopic ⭐ | 0.4327 | 0.2314 |
| 7 | prodlda | 0.4179 | 0.0077 |
| 8 | zeroshot | 0.3974 | 0.0036 |
| 9 | combined | 0.3755 | 0.0258 |

### K = 100

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 2.9580 | 0.0045 |
| 2 | Llama-3.2-1B-Instruct ⭐ | 2.9580 | 0.0130 |
| 3 | zeroshot | 2.8180 | 0.0228 |
| 4 | combined | 2.7560 | 0.0740 |
| 5 | bertopic | 2.6280 | 0.0356 |
| 6 | prodlda | 2.5540 | 0.0789 |
| 7 | etm | 2.2240 | 0.0270 |
| 8 | fastopic | 2.0360 | 0.0391 |
| 9 | lda | 1.7780 | 0.0228 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | fastopic ⭐ | -0.1519 | 0.1206 |
| 2 | combined ⭐ | -0.1927 | 0.0069 |
| 3 | zeroshot ⭐ | -0.1995 | 0.0027 |
| 4 | prodlda ⭐ | -0.2176 | 0.0110 |
| 5 | bertopic | -0.2997 | 0.0035 |
| 6 | etm | -0.3019 | 0.0035 |
| 7 | lda | -0.3559 | 0.0029 |
| 8 | Llama-3.2-1B-Instruct | -0.3640 | 0.0009 |
| 9 | Llama-3.2-3B-Instruct | -0.3755 | 0.0039 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | bertopic ⭐ | 0.9967 | 0.0002 |
| 2 | Llama-3.2-3B-Instruct | 0.9908 | 0.0003 |
| 3 | Llama-3.2-1B-Instruct | 0.9886 | 0.0003 |
| 4 | zeroshot | 0.9757 | 0.0008 |
| 5 | prodlda | 0.9753 | 0.0048 |
| 6 | combined | 0.9633 | 0.0111 |
| 7 | etm | 0.9319 | 0.0017 |
| 8 | lda | 0.8721 | 0.0125 |
| 9 | fastopic | 0.3192 | 0.3541 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.6752 | 0.0051 |
| 2 | Llama-3.2-3B-Instruct | 0.6621 | 0.0071 |
| 3 | bertopic | 0.2974 | 0.0027 |
| 4 | zeroshot | 0.2791 | 0.0034 |
| 5 | combined | 0.2460 | 0.0315 |
| 6 | prodlda | 0.2410 | 0.0068 |
| 7 | lda | 0.1766 | 0.0019 |
| 8 | etm | 0.1518 | 0.0045 |
| 9 | fastopic | 0.1068 | 0.0294 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.1783 | 0.0019 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 0.1774 | 0.0026 |
| 3 | etm | 0.1689 | 0.0026 |
| 4 | zeroshot | 0.1391 | 0.0020 |
| 5 | combined | 0.1332 | 0.0015 |
| 6 | bertopic | 0.1327 | 0.0011 |
| 7 | fastopic | 0.1321 | 0.0198 |
| 8 | prodlda | 0.1218 | 0.0014 |
| 9 | lda | 0.0970 | 0.0005 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-3B-Instruct ⭐ | 0.6214 | 0.0058 |
| 2 | Llama-3.2-1B-Instruct | 0.6060 | 0.0015 |
| 3 | lda | 0.5941 | 0.0037 |
| 4 | bertopic | 0.5138 | 0.0040 |
| 5 | etm | 0.5103 | 0.0046 |
| 6 | prodlda | 0.4107 | 0.0088 |
| 7 | zeroshot | 0.3964 | 0.0026 |
| 8 | combined | 0.3916 | 0.0059 |
| 9 | fastopic | 0.2924 | 0.1605 |
