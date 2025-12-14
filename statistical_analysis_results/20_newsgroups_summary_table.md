# Statistical Analysis Summary: 20_newsgroups

## Best Methods by K and Metric

| K | llm_rating | npmi | inverted_rbo | purity | word2vec_word_embeddings | cv |
|---|---|---|---|---|---|---|
| 25 | Llama-3.2-1B-Instruct (2.888±0.033)<br>Llama-3.2-3B-Instruct (2.888±0.052) | etm (0.052±0.005)<br>zeroshot (0.045±0.014)<br>combined (0.042±0.011) | fastopic (0.998±0.001) | Llama-3.2-1B-Instruct (0.540±0.007) | Llama-3.2-1B-Instruct (0.212±0.003) | zeroshot (0.598±0.015)<br>combined (0.597±0.013)<br>prodlda (0.584±0.018) |
| 50 | Llama-3.2-1B-Instruct (2.892±0.011)<br>Llama-3.2-3B-Instruct (2.884±0.026) | etm (0.047±0.001) | fastopic (0.996±0.001) | Llama-3.2-1B-Instruct (0.560±0.005) | etm (0.202±0.002) | zeroshot (0.574±0.016)<br>combined (0.567±0.012) |
| 75 | Llama-3.2-1B-Instruct (2.885±0.040)<br>Llama-3.2-3B-Instruct (2.845±0.028) | etm (0.042±0.002) | fastopic (0.996±0.001) | Llama-3.2-1B-Instruct (0.566±0.004) | etm (0.200±0.002) | zeroshot (0.545±0.009)<br>combined (0.535±0.012) |
| 100 | Llama-3.2-1B-Instruct (2.890±0.028) | etm (0.039±0.002) | fastopic (0.996±0.001) | Llama-3.2-1B-Instruct (0.578±0.006) | etm (0.198±0.002) | zeroshot (0.541±0.006) |

## Detailed Statistics

### K = 25

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.8880 | 0.0335 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 2.8880 | 0.0522 |
| 3 | fastopic | 2.5840 | 0.0456 |
| 4 | zeroshot | 2.5760 | 0.0219 |
| 5 | combined | 2.5520 | 0.0522 |
| 6 | prodlda | 2.4880 | 0.0657 |
| 7 | etm | 2.3280 | 0.0335 |
| 8 | bertopic | 2.2800 | 0.0632 |
| 9 | lda | 2.2240 | 0.0876 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.0519 | 0.0053 |
| 2 | zeroshot ⭐ | 0.0453 | 0.0140 |
| 3 | combined ⭐ | 0.0423 | 0.0112 |
| 4 | prodlda | 0.0363 | 0.0056 |
| 5 | Llama-3.2-1B-Instruct | 0.0233 | 0.0023 |
| 6 | lda | 0.0202 | 0.0105 |
| 7 | fastopic | 0.0122 | 0.0075 |
| 8 | Llama-3.2-3B-Instruct | 0.0047 | 0.0038 |
| 9 | bertopic | 0.0045 | 0.0161 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | fastopic ⭐ | 0.9981 | 0.0010 |
| 2 | Llama-3.2-3B-Instruct | 0.9951 | 0.0006 |
| 3 | Llama-3.2-1B-Instruct | 0.9945 | 0.0008 |
| 4 | zeroshot | 0.9908 | 0.0027 |
| 5 | combined | 0.9901 | 0.0017 |
| 6 | prodlda | 0.9898 | 0.0013 |
| 7 | bertopic | 0.9879 | 0.0019 |
| 8 | lda | 0.9487 | 0.0027 |
| 9 | etm | 0.9406 | 0.0089 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.5399 | 0.0071 |
| 2 | Llama-3.2-3B-Instruct | 0.5024 | 0.0126 |
| 3 | fastopic | 0.3812 | 0.0190 |
| 4 | zeroshot | 0.3417 | 0.0113 |
| 5 | etm | 0.3324 | 0.0141 |
| 6 | combined | 0.3318 | 0.0141 |
| 7 | prodlda | 0.3127 | 0.0098 |
| 8 | lda | 0.3004 | 0.0134 |
| 9 | bertopic | 0.2248 | 0.0163 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.2116 | 0.0026 |
| 2 | etm | 0.2032 | 0.0045 |
| 3 | Llama-3.2-3B-Instruct | 0.1912 | 0.0038 |
| 4 | fastopic | 0.1898 | 0.0015 |
| 5 | zeroshot | 0.1643 | 0.0038 |
| 6 | combined | 0.1619 | 0.0030 |
| 7 | prodlda | 0.1612 | 0.0033 |
| 8 | lda | 0.1597 | 0.0058 |
| 9 | bertopic | 0.1414 | 0.0043 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | 0.5979 | 0.0151 |
| 2 | combined ⭐ | 0.5968 | 0.0127 |
| 3 | prodlda ⭐ | 0.5841 | 0.0183 |
| 4 | fastopic | 0.5691 | 0.0173 |
| 5 | Llama-3.2-1B-Instruct | 0.5538 | 0.0070 |
| 6 | etm | 0.5485 | 0.0128 |
| 7 | Llama-3.2-3B-Instruct | 0.5099 | 0.0156 |
| 8 | lda | 0.4762 | 0.0204 |
| 9 | bertopic | 0.4196 | 0.0166 |

### K = 50

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.8920 | 0.0110 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 2.8840 | 0.0261 |
| 3 | combined | 2.5600 | 0.0721 |
| 4 | zeroshot | 2.5440 | 0.0607 |
| 5 | fastopic | 2.5400 | 0.0424 |
| 6 | prodlda | 2.4520 | 0.0716 |
| 7 | etm | 2.3480 | 0.0363 |
| 8 | bertopic | 2.3320 | 0.0363 |
| 9 | lda | 2.2080 | 0.0687 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.0474 | 0.0009 |
| 2 | combined | 0.0365 | 0.0055 |
| 3 | zeroshot | 0.0310 | 0.0071 |
| 4 | prodlda | 0.0118 | 0.0042 |
| 5 | lda | 0.0045 | 0.0068 |
| 6 | Llama-3.2-1B-Instruct | 0.0029 | 0.0062 |
| 7 | fastopic | -0.0067 | 0.0059 |
| 8 | bertopic | -0.0080 | 0.0028 |
| 9 | Llama-3.2-3B-Instruct | -0.0119 | 0.0047 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | fastopic ⭐ | 0.9959 | 0.0010 |
| 2 | zeroshot | 0.9914 | 0.0013 |
| 3 | prodlda | 0.9913 | 0.0008 |
| 4 | bertopic | 0.9911 | 0.0009 |
| 5 | combined | 0.9908 | 0.0013 |
| 6 | Llama-3.2-3B-Instruct | 0.9906 | 0.0006 |
| 7 | Llama-3.2-1B-Instruct | 0.9896 | 0.0003 |
| 8 | lda | 0.9754 | 0.0030 |
| 9 | etm | 0.9281 | 0.0029 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.5602 | 0.0052 |
| 2 | Llama-3.2-3B-Instruct | 0.5384 | 0.0043 |
| 3 | fastopic | 0.4019 | 0.0061 |
| 4 | zeroshot | 0.3849 | 0.0087 |
| 5 | combined | 0.3780 | 0.0069 |
| 6 | etm | 0.3762 | 0.0110 |
| 7 | prodlda | 0.3431 | 0.0084 |
| 8 | lda | 0.3071 | 0.0085 |
| 9 | bertopic | 0.2831 | 0.0150 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.2020 | 0.0016 |
| 2 | Llama-3.2-1B-Instruct | 0.1970 | 0.0024 |
| 3 | fastopic | 0.1857 | 0.0030 |
| 4 | Llama-3.2-3B-Instruct | 0.1827 | 0.0023 |
| 5 | zeroshot | 0.1557 | 0.0027 |
| 6 | combined | 0.1547 | 0.0029 |
| 7 | prodlda | 0.1506 | 0.0029 |
| 8 | lda | 0.1498 | 0.0023 |
| 9 | bertopic | 0.1399 | 0.0020 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | 0.5735 | 0.0163 |
| 2 | combined ⭐ | 0.5675 | 0.0124 |
| 3 | fastopic | 0.5381 | 0.0139 |
| 4 | etm | 0.5283 | 0.0031 |
| 5 | prodlda | 0.5253 | 0.0082 |
| 6 | Llama-3.2-1B-Instruct | 0.5121 | 0.0067 |
| 7 | Llama-3.2-3B-Instruct | 0.4742 | 0.0088 |
| 8 | lda | 0.4545 | 0.0136 |
| 9 | bertopic | 0.4035 | 0.0046 |

### K = 75

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.8853 | 0.0396 |
| 2 | Llama-3.2-3B-Instruct ⭐ | 2.8453 | 0.0276 |
| 3 | zeroshot | 2.5200 | 0.0411 |
| 4 | fastopic | 2.5173 | 0.0146 |
| 5 | combined | 2.4640 | 0.0484 |
| 6 | prodlda | 2.4080 | 0.0348 |
| 7 | bertopic | 2.3387 | 0.0223 |
| 8 | etm | 2.2853 | 0.0260 |
| 9 | lda | 2.1493 | 0.0318 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.0422 | 0.0019 |
| 2 | zeroshot | 0.0202 | 0.0064 |
| 3 | combined | 0.0186 | 0.0082 |
| 4 | prodlda | 0.0037 | 0.0037 |
| 5 | Llama-3.2-1B-Instruct | -0.0061 | 0.0023 |
| 6 | bertopic | -0.0145 | 0.0033 |
| 7 | fastopic | -0.0150 | 0.0041 |
| 8 | Llama-3.2-3B-Instruct | -0.0203 | 0.0046 |
| 9 | lda | -0.0242 | 0.0068 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | fastopic ⭐ | 0.9965 | 0.0006 |
| 2 | bertopic | 0.9930 | 0.0003 |
| 3 | zeroshot | 0.9908 | 0.0002 |
| 4 | combined | 0.9908 | 0.0008 |
| 5 | prodlda | 0.9906 | 0.0005 |
| 6 | Llama-3.2-3B-Instruct | 0.9893 | 0.0006 |
| 7 | Llama-3.2-1B-Instruct | 0.9879 | 0.0003 |
| 8 | lda | 0.9858 | 0.0009 |
| 9 | etm | 0.9289 | 0.0041 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.5664 | 0.0038 |
| 2 | Llama-3.2-3B-Instruct | 0.5572 | 0.0055 |
| 3 | fastopic | 0.4161 | 0.0080 |
| 4 | zeroshot | 0.3941 | 0.0042 |
| 5 | combined | 0.3851 | 0.0065 |
| 6 | etm | 0.3758 | 0.0164 |
| 7 | bertopic | 0.3597 | 0.0229 |
| 8 | prodlda | 0.3525 | 0.0075 |
| 9 | lda | 0.3081 | 0.0060 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.1999 | 0.0016 |
| 2 | Llama-3.2-1B-Instruct | 0.1917 | 0.0024 |
| 3 | fastopic | 0.1832 | 0.0026 |
| 4 | Llama-3.2-3B-Instruct | 0.1773 | 0.0016 |
| 5 | zeroshot | 0.1516 | 0.0011 |
| 6 | combined | 0.1501 | 0.0029 |
| 7 | prodlda | 0.1454 | 0.0013 |
| 8 | lda | 0.1416 | 0.0019 |
| 9 | bertopic | 0.1394 | 0.0020 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | 0.5445 | 0.0094 |
| 2 | combined ⭐ | 0.5347 | 0.0122 |
| 3 | etm | 0.5125 | 0.0029 |
| 4 | fastopic | 0.5114 | 0.0065 |
| 5 | prodlda | 0.5103 | 0.0088 |
| 6 | Llama-3.2-1B-Instruct | 0.4998 | 0.0051 |
| 7 | Llama-3.2-3B-Instruct | 0.4558 | 0.0062 |
| 8 | lda | 0.4101 | 0.0092 |
| 9 | bertopic | 0.3961 | 0.0101 |

### K = 100

#### llm_rating

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 2.8900 | 0.0283 |
| 2 | Llama-3.2-3B-Instruct | 2.8340 | 0.0114 |
| 3 | fastopic | 2.5140 | 0.0207 |
| 4 | combined | 2.4900 | 0.0400 |
| 5 | zeroshot | 2.4860 | 0.0297 |
| 6 | prodlda | 2.3900 | 0.0552 |
| 7 | bertopic | 2.3440 | 0.0305 |
| 8 | etm | 2.2660 | 0.0288 |
| 9 | lda | 2.0980 | 0.0164 |

#### npmi

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.0389 | 0.0018 |
| 2 | zeroshot | 0.0195 | 0.0043 |
| 3 | combined | 0.0132 | 0.0025 |
| 4 | prodlda | 0.0010 | 0.0023 |
| 5 | Llama-3.2-1B-Instruct | -0.0167 | 0.0024 |
| 6 | bertopic | -0.0194 | 0.0049 |
| 7 | fastopic | -0.0227 | 0.0034 |
| 8 | Llama-3.2-3B-Instruct | -0.0278 | 0.0030 |
| 9 | lda | -0.0336 | 0.0012 |

#### inverted_rbo

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | fastopic ⭐ | 0.9958 | 0.0008 |
| 2 | bertopic | 0.9933 | 0.0004 |
| 3 | prodlda | 0.9906 | 0.0005 |
| 4 | zeroshot | 0.9896 | 0.0006 |
| 5 | combined | 0.9896 | 0.0005 |
| 6 | lda | 0.9889 | 0.0004 |
| 7 | Llama-3.2-3B-Instruct | 0.9887 | 0.0005 |
| 8 | Llama-3.2-1B-Instruct | 0.9873 | 0.0003 |
| 9 | etm | 0.9335 | 0.0014 |

#### purity

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | Llama-3.2-1B-Instruct ⭐ | 0.5782 | 0.0057 |
| 2 | Llama-3.2-3B-Instruct | 0.5645 | 0.0038 |
| 3 | fastopic | 0.4381 | 0.0065 |
| 4 | bertopic | 0.3961 | 0.0053 |
| 5 | zeroshot | 0.3944 | 0.0063 |
| 6 | combined | 0.3831 | 0.0094 |
| 7 | etm | 0.3691 | 0.0096 |
| 8 | prodlda | 0.3537 | 0.0056 |
| 9 | lda | 0.3029 | 0.0043 |

#### word2vec_word_embeddings

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | etm ⭐ | 0.1980 | 0.0019 |
| 2 | Llama-3.2-1B-Instruct | 0.1850 | 0.0014 |
| 3 | fastopic | 0.1804 | 0.0016 |
| 4 | Llama-3.2-3B-Instruct | 0.1753 | 0.0019 |
| 5 | zeroshot | 0.1534 | 0.0011 |
| 6 | combined | 0.1480 | 0.0015 |
| 7 | prodlda | 0.1439 | 0.0019 |
| 8 | lda | 0.1363 | 0.0017 |
| 9 | bertopic | 0.1359 | 0.0018 |

#### cv

| Rank | Method | Mean | Std |
|------|--------|------|-----|
| 1 | zeroshot ⭐ | 0.5413 | 0.0058 |
| 2 | combined | 0.5209 | 0.0077 |
| 3 | etm | 0.5031 | 0.0046 |
| 4 | fastopic | 0.4976 | 0.0070 |
| 5 | prodlda | 0.4962 | 0.0044 |
| 6 | Llama-3.2-1B-Instruct | 0.4798 | 0.0037 |
| 7 | Llama-3.2-3B-Instruct | 0.4443 | 0.0051 |
| 8 | lda | 0.3864 | 0.0085 |
| 9 | bertopic | 0.3857 | 0.0096 |
