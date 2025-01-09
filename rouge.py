import nltk
import numpy as np
import re
import string
from collections import Counter
from rouge_score import rouge_scorer, scoring
from openai import OpenAI
from mosestokenizer import *
from collections import defaultdict
from tqdm import tqdm

nltk.download('stopwords')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2',  'rougeL', 'rougeLsum'], use_stemmer=True)

def rouge_metrics(references: list[str], predictions: list[str]):
    references = ["\n".join(nltk.sent_tokenize(ref.strip())) for ref in references]
    predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    aggregator = scoring.BootstrapAggregator(n_samples=10000)
    result = {k: [] for k in scorer.rouge_types}
    for ref_text, pred_text in zip(references, predictions):
        score = scorer.score(ref_text, pred_text)
        aggregator.add_scores(score)
    result = aggregator.aggregate()
    for key in result:
        result[key] = round(result[key].mid.fmeasure, 4)
    return result