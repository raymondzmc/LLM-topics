import os
import json
import numpy as np
from utils.metrics import (
    compute_llm_rating,
    Word2VecEmbeddingCoherence,
    Coherence,
)
from llm import jinja_template_manager
from settings import settings
from openai import OpenAI
from collections import defaultdict

def render_messages(topic: list[str]):
    system_prompt = jinja_template_manager.render(
        "categorize_topic_dbpedia.jinja",
        topic=topic,
    )
    messages = [
        {'role': 'user', 'content': system_prompt},
    ]
    return messages

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

with open('data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt', 'r', encoding='utf-8') as f:
    bow_corpus = [line.strip().split() for line in f]
    ignore_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) == 0 or doc == ['null']]
    bow_corpus = [doc for i, doc in enumerate(bow_corpus) if i not in ignore_indices]

result_paths = [
    ('etm', 'results/dbpedia_14/etm_K25/seed_0'),
    ('zeroshot', 'results/dbpedia_14/zeroshot_K25/seed_0'),
    ('ours', 'results/dbpedia_14/Llama-3.2-1B-Instruct/25_KL/seed_0'),
]
we_coherence = Word2VecEmbeddingCoherence(top_k=10)
npmi = Coherence(measure='c_npmi', texts=bow_corpus, topk=10)
client = OpenAI(api_key=settings.openai_api_key)

# Dictionary to store topics by category
topics_by_category = defaultdict(list)

for model_name, path in result_paths:
    topic_file = os.path.join(path, 'topics.json')
    with open(topic_file, 'r') as f:
        topics = json.load(f)

    for i, topic_words in enumerate(topics):
        topic_words = topic_words[:10]
        llm_score = float(compute_llm_rating([topic_words])[0])
        word2vec_score = float(we_coherence.score({"topics": [topic_words]}))
        npmi_score = float(npmi.score({"topics": [topic_words]}))
        
        messages = render_messages(topic_words)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        category = response.choices[0].message.content.strip()
        
        # Store the topic with all its metrics
        topics_by_category[category].append({
            'model': model_name,
            'topic_id': i,
            'words': topic_words,
            'llm_score': llm_score,
            'word2vec_score': word2vec_score,
            'npmi_score': npmi_score
        })

# save topics_by_category to a json file
with open('topics_by_category.json', 'w') as f:
    json.dump(topics_by_category, f, indent=4, cls=NumpyEncoder)

# Print topics organized by category
print("\n" + "="*80)
print("TOPICS ORGANIZED BY CATEGORY")
print("="*80)

for category, topics in sorted(topics_by_category.items()):
    print(f"\n\n### CATEGORY: {category} ###")
    print("-" * 50)
    
    # Sort topics by LLM score within each category (highest first)
    topics.sort(key=lambda x: x['llm_score'], reverse=True)
    
    for topic in topics:
        print(f"Model: {topic['model']} (Topic #{topic['topic_id']})")
        print(f"Words: {', '.join(topic['words'])}")
        print(f"Scores: LLM={topic['llm_score']:.4f}, Word2Vec={topic['word2vec_score']:.4f}, NPMI={topic['npmi_score']:.4f}")
        print("-" * 30)