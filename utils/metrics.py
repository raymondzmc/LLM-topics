import os
import json
import itertools
import numpy as np
from openai import OpenAI
from collections import defaultdict
from settings import settings
from llm import jinja_template_manager
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence
from gensim.downloader import load as gensim_load
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import pairwise_distances


def compute_llm_rating(topics: list[list[str]], model: str = "gpt-4o"):
    system_prompt = jinja_template_manager.render("topic_ratings_system.jinja")
    topic_ratings: list[int] = []

    def render_messages(topic: list[str]):
        user_prompt = jinja_template_manager.render(
            "topic_ratings_user.jinja",
            topic=topic,
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        return messages

    client = OpenAI(api_key=settings.openai_api_key)
    for topic in topics:
        messages = render_messages(topic)
        rating: int | None = None
        temperature: float = 0.0
        num_attempts: int = 0
        while rating is None and num_attempts < 5:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=1,
            )
            try:
                _rating = int(response.choices[0].message.content)
            except:
                continue

            if _rating in [1, 2, 3]:
                rating = _rating
            else:
                temperature += 0.1
                num_attempts += 1

        if rating is None:
            raise ValueError(f"Could not get a valid LLM rating for topic \"{topic}\" after 5 attempts.")

        topic_ratings.append(rating)
    return topic_ratings


class PairwiseEmbeddings(AbstractMetric):
    def __init__(self, embeddings, topk=10):
        super().__init__()
        self.embeddings = embeddings
        self.topk = topk

    def score(self, model_output):
        topics = model_output["topics"]
        assert all(word in self.embeddings for topic in topics for word in topic), "All words must be in the embeddings"

        result = 0.0
        for topic in topics:
            E = []
            for word in topic[0:self.topk]:
                word_embedding = self.embeddings[word]   # OpenAI embeddings are already to length 1
                E.append(word_embedding)

            E = np.array(E)
            distances = np.sum(1 - pairwise_distances(E, metric='cosine') - np.diag(np.ones(len(E))))
            topic_coherence = distances/(self.topk*(self.topk-1))

            # Update result with the computed coherence of the topic
            result += topic_coherence
        result = result/len(topics)
        return result


class Word2VecEmbeddingCoherence(AbstractMetric):
    def __init__(self, word2vec_path='word2vec-google-news-300.kv', binary=True, top_k=10):
        """
        :param word2vec_path: if word2vec_file is specified, it retrieves the
         word embeddings file (in word2vec format) to compute similarities
         between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.binary = binary
        self.top_k = top_k
        if word2vec_path is None:
            self.wv = gensim_load('word2vec-google-news-300')
            self.wv.save_word2vec_format(word2vec_path, binary=binary)
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self, model_output):
        topics = model_output["topics"]
        if self.top_k > len(topics[0]):
            raise Exception(f'Words in topics are less than top_k: {self.top_k}')
        else:
            arrays = []
            for topic in topics:
                if len(topic) > 0:
                    local_sim = []
                    for w1, w2 in itertools.combinations(topic[:self.top_k], 2):
                        if (w1 in self.wv.index_to_key and w2 in self.wv.index_to_key):
                            local_sim.append(self.wv.similarity(w1, w2))
                    arrays.append(np.mean(local_sim))
            return np.mean(arrays)


def evaluate_topic_model(model_output, top_words=10, test_corpus=None, embeddings=None):
    assert 'topics' in model_output, "model_output must contain 'topics'"

    evaluation_results = {}

    td = TopicDiversity(topk=top_words)
    td_score = td.score(model_output)
    print("Topic Diversity:", td_score)
    evaluation_results['topic_diversity'] = float(td_score)
    
    irbo = InvertedRBO(topk=top_words)
    irbo_score = irbo.score(model_output)
    print("Inverted RBO:", irbo_score)
    evaluation_results['inverted_rbo'] = float(irbo_score)
    
    if test_corpus is not None:
        npmi = Coherence(measure='c_npmi', texts=test_corpus, topk=top_words)
        npmi_score = npmi.score(model_output)
        print("NPMI:", npmi_score)
        evaluation_results['npmi'] = float(npmi_score)

    if embeddings is not None:
        openai_we = PairwiseEmbeddings(embeddings, topk=top_words)
        openai_we_score = openai_we.score(model_output)
        print("(OpenAI) Word Embeddings:", openai_we_score)
        evaluation_results['openai_word_embeddings'] = float(openai_we_score)

    word2vec_we = Word2VecEmbeddingCoherence(top_k=top_words)
    word2vec_we_score = word2vec_we.score(model_output)
    print("(Word2Vec) Word Embeddings:", word2vec_we_score)
    evaluation_results['word2vec_word_embeddings'] = float(word2vec_we_score)

    llm_ratings = compute_llm_rating(model_output['topics'])
    llm_average_rating = float(np.mean(llm_ratings))
    print("LLM Rating:", llm_average_rating)
    evaluation_results['llm_rating'] = llm_average_rating
    return evaluation_results


def compute_aggregate_results(results_path):
    aggregated_results = defaultdict(float)
    counts = defaultdict(int)
    for seed_dir in os.listdir(results_path):
        results_file = os.path.join(results_path, seed_dir, 'evaluation_results.json')
        if os.path.exists(results_file):
            results = json.load(open(results_file, encoding='utf-8'))

            # Backwards compatibility (used to save topics and results)
            if isinstance(results, list):
                results = results[1]

            assert isinstance(results, dict)
            for k, v in results.items():
                aggregated_results[k] += v
                counts[k] += 1

    metrics = aggregated_results.keys()
    for k in metrics:
        aggregated_results[k] /= counts[k]
        print(f"[{k}] {aggregated_results[k]} (from {counts[k]} runs)")
    return aggregated_results
        