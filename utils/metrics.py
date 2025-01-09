import math
from collections import defaultdict
import numpy as np

def compute_npmi_score(topics, documents):
    """
    Computes the topic coherence (NPMI) for each topic in 'topics' 
    based on the collection of documents in 'documents'.

    :param topics: List of topics, each topic is a list of words [ [w1, w2, ...], [w1, w2, ...], ... ]
    :param documents: List of documents, each document is a list of words [ [d1, d2, ...], [d1, d2, ...], ... ]
    :return: A list of coherence scores, one for each topic
    """

    # 1. Collect all unique words from the topics
    topic_words = set()
    for topic in topics:
        topic_words.update(topic)

    # 2. Initialize counters
    word_doc_count = defaultdict(int)   # Count of docs that contain each word
    pair_doc_count = defaultdict(int)   # Count of docs that contain each (word1, word2) pair

    # 3. Single pass over documents to fill the counters
    num_docs = len(documents)
    for doc in documents:
        doc_set = set(doc)
        # Only keep words that appear in any of the topics
        filtered_words = doc_set.intersection(topic_words)
        
        # Update individual word counts
        for w in filtered_words:
            word_doc_count[w] += 1
        
        # Update pairwise counts
        filtered_list = sorted(filtered_words)
        for i in range(len(filtered_list)):
            for j in range(i + 1, len(filtered_list)):
                w1 = filtered_list[i]
                w2 = filtered_list[j]
                pair_doc_count[(w1, w2)] += 1

    # Helper function to compute NPMI for a pair of words
    def npmi_score(w1, w2):
        pair_count = pair_doc_count.get((w1, w2), 0)
        if pair_count == 0:
            return 0.0  # No co-occurrence => NPMI = 0
        
        p_xy = pair_count / num_docs
        p_x = word_doc_count[w1] / num_docs
        p_y = word_doc_count[w2] / num_docs
        
        # If any probability is zero (should not happen if pair_count > 0, but just in case):
        if p_x == 0 or p_y == 0 or p_xy == 0:
            return 0.0
        
        # PMI = log ( p_xy / (p_x * p_y) )
        # Choose log base 2 or natural log (ln) consistently.
        pmi = math.log(p_xy / (p_x * p_y), 2)
        
        # NPMI = PMI / -log(p_xy)
        if p_xy <= 0:
            return 0.0
        return pmi / -math.log(p_xy, 2)

    # 4. Compute coherence (average NPMI) for each topic
    topic_coherences = []
    for topic in topics:
        total_npmi = 0.0
        count_pairs = 0
        
        # Go through each pair of words in the topic
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w1, w2 = topic[i], topic[j]
                # Ensure (w1, w2) matches how we counted in pair_doc_count
                if w1 > w2:
                    w1, w2 = w2, w1
                
                total_npmi += npmi_score(w1, w2)
                count_pairs += 1
        
        # Average NPMI for all pairs in the topic
        if count_pairs > 0:
            topic_coherences.append(total_npmi / count_pairs)
        else:
            topic_coherences.append(0.0)

    return np.mean(topic_coherences)


