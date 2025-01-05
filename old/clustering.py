import numpy as np
from sklearn.cluster import KMeans
from datasets import load_from_disk
import torch
import json
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

# Replace with your actual model name and token
model_name = "meta-llama/Llama-3.2-1B"
token = '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'
login(token)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16  # Ensure compatibility with your hardware
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load your vocabulary
training_set = load_from_disk('training_set')
with open('training_set/vocab.json', 'r') as f:
    vocab = json.load(f)

# Map words to token IDs
word_token_ids = {}
for word in vocab:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if token_ids:
        word_token_ids[word] = token_ids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Get the embedding layer
embedding_layer = model.get_input_embeddings()

# Collect embeddings for each word
word_embeddings = {}
for word, token_ids in tqdm(word_token_ids.items()):
    # Get embeddings for the token IDs
    token_embeddings = embedding_layer(torch.tensor(token_ids).to(device))
    # Average the embeddings if multiple tokens
    word_embedding = token_embeddings.mean(dim=0)
    # Convert to float32 before converting to NumPy array
    word_embeddings[word] = word_embedding.float().cpu().detach().numpy()

# Prepare data for clustering
embedding_matrix = np.stack(list(word_embeddings.values()))
word_list = list(word_embeddings.keys())

# Perform KMeans clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embedding_matrix)
labels = kmeans.labels_

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Compute distances of all words to each cluster centroid
# Create a dictionary to hold top words for each cluster
top_words_per_cluster = {}

for cluster_id in range(num_clusters):
    centroid = cluster_centers[cluster_id]
    # Compute distances from all words to this centroid
    distances = np.linalg.norm(embedding_matrix - centroid, axis=1)
    # Combine words and their distances
    word_distances = list(zip(word_list, distances))
    # Sort by distance
    word_distances.sort(key=lambda x: x[1])
    # Select top 10 words
    top_words = [word for word, distance in word_distances[:10]]
    top_words_per_cluster[cluster_id] = top_words

# Print the top 10 words for each cluster
for cluster_id in sorted(top_words_per_cluster.keys()):
    top_words = top_words_per_cluster[cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(", ".join(top_words))