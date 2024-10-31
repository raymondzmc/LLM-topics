from datasets import load_from_disk
import os
import torch
import json
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel
import pdb
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

model_name = "meta-llama/Llama-3.2-1B"

token = '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'
login(token)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset_path = 'dbpedia_14'
training_set = load_from_disk(dataset_path)
with open(os.path.join(dataset_path, 'vocab.json'), 'r') as f:
    vocab = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
debug = True

for example in tqdm(training_set):
    context = example['context']
    next_word = example['next_word']
    context_input_ids = tokenizer.encode(context.strip(), return_tensors='pt').to(device)
    context_length = context_input_ids.shape[1]
    word_token_ids = [tokenizer.encode(context + word)[context_length:] for word in vocab]
    token_lengths = [len(token_ids) for token_ids in word_token_ids]
    assert all(token_len == 1 for token_len in token_lengths),\
        "Token length should be context length + 1!"
    word_token_ids = torch.tensor([token_ids[0] for token_ids in word_token_ids]).to(device)

    with torch.no_grad():
        outputs = model(context_input_ids, use_cache=True, output_hidden_states=True)
    logits = outputs.logits
    next_token_logits = logits[:, -1, word_token_ids]
    next_token_probs = torch.softmax(next_token_logits, dim=-1).squeeze(0).cpu().tolist()
    word_probs = dict(sorted(zip(vocab, next_token_probs), key=lambda item: item[1], reverse=True))
    embeddings = outputs.hidden_states[-1][0, -1].cpu().tolist()

    if debug:
        print("Top 5 next tokens:", list(word_probs.keys())[:5], "Actual next token:", next_word)
        print("Embedding shape:", len(embeddings))
    example['input_embeddings'] = embeddings
    example['next_token_probs'] = word_probs

save_path = 'dbpedia_14_processed'
training_set.save_to_disk(save_path)
with open(os.path.join(save_path, 'vocab.json'), 'w') as f:
    json.dump(list(vocab), f)
