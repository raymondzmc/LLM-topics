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
    all_input_ids = torch.cat([context_input_ids] + word_token_ids, dim=1)
    # Get the longest common prefix for caching
    longest_prefix_length = all_input_ids.size(1)
    for cache_length in range(longest_prefix_length):
        current_slice = all_input_ids[:, cache_length:cache_length + 1]
        if not torch.all(current_slice == current_slice[0]):
            break
    
    # Get the past_key_values for the cache prefix
    cache_input_ids = all_input_ids[0, :cache_length]
    with torch.no_grad():
        cache_outputs = model(cache_input_ids.unsqueeze(0), use_cache=True)
    past_key_values = cache_outputs.past_key_values
    cache_log_probs = F.log_softmax(cache_outputs.logits[:, -1], dim=-1)

    word_token_lengths = all_inputs['attention_mask'][:, cache_length:].sum(dim=1)
    single_token_word_indices = (word_token_lengths == 1).nonzero(as_tuple=True)[0]
    all_inputs['input_ids']
    vocab_next_token_probs = {}
    
    single_token_word_token_ids = all_input_ids[single_token_word_indices, cache_length]
    single_token_word_logits = cache_outputs.logits[:, -1, single_token_word_token_ids]
    single_token_word_softmax_prob = torch.softmax(single_token_word_logits, dim=-1).squeeze(0)
    for idx, prob in zip(single_token_word_indices, single_token_word_softmax_prob):
        vocab_next_token_probs[vocab[idx]] = prob
    
    word_probs = dict(sorted(zip(vocab_next_token_probs.keys(), vocab_next_token_probs.values()), key=lambda item: item[1]))
    pdb.set_trace()
    For vocab words with length > 1, we need to compute the log probabilities for each token in the word
    
    Initialize the word_log_probs dictionary
    
    for batch_start in range(0, len(vocab), batch_size):
        batch_words = vocab[batch_start:batch_start+batch_size]
        batch_input_ids = all_inputs['input_ids'][batch_start:batch_start+batch_size, cache_length:]
        batch_size_actual = len(batch_words)
        
        # Expand past_key_values to match batch size
        expanded_past_key_values = []
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value), each of shape (1, num_heads, seq_len, head_dim)
            # We need to repeat along the batch dimension to match batch_size_actual
            expanded_layer_past = tuple(p.repeat(batch_size_actual, 1, 1, 1) for p in layer_past)
            expanded_past_key_values.append(expanded_layer_past)
        
        batch_attention_mask = all_inputs['attention_mask'][batch_start:batch_start+batch_size]
        batch_position_ids = torch.tensor(list(range(cache_length,
                                                     cache_length + batch_input_ids.size(1)))).repeat(batch_size_actual, 1)
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                past_key_values=expanded_past_key_values,
                position_ids=batch_position_ids,
                use_cache=True
            )
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        # Compute log probabilities for each word
        for i, word in enumerate(batch_words):
            token_log_probs = cache_log_probs[i] + log_probs[i]
            # Concatenate the logits of the cache and the new tokens
            word_logits = torch.cat([cache_outputs.logits[:, -1], outputs.logits[i, :-1]], dim=0)
            log_probs = F.log_softmax(word_logits, dim=-1)
            word_token_ids = batch_input_ids[i][batch_input_ids[i] != tokenizer.pad_token_id]
            word_log_probs[word] = log_probs[:, word_token_ids].sum().item() / len(word_token_ids)
            
        # tokenizer.decode(model.generate(tokenizer.encode(context + 'Joe', return_tensors='pt'))[0])

    total_log_probs_tensor = torch.tensor(list(word_log_probs.values()))
    normalized_probs = torch.softmax(total_log_probs_tensor, dim=0)
    # word_probs = dict(zip(word_log_probs.keys(), normalized_probs.tolist()))
    word_probs = dict(sorted(zip(word_log_probs.keys(), normalized_probs.tolist()), key=lambda item: item[1]))

    pdb.set_trace()
    
        
