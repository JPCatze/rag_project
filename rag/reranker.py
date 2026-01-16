import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def format_instruction(instruction, query, doc):
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, 
        padding=False, truncation='longest_first',
        return_attention_mask=False, 
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens

    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
    
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

#batch processing
@torch.no_grad()
def compute_logits(inputs, batch_size=4, **kwargs):  
    all_scores = []
    num_samples = inputs['input_ids'].size(0)
    
    for i in range(0, num_samples, batch_size):
    
        batch_inputs = {
            key: value[i:i+batch_size] 
            for key, value in inputs.items()
        }
        
        batch_logits = model(**batch_inputs).logits[:, -1, :]
        true_vector = batch_logits[:, token_true_id]
        false_vector = batch_logits[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        
        all_scores.extend(scores)
    
    return all_scores

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B",device_map='cuda:0',dtype=torch.float16).eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
task = 'Given a web search query, retrieve relevant passages that answer the query'

def rerank(query,documents):

    query = query
    documents = documents

    pairs = [format_instruction(task, query, doc) for doc in documents]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    zip_score = list(zip(documents,scores))
    zip_score.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in zip_score[:3]]

    return top_docs
