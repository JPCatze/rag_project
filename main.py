import pyarrow.parquet as pq
import chromadb

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag import query,rerank

app = Flask(__name__)

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_or_create_collection(name="test_collection")

table = pq.read_table("data/wiki_text.parquet", columns=["text"])

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype="auto",
    device_map='cuda:0',
    load_in_4bit=True,
)

def get_document(data):
    zip_dict = query(data)
    return zip_dict

def get_topdoc(query,data):
    top_doc = rerank(query,data)
    return top_doc

def fetch_text(dict):
    list_doc = [f"{k}:{table['text'][v].as_py()}" for k, v in dict.items()]
    return list_doc


@app.route("/chat")
def main():

    inputs = request.args.get("query")

    if not inputs:
        return jsonify({"error": "Missing conversation parameter"}), 400
    
    zip_dict = get_document(inputs)
    list_doc = fetch_text(zip_dict)
    top_doc = get_topdoc(inputs,list_doc)

    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use the provided context to answer the user's question."
    },
    {
        "role": "user",
        "content": f"Context:\n{top_doc}\n\nQuestion:\n{inputs}"
    }
]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7, 
        top_p=0.8
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content  = tokenizer.decode(output_ids)

    return jsonify({"response": content})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000, debug=False)
