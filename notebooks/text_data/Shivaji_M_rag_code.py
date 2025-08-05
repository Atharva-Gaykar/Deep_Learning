from flask import Flask, render_template, request, jsonify
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
import torch
from pynvml import *

app = Flask(__name__)
load_dotenv()

# === Load and split documents ===
folder_path = "Shivaji_Maharaj_rag_data"
docs = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
split_docs = text_splitter.split_documents(docs)

# === Embeddings and FAISS ===
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question_k="which fort did shivaji capture at the age of 16?"
retrieved_docs = retriever.invoke(question_k)
formatted_context = format_docs(retrieved_docs)

print("\n==== Retrieved Document Info ====")
for doc in retrieved_docs:
     print(f"[SOURCE FILE]: {doc.metadata.get('source', 'N/A')}")
     print(f"[CONTENT]: {doc.page_content[:300]}...\n") 
# === Load LoRA fine-tuned FLAN-T5 model ===
base_model_name = "google/flan-t5-base"
lora_model_path = "rag_bot/shiv_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.to(device)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_length=200
)
llm = HuggingFacePipeline(pipeline=pipe)

def build_prompt(inputs):
    return f"Context: {inputs['context']}\n\nQuestion: {inputs['question']}\nAnswer:"

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
chain = parallel_chain | RunnableLambda(build_prompt) | llm

@app.route("/", methods=["GET", "POST"])
def handle_requests():
    if request.method == "GET":
        return render_template("index2.html")

    if request.method == "POST":
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please ask a valid question."})

        # Monitor GPU memory
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        start_time = time.time()
        result = chain.invoke(question)
        end_time = time.time()

        mem_info = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        return jsonify({
            "answer": result,
            "gpu_memory_mb": f"{mem_info.used / 1024**2:.2f}",
            "inference_time_sec": f"{end_time - start_time:.2f}"
        })

if __name__ == "__main__":
    app.run(debug=True)
