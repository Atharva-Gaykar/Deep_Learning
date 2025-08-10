from flask import Flask, render_template, request, jsonify
import os
import time
import mysql.connector
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
import google.generativeai as genai
import cohere
import torch  # for device check

app = Flask(__name__)
load_dotenv()

# --- Gemini API Config ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found.")
genai.configure(api_key=gemini_api_key)

# --- Cohere API Config ---
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found.")
co = cohere.ClientV2(api_key=cohere_api_key)

# --- MySQL Config ---
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "shiv_rag_db"
}

def save_chat_to_db(question, answer, inference_time):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                inference_time_sec FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT INTO chat_logs (question, answer, inference_time_sec) VALUES (%s, %s, %s)",
            (question, answer, inference_time)
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Saved to DB:", question[:50], "...")
    except mysql.connector.Error as err:
        print(f"❌ DB Error: {err}")

# --- Load & split docs ---
folder_path = "Shivaji_Maharaj_rag_data"
docs = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file), encoding="utf-8")
        docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# --- Embeddings & FAISS ---
device_used = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Using device for embeddings: {device_used}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device_used}
)
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- Format docs with Cohere reranking ---
def format_docs(docs, query):
    # Show retrieved docs before reranking
    print("\n🔍 Retrieved documents BEFORE rerank:")
    for i, doc in enumerate(docs, 1):
        print(f"[Doc {i}] {doc.page_content[:200]}...\n")

    # Extract only text
    doc_texts = [doc.page_content for doc in docs]

    # Cohere rerank
    rerank_results = co.rerank(
        query=query,
        documents=doc_texts,
        top_n=min(2, len(doc_texts)),
        model="rerank-english-v3.0",
    )

    # Print rerank scores and text snippets
    print("\n📊 Cohere Rerank Results:")
    for idx, res in enumerate(rerank_results.results, 1):
        print(f"Rank {idx} | Score: {res.relevance_score:.4f} | Text: {doc_texts[res.index][:200]}...\n")

    # Sort docs by rerank order
    reranked_texts = [doc_texts[result.index] for result in rerank_results.results]

    final_context = "\n\n".join(reranked_texts)
    print("\n📄 Final Context Sent to Gemini:\n", final_context, "\n" + "="*80)
    return final_context

# --- LLM ---
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

def invoke_gemini(prompt):
    try:
        print("\n📝 Final Prompt Sent to Gemini:\n", prompt, "\n" + "="*80)
        response = llm.generate_content(prompt)
        print("\n🤖 Gemini Output:\n", response.text, "\n" + "="*80)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def build_prompt(inputs):
    few_shot_examples = """
Example 1:
Context:
The cause of his death is disputed, with various records suggesting bloody flux, anthrax, or fever.

Question:
What are the disputed causes of Shivaji's death?

Answer:
The disputed causes of Shivaji's death are bloody flux, anthrax, or fever.

---

Example 2:
Context:
Following his death, his childless wife, Putalabai, committed sati by jumping into his funeral pyre.

Question:
Which wife committed sati after Shivaji's death?

Answer:
Putalabai committed sati by jumping into Shivaji's funeral pyre.

---

Example 3:
Context:
After Shivaji's death, Soyarabai and several ministers conspired to crown her son Rajaram. On April 21, 1680, ten-year-old Rajaram was installed on the throne.

Question:
Who was initially installed on the throne after Shivaji's death?

Answer:
Ten-year-old Rajaram was initially installed on the throne on April 21, 1680.
"""
    prompt = (
        "You are a helpful assistant. Always answer based only on the provided context. "
        "If the context does not contain enough information, say 'I don't have enough information from the provided context.'\n\n"
        "Here are some examples of how you should answer:\n"
        f"{few_shot_examples}\n\n"
        "Now, use the same style to answer this question:\n\n"
        "## Context:\n"
        f"{inputs['context']}\n\n"
        "## Question:\n"
        f"{inputs['question']}\n\n"
        "## Answer:"
    )
    return prompt

# --- Chain ---
def get_context_with_rerank(question):
    retrieved_docs = retriever.invoke(question)
    return format_docs(retrieved_docs, question)

final_chain = (
    RunnableParallel({
        "context": RunnableLambda(lambda q: get_context_with_rerank(q)),
        "question": RunnablePassthrough()
    })
    | RunnableLambda(build_prompt)
    | RunnableLambda(invoke_gemini)
)

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def handle_requests():
    if request.method == "GET":
        return render_template("Shivaji_bot_UI.html")

    if request.method == "POST":
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please ask a valid question."})

        start_time = time.time()
        result = final_chain.invoke(question)
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        save_chat_to_db(question, result, inference_time)

        return jsonify({
            "answer": result,
            "inference_time_sec": f"{inference_time:.2f}"
        })

if __name__ == "__main__":
    app.run(debug=True)
