from flask import Flask, render_template, request, jsonify
import os
import time
import mysql.connector  # <-- Added for MySQL
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

# --- Gemini API Configuration ---
# Make sure to set your GEMINI_API_KEY in your .env file or as an environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your environment.")
genai.configure(api_key=gemini_api_key)

# === MySQL Database Config ===
db_config = {
    "host": "localhost",
    "user": "root",  # Change to your MySQL username
    "password": "",  # Change to your MySQL password
    "database": "shiv_rag_db"
}

# === MySQL Save Function ===
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
    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")

# === Load and split documents ===
folder_path = "Shivaji_Maharaj_rag_data"
docs = []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, file), encoding="utf-8")
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
    search_kwargs={"k": 3}
)

# Test retrieval
# test_query = "Who was Shivaji Maharaj?"
# retrieved_docs = retriever.get_relevant_documents(test_query)
# if retrieved_docs:
#     print(f"🔍 Retrieval test for '{test_query}':")
#     print(retrieved_docs[0].page_content[:200])
# else:
#     print("❌ Retrieval failed. No relevant documents found.")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === Gemini LLM Integration ===
# Initialize the Gemini Flash model
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

def invoke_gemini(prompt):
    """
    Function to invoke the Gemini model and get the response.
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"
    
# gemini_test = invoke_gemini("Say hello in one short sentence.")
# if "error" in gemini_test.lower():
#     print("❌ Gemini API test failed:", gemini_test)
# else:
#     print("✅ Gemini API working. Sample output:", gemini_test)
def build_prompt(inputs):
    """
    Builds a clear and instructional prompt for the Gemini model.
    """
    prompt = (
        "You are a helpful assistant. Please answer the following question based only on the provided context.\n\n"
        "## Context:\n"
        f"{inputs['context']}\n\n"
        "## Question:\n"
        f"{inputs['question']}\n\n"
        "## Answer:"
    )
    return prompt

# === LangChain RAG Chain with Gemini ===
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# The final chain now uses the Gemini model
chain = parallel_chain | RunnableLambda(build_prompt) | RunnableLambda(invoke_gemini)

# # Test full RAG pipeline
# rag_test = chain.invoke(test_query)
# print(f"🤖 RAG pipeline test output for '{test_query}': {rag_test[:200]}")

@app.route("/", methods=["GET", "POST"])
def handle_requests():
    if request.method == "GET":
        return render_template("index2.html")

    if request.method == "POST":
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please ask a valid question."})

        start_time = time.time()
        result = chain.invoke(question)
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        # === Save question & answer to MySQL ===
        save_chat_to_db(question, result, inference_time)

        return jsonify({
            "answer": result,
            "inference_time_sec": f"{inference_time:.2f}"
        })

if __name__ == "__main__":
    app.run(debug=True)
