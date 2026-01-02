import os
import uuid
from operator import itemgetter
import gradio as gr
from dotenv import load_dotenv
from langsmith import traceable

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Setup & Data Loading ---
load_dotenv()
DB_DIR = "chroma_db"
COLLECTION_NAME = "historical_figures"

# Load and split the PDF
loader = PyPDFLoader("historical_figures.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents)

print(f"📄 PDF pages loaded: {len(documents)}")
print(f"✂️ Chunks created: {len(docs)}")

for i, d in enumerate(docs[:3]):
    print(f"\n--- Chunk {i+1} preview ---")
    print(d.page_content[:300])

# --- Vector Store (SAFE INIT) ---
embedding = OllamaEmbeddings(model="granite-embedding:latest")

# IMPORTANT: create collection cleanly if DB was deleted
if not os.path.exists(DB_DIR):
    print("📦 Creating new Chroma database...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )
else:
    print("📦 Loading existing Chroma database...")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    # ✅ FIX: Reset collection and re-add docs to ensure new chunks are indexed
    print("🧹 Resetting existing Chroma collection...")
    vector_store.reset_collection()
    vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("📊 Total documents in Chroma:", vector_store._collection.count())


# --- 2. LLM & Chain Definition ---
llm = OllamaLLM(model="gemma:2b")

prompt_template = """
You are HistoryBot, an expert in historical figures.
Answer the user's question using ONLY the context provided.
If the answer is not present, say you don't know.

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"],
)

qa_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | PROMPT
    | llm
)

# --- 3. Chat History ---
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@traceable
def chat_historybot(user_input, session_id):
    history = get_session_history(session_id)

    chat_history_text = "\n".join(
        [
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in history.messages
        ]
    )
    
    # --- DEBUG: Check what Chroma retrieves --
    docs_retrieved = vector_store.similarity_search(user_input, k=3)
    print(f"\n🔍 Query: {user_input}")
    for i, doc in enumerate(docs_retrieved):
        print(f"📄 Chunk {i+1} Preview: {doc.page_content[:200]}")  # first 200 chars

    answer = qa_chain.invoke(
        {
            "question": user_input,
            "chat_history": chat_history_text,
        }
    )

    history.add_user_message(user_input)
    history.add_ai_message(answer)

    return answer

# --- 4. Gradio UI ---
with gr.Blocks(title="History Chatbot") as demo:
    gr.Markdown("# 🏛️ History Chatbot")
    gr.Markdown("Ask questions based on the provided historical document.")

    chatbot = gr.Chatbot(label="History Assistant")
    session_id = gr.State(str(uuid.uuid4()))

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question and press Enter...",
            show_label=False,
            scale=4,
        )
        clear = gr.Button("Clear Chat", scale=1)

    def respond(message, history, sid):
        if not message.strip():
            return "", history

        reply = chat_historybot(message, sid)

        # ✅ REQUIRED messages format (prevents crash)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        return "", history

    msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

# --- 5. Run ---
if __name__ == "__main__":
    print("🚀 Starting History Chatbot...")
    demo.launch()
