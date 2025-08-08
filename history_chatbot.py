from gc import collect
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
from langsmith import traceable
import gradio as gr
import os
import time
import uuid

load_dotenv()
LANGCHAIN_TRACING_V2 = True

DB_DIR = "chroma_db"
COLLECTION_NAME = "historical_figures"
print("LangSmith API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("LangSmith Tracing V2:", os.getenv("LANGCHAIN_TRACING_V2"))

# Load and split the document
loader = PyPDFLoader("historical_figures.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)
docs = text_splitter.split_documents(document)
print(f"Total number of chunks: {len(docs)}")

# Embedding and Vector Store (optimized)
embedding = OllamaEmbeddings(model="granite-embedding:latest")

if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    print("Creating and persisting new vector store...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )
    vector_store.persist()
else:
    print("Loading existing vector store...")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )

# Prompt template with context and chat history
prompt_template = """
You are HistoryBot, an expert in historical figures.
Answer the user's question using only the context provided.
If you don't know the answer, just say you don't know. Don't make things up.

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)

# LLM
llm = Ollama(model="gemma:2b")

# Store to keep chat histories per session
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@traceable
def chat_historybot(user_input, session_id):
    if not user_input.strip():
        return "Please enter a question."

    history = get_session_history(session_id)
    history.add_user_message(user_input)

    # Build chat history text for prompt
    chat_history_text = ""
    for msg in history.messages:
        role = msg.type  # 'human' or 'ai'
        prefix = "User:" if role == "human" else "Bot:"
        chat_history_text += f"{prefix} {msg.content}\n"

    # Retrieve relevant docs from vector store
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(user_input)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    # Format the full prompt text
    prompt_text = PROMPT.format(
        chat_history=chat_history_text,
        context=context_text,
        question=user_input
    )

    start = time.time()
    answer = llm(prompt_text)
    print(f"⏱️ Response time: {time.time() - start:.2f} seconds")

    history.add_ai_message(answer)

    return answer

with gr.Blocks(css="body { background-color: #D6EF88 !important; }") as demo:
    gr.Markdown("### Hello, I am HistoryBot. How can I assist you today?")
    chatbot = gr.Chatbot()
    session_id = gr.State(str(uuid.uuid4()))
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Ask me anything about historical figures...")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear History")

    def respond(message, chat_history, session_id):
        answer = chat_historybot(message, session_id)
        chat_history.append((message, answer))
        return "", chat_history, session_id

    def clear(session_id):
        if session_id in store:
            store[session_id].clear()
        return [], "", session_id

    submit_btn.click(respond, [txt, chatbot, session_id], [txt, chatbot, session_id])
    clear_btn.click(fn=clear, inputs=[session_id], outputs=[chatbot, txt, session_id])

if __name__ == "__main__":
    demo.launch()