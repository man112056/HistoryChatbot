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

# --- 1. Setup & Data Loading ---
load_dotenv()
DB_DIR = "chroma_db"
COLLECTION_NAME = "historical_figures"

# Load and split the PDF
loader = PyPDFLoader("historical_figures.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Initialize Vector Store
embedding = OllamaEmbeddings(model="granite-embedding:latest")
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory=DB_DIR,
    collection_name=COLLECTION_NAME,
)

# Add documents to store if they aren't already there
if not vector_store.get()['ids']:
    vector_store.add_documents(docs)

retriever = vector_store.as_retriever()

# --- 2. LLM & Chain Definition ---
llm = OllamaLLM(model="gemma:2b")

prompt_template = """
You are HistoryBot, an expert in historical figures. 
Answer the user's question using ONLY the context provided. If the answer isn't in the context, say you don't know.

Conversation History: 
{chat_history}

Context: 
{context}

Question: 
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["chat_history", "context", "question"]
)

# Build a simple callable chain
qa_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | PROMPT
    | llm
)

# --- 3. History Management ---
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@traceable
def chat_historybot(user_input, session_id):
    history = get_session_history(session_id)
    
    # Format existing history for prompt
    chat_history_text = "\n".join(
        [f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" 
         for m in history.messages]
    )
    
    # Run the chain
    answer = qa_chain.invoke({
        "question": user_input, 
        "chat_history": chat_history_text
    })
    
    # Save to memory
    history.add_user_message(user_input)
    history.add_ai_message(answer)
    
    return answer

# --- 4. Gradio UI ---
with gr.Blocks(title="History Chatbot") as demo:
    gr.Markdown("# 🏛️ History Chatbot")
    gr.Markdown("Ask me questions about the historical figures in your document.")
    
    chatbot = gr.Chatbot(label="History Assistant")
    session_id = gr.State(str(uuid.uuid4()))
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question here and press Enter...", 
            show_label=False,
            scale=4
        )
        clear = gr.Button("Clear Chat", scale=1)

    def respond(message, chat_history, sess_id):
        if not message.strip():
            return "", chat_history

        bot_message = chat_historybot(message, sess_id)

        # Use Gradio v6+ dict format
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})

        return "", chat_history

    msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot])
    clear.click(lambda: (None, []), None, [chatbot], queue=False)

if __name__ == "__main__":
    print("Starting History Chatbot...")
    demo.launch()
