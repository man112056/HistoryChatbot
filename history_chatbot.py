from gc import collect
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# Load and split the document
loader = PyPDFLoader("historical_figures.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)
docs = text_splitter.split_documents(document)
print(f"Total number of chunks: {len(docs)}")
print(document[0].page_content[:500])

# Embedding and Vector store
embedding = OllamaEmbeddings(model="granite-embedding:latest")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="historical_figures",
    persist_directory="chroma_db"
)
vector_store.persist()
print("Chroma vectorstore created and persisted.")

# Prompt
prompt_template = """
You are HistoryBot, an expert in historical figures.
Answer the user's question using only the context provided.
If you don't know the answer, just say you don't know. Don't make things up.

Context:
{context}

Question:
{question}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# LLM
llm = Ollama(model="llama3:latest")

# Memory with chat history
message_history = InMemoryChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    return_messages=True
)

# Conversational Retrieval QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Gradio Interface
def chat_historybot(user_input):
    if not user_input.strip():
        return "Please enter a question."
    result = qa_chain.invoke({"question": user_input})
    return result["answer"]

with gr.Blocks(css="body { background-color: #D6EF88 !important; }") as demo:
    gr.Markdown("### Hello, I am HistoryBot. How can I assist you today?")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Ask me anything about historical figures...")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear History")

    def respond(message, chat_history):
        answer = chat_historybot(message)
        chat_history.append((message, answer))
        return "", chat_history

    def clear():
        message_history.clear()
        return [], ""

    submit_btn.click(respond, [txt, chatbot], [txt, chatbot])
    clear_btn.click(fn=clear, outputs=[chatbot, txt])

if __name__ == "__main__":
    demo.launch()