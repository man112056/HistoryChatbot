from gc import collect
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader("historical_figures.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,)

docs = text_splitter.split_documents(document)
print(f"Total number of chunks: {len(docs)}")
print(document[0].page_content[:500]) 

embedding = OllamaEmbeddings(model="granite-embedding:latest")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="historical_figures",
    persist_directory="chroma_db")
vector_store.persist()
print("Chroma vectorstore created and persisted.")

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
    template=prompt_template, input_variables=["context", "question"]
)

# Load your local LLM (like llama3 or mistral)
llm = Ollama(model="gemma3")

# Setup RetrievalQA with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = vector_store.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

query = "Who was the first President of India?"
result = qa_chain.invoke(query)

print("Answer:", result['result'])
print("\nSources:")
for doc in result['source_documents']:
    print(doc.metadata['source'])