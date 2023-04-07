from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

with open("data/external/state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

client_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/home/jeremie/Documents/workspace/rimay_search/data/.chromadb",
    # Optional, defaults to .chromadb/ in the current directory
)


embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(
    texts,
    embeddings,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
    client_settings=client_settings,
)
# Running Chroma using direct local API.
# Using DuckDB in-memory for database. Data will be transient.


chain = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(temperature=0),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
)
response = chain(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)
print(response)
# {'answer': ' The president honored Justice Breyer for his service and
# mentioned his legacy of excellence.\n',
#  'sources': '31-pl'}Best
