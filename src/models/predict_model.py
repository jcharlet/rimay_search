from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging

load_dotenv(find_dotenv())

COL_STATE_OF_THE_UNION = "state_of_the_union"
COL_OPENMINDFULNESS = "openmindfulness_contents"

project_dir = Path(__file__).resolve().parents[2]

embeddings = OpenAIEmbeddings()
persist_directory = f"{project_dir}/data/.chromadb"
client_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
)

logger = logging.getLogger(__name__)


def embed_dataset_state_of_the_union(
    input_filepath=f"{project_dir}/data/external/state_of_the_union.txt",
) -> Chroma:
    """
    embed dataset state of the union with openai embeddings and Chroma DB
    (for testing purpose)
    """
    logger.info(
        "Embedding dataset with openai embeddings "
        + "on collection {COL_STATE_OF_THE_UNION}"
    )
    state_of_the_union = Path(input_filepath).read_text()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    collection = Chroma.from_texts(
        texts,
        embeddings,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
        client_settings=client_settings,
        collection_name=COL_STATE_OF_THE_UNION,
        persist_directory=persist_directory,
    )
    collection.persist()

    return collection


def embed_dataset(
    input_filepath=f"{project_dir}/data/processed/openmindfulness_contents.csv",
    collection_name=COL_OPENMINDFULNESS,
) -> Chroma:
    logger.info(
        f"Embedding dataset with openai embeddings on collection {collection_name}"
    )
    df = pd.read_csv(input_filepath)
    df = df[(df.sort_chapter == 3) & (df.sort_step_nb == 5)]

    metadatas = df[
        [
            "sort_chapter",
            "sort_step_nb",
            "sort_section_nb",
            "sort_paragraph_nb",
            "page_title",
            "contents_to_embed_length",
            "contents_to_embed",
            "url",
        ]
    ].to_json(orient="records")

    collection = Chroma.from_texts(
        df.contents_to_embed.values.tolist(),
        embeddings,
        metadatas=metadatas,
        client_settings=client_settings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    collection.persist()

    return collection


def run_similarity_search(query: str, collection_name=COL_OPENMINDFULNESS):
    logger.info(
        f"Embedding dataset with openai embeddings on collection {collection_name}"
    )
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    return collection.similarity_search(query, k=4)


def run_query_with_qa_with_sources(query: str, collection_name=COL_OPENMINDFULNESS):
    logger.info(f"Run query {query} on {collection_name} with openai QA")

    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    return chain(
        {"question": query},
        return_only_outputs=True,
    )


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]


#     main()
