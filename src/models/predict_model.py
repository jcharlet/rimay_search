from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional
import click

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


@click.group()
def cli():
    pass

def embed_dataset_state_of_the_union(
    input_filepath: str =f"{project_dir}/data/external/state_of_the_union.txt",
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


@cli.command()
@click.argument('input_filepath', type=str)
@click.argument('collection_name', type=str)
def embed_dataset(
    input_filepath: str =f"{project_dir}/data/processed/openmindfulness_contents.csv",
    collection_name: str =COL_OPENMINDFULNESS,
) -> Chroma:
    """embed dataset with openai embeddings and Chroma DB

    Args:
        input_filepath (str, optional): filepath of dataset to embed. Defaults to f"{project_dir}/data/processed/openmindfulness_contents.csv".
        collection_name (str, optional): chromadb collection name. Defaults to COL_OPENMINDFULNESS.

    Returns:
        Chroma: Chroma DB collection
    """
    logger.info(
        f"Embedding dataset with openai embeddings on collection {collection_name}"
    )
    df = pd.read_csv(input_filepath)
    df = df[(df.sort_chapter == 3) & (df.sort_step_nb == 5)]
    df["source"] = df["id"]

    metadatas = eval(
        df[
            [
                "sort_chapter",
                "sort_step_nb",
                "sort_section_nb",
                "sort_paragraph_nb",
                "page_title",
                "contents_to_embed_length",
                "contents_to_embed",
                "url",
                "source",
            ]
        ].to_json(orient="records")
    )

    collection = Chroma.from_texts(
        df.contents_to_embed.values.tolist(),
        embeddings,
        metadatas=metadatas,
        ids=df.id.values.tolist(),
        client_settings=client_settings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    collection.persist()

    return collection


def run_similarity_search(query: str, collection_name: str =COL_OPENMINDFULNESS) -> list[Document]:
    """run similarity search with openai embeddings

    Args:
        query (str): the query to run
        collection_name (str): chromadb collection name to query

    Returns:
        List[Document]: List of documents most similar to the query text.
    """
    logger.info(
        f"Embedding dataset with openai embeddings on collection {collection_name}"
    )
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    response = collection.similarity_search(query, k=4)
    logger.info(response)
    return response


@cli.command()
@click.argument('query', type=str)
@click.argument('collection_name', type=str)
def run_query_with_qa_with_sources(query: str, collection_name: str = COL_OPENMINDFULNESS) -> Dict[str, Any]:
    """ run query with openai QA

    Args:
        query (str): the question to ask
        collection_name (str, optional): chromadb collection name to query

    Returns:
        Dict[str, Any]: {"answer": str, "sources": str}
    """
    logger.info(f"Run query {query} on {collection_name} with openai QA")

    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0),
        # chain_type="stuff", # cannot work with large documents
        chain_type="map_reduce",  # runs a lot of queries under the hood..
        # chain_type="map_rerank", # does not provide sources
        retriever=docsearch.as_retriever(),
        # verbose=True,
    )
    response = chain(
        {"question": query},
        return_only_outputs=True,
    )
    logger.info(response)
    return response

if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.getLogger("chromadb").setLevel(logging.WARN)
    logging.getLogger("clickhouse_connect").setLevel(logging.WARN)
    cli()