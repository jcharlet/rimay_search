import os
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
from typing import Any, Dict, List
import click
from langchain.callbacks import get_openai_callback
from enum import Enum, unique


load_dotenv(find_dotenv())

COL_STATE_OF_THE_UNION = "state_of_the_union"
COL_OPEN_MINDFULNESS = "openmindfulness_contents"

project_dir = Path(__file__).resolve().parents[2]

persist_directory = f"{project_dir}/data/.chromadb"
client_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass

embeddings = None
def get_embeddings():  # sourcery skip: raise-specific-error
    global embeddings
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None or openai_api_key == "":
        logger.warn("No OpenAI API key found")
        raise Exception("No OpenAI API key found")
    else:
        try:
            if embeddings is None:
                embeddings = OpenAIEmbeddings()
        except Exception:
            embeddings = OpenAIEmbeddings()

    return embeddings

def _embed_dataset(
    collection_name: str,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> Chroma:
    """Embed dataset with OpenAI embeddings and Chroma DB.

    Args:
        input_filepath (str): Filepath of the dataset to embed.
        collection_name (str): Name of the Chroma DB collection to create.
        texts (List[str]): List of texts to embed.
        metadatas (List[Dict[str, Any]]): List of metadata dictionaries.

    Returns:
        Chroma: Chroma DB collection.
    """
    logger.info(
        f"Embedding dataset with OpenAI embeddings on collection {collection_name}"
    )

    # Delete existing collection.
    try:
        collection = Chroma(
            client_settings=client_settings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        collection.delete_collection()
        collection.persist()
        logger.info(f"Deleted collection {collection_name}")
    except Exception as e:
        logger.error(
            f"Error while deleting collection {collection_name}", e, stack_info=True
        )

    with get_openai_callback() as cb:
        # Create new collection.
        collection = Chroma.from_texts(
            texts,
            get_embeddings(),
            metadatas=metadatas,
            ids=ids,
            client_settings=client_settings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        collection.persist()
        logger.info(f"Created collection {collection_name}")

        metadata = {
            "cost": {
                "Total Cost (USD)": cb.total_cost,
                "Successful Requests": cb.successful_requests,
            },
            "tokens": {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
            },
        }
        logger.info(metadata)

    return collection


# @cli.command()
# @click.argument('collection_name', type=str)
def embed_dataset(collection_name: str):
    assert collection_name in [COL_STATE_OF_THE_UNION, COL_OPEN_MINDFULNESS]
    if collection_name == COL_STATE_OF_THE_UNION:
        state_of_the_union = Path(
            f"{project_dir}/data/external/state_of_the_union.txt"
        ).read_text()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(state_of_the_union)
        metadata = [{"source": f"{i}-pl"} for i in range(len(texts))]
        ids = [f"{i}-pl" for i in range(len(texts))]
        collection_name = COL_STATE_OF_THE_UNION

    elif collection_name == COL_OPEN_MINDFULNESS:
        df = pd.read_csv(f"{project_dir}/data/processed/openmindfulness_contents.csv")
        # df = df[(df.sort_chapter == 3) & (df.sort_step_nb == 5)]
        df["source"] = df["id"]
        texts = df.contents_to_embed.values.tolist()
        metadata = eval(
            df[
                [
                    "sort_chapter",
                    "sort_step_nb",
                    "sort_section_nb",
                    "sort_paragraph_nb",
                    "page_title",
                    "contents_to_embed_length",
                    "url",
                    "source",
                ]
            ].to_json(orient="records")
        )
        ids = df.id.values.tolist()

    return _embed_dataset(collection_name, texts, metadata, ids)


def get_doc_by_id(id: str, collection_name: str = COL_OPEN_MINDFULNESS) -> dict:
    """Get document by id from embedded collection.

    Args:
        id (str): document id

    Returns:
        dict: if existing, dict document of the given id.
    """
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        client_settings=client_settings,
    )
    results = collection._collection.get(id)
    if results["ids"] == []:
        return None
    doc = {
        "id": results["ids"][0],
        "document": results["documents"][0],
        "metadata": results["metadatas"][0],
    }
    if "url" in doc["metadata"].keys():
        doc["metadata"]["url"] = doc["metadata"]["url"].replace("\\", "")
    return doc


def run_similarity_search(
    query: str, collection_name: str = COL_OPEN_MINDFULNESS
) -> list[Document]:
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
        embedding_function=get_embeddings(),
        client_settings=client_settings,
    )

    response = collection.similarity_search(query, k=4)
    logger.info(response)
    return response


@unique
class ResponseSize(int, Enum):
    SMALL = 256
    MEDIUM = 512
    LARGE = 1024


def get_tokens_limits(response_size: ResponseSize) -> tuple[int, int]:
    if response_size == ResponseSize.SMALL:
        max_response_tokens = 256
        max_tokens_limit_for_chain = 3375
    elif response_size == ResponseSize.MEDIUM:
        max_response_tokens = 512
        max_tokens_limit_for_chain = 3375 - 256 - 100
    elif response_size == ResponseSize.LARGE:
        max_response_tokens = 1024
        max_tokens_limit_for_chain = 3375 - 768 - 100
    return max_response_tokens, max_tokens_limit_for_chain


# FIXME function does not work in unit test with response_size and collection_name
# arguments: raise TypeError(f"unexpected keyword argument {k}") from click and
# couldn't solve that - disabling click command for now
# @cli.command()
# @click.argument('query', type=str)
# @click.argument('response_size', type=str, default=ResponseSize.SMALL.name)
# @click.argument('collection_name', type=str, default=COL_OPENMINDFULNESS)
def run_query_with_qa_with_sources(
    query: str,
    response_size: ResponseSize = ResponseSize.SMALL,
    collection_name: str = COL_OPEN_MINDFULNESS,
) -> Dict[str, Any]:
    """run query with openai QA

    Args:
        query (str): the question to ask

        response_size (str, optional): the size of the response. Defaults to SMALL

        collection_name (str, optional): chromadb collection name to query

    Returns:
        Dict[str, Any]: {"answer": str, "sources": str}
    """
    logger.info(f"Run query {query} on {collection_name} with openai QA")

    docsearch = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        client_settings=client_settings,
    )
    max_response_tokens, max_tokens_limit_for_chain = get_tokens_limits(response_size)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0, max_tokens=max_response_tokens),  # default 256
        # chain_type="stuff", # cannot work with large documents
        chain_type="map_reduce",  # runs a lot of queries under the hood..
        # chain_type="map_rerank", # does not provide sources
        retriever=docsearch.as_retriever(),
        # verbose=True,
        reduce_k_below_max_tokens=True,  # default False
        max_tokens_limit=max_tokens_limit_for_chain,
        # maximum total context length for openai request is 4097 tokens. default 3375
    )

    with get_openai_callback() as cb:
        response = chain(
            {"question": query},
            return_only_outputs=True,
        )

        answer = response["answer"]

        sources = list(sorted(response["sources"].split(", ")))
        sources = [
            get_doc_by_id(source, collection_name=collection_name) for source in sources
        ]
        metadata = {
            "cost": {
                "Total Cost (USD)": cb.total_cost,
                "Successful Requests": cb.successful_requests,
            },
            "tokens": {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
            },
        }
        logger.info(metadata)

    logger.info(response)
    return answer, sources, metadata


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.getLogger("chromadb").setLevel(logging.WARN)
    logging.getLogger("clickhouse_connect").setLevel(logging.WARN)
    cli()
