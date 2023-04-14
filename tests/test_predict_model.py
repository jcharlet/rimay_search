import pytest
import logging
from bs4 import BeautifulSoup
from src.models.predict_model import embed_dataset, get_doc_by_id, run_similarity_search, run_query_with_qa_with_sources, COL_STATE_OF_THE_UNION, COL_OPEN_MINDFULNESS, ResponseSize
import pandas as pd
from pathlib import Path

project_path = str(Path(__file__).resolve().parents[1])

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

def test_similarity_search_on_state_of_the_union():
    # given a dataset embedded
    # embed_dataset(COL_STATE_OF_THE_UNION)
    
    # when I search for a paragraph with similarity search
    response = run_similarity_search("What did the president say about Justice Breyer", collection_name=COL_STATE_OF_THE_UNION)
    
    # Then I expect the correct paragraph to be returned
    top_doc = response[0]
    assert top_doc['metadata']['source'] == '31-pl', "expected source found"

def test_run_query_with_qa_with_sources_on_state_of_the_union():
    # given a dataset embedded
    # embed_dataset(COL_STATE_OF_THE_UNION)
    
    # and given a query
    question = "What did the president say about Justice Breyer"
    
    # when I ask a question
    answer, sources, metadata = run_query_with_qa_with_sources(question, collection_name=COL_STATE_OF_THE_UNION)
    # response = run_query_with_qa_with_sources(question, response_size=ResponseSize.LARGE, collection_name=COL_STATE_OF_THE_UNION)
    
    # then I expect the correct answer to be returned, using the right source
    assert sources is not None
    assert sources != []
    assert sources[0]['metadata']['source'] == '31-pl', "expected source found"
    
    # cannot make an exact text comparison because the answer can vary - however it should like this:
    print(answer)
    # The president said "Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service
    # The president honored Justice Breyer for his service and mentioned his legacy of excellence.\n

def test_embedding_and_qa_query():
    # When I embed a sample of our dataset
    # embed_dataset(COL_OPEN_MINDFULNESS)

    # and I ask a question
    question = "Comment intégrer ses émotions avec la méthode en trois temps ?"
    answer, sources, _ = run_query_with_qa_with_sources(question)
    # response = run_query_with_qa_with_sources(question, response_size=ResponseSize.LARGE)

    # then I expect the correct answer to be returned, using the right source
    print(answer)     # cannot make an exact text comparison because the answer can vary
    assert sources is not None
    assert sources != []
    source_ids = [source['metadata']['source'] for source in sources]
    assert '3.5.20.01' in source_ids, "expected source found"
    # assert list(sorted(sources.split(", "))) == sorted(
    #     ['3.5.20.01', '3.5.20.02', '3.5.24.01', '3.5.20.03']
    # )
    
def test_retrieve_source():
    # given a dataset embedded
    # embed_dataset(COL_OPEN_MINDFULNESS)
    
    # given a source id
    id="3.1.01.01"
    # id="31-pl"
    
    # when I retrieve the source
    doc = get_doc_by_id(id)
    # doc = get_doc_by_id(id,collection_name=COL_STATE_OF_THE_UNION)
    
    # Then I get all its contents and metadata
    assert doc is not None
    logger.info(doc)
    assert doc['metadata']['page_title'] == "PREMIERE ETAPE\nLA PRESENCE ATTENTIVE AU CORPS"
    
  