import pytest
import logging
from bs4 import BeautifulSoup
from src.models.predict_model import run_similarity_search, run_query_with_qa_with_sources, COL_STATE_OF_THE_UNION, COL_OPENMINDFULNESS
import pandas as pd
from pathlib import Path

project_path = str(Path(__file__).resolve().parents[1])

@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

def test_similarity_search_on_state_of_the_union():
    # given a dataset embedded
    # collection = embed_dataset_state_of_the_union()
    
    # when I search for a paragraph with similarity search
    response = run_similarity_search("What did the president say about Justice Breyer", collection_name=COL_STATE_OF_THE_UNION)
    
    # Then I expect the correct paragraph to be returned
    top_doc = response[0]
    assert top_doc.metadata['source'] == '31-pl', "expected source found"

def test_run_query_with_qa_with_sources():
    # given a dataset embedded
    # collection = embed_dataset_state_of_the_union()
    
    # and given a query
    question = "What did the president say about Justice Breyer"
    
    # when I ask a question
    response = run_query_with_qa_with_sources(question, collection_name=COL_STATE_OF_THE_UNION)
    
    # then I expect the correct answer to be returned, using the right source
    assert response['sources'] == '31-pl', "expected source found"
    
    # cannot make an exact text comparison because the answer can vary - however it should like this:
    # assert response['answer'] == 'The president honored Justice Breyer for his service and mentioned his legacy of excellence.\n', "expected answer to our question"

