import pytest
import logging
from bs4 import BeautifulSoup
from src.models.predict_model import embed_dataset, embed_dataset_state_of_the_union, run_similarity_search, run_query_with_qa_with_sources, COL_STATE_OF_THE_UNION, COL_OPENMINDFULNESS
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

def test_run_query_with_qa_with_sources_on_state_of_the_union():
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

def test_embedding_and_qa_query():
    # When I embed a sample of our dataset
    # embed_dataset()

    # and I ask a question
    question = "Comment intégrer ses émotions avec la méthode en trois temps ?"
    response = run_query_with_qa_with_sources(question)

    # then I expect the correct answer to be returned, using the right source

    # assert reponse['sources'] equals array ['3.5.20.01, 3.5.20.02, 3.5.24.01, 3.5.20.03']
    assert list(sorted(response['sources'].split(", "))) == sorted(
        ['3.5.20.01', '3.5.20.02', '3.5.24.01', '3.5.20.03']
    )
    
    