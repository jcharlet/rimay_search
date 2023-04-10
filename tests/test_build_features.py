import requests
import pytest
import logging
from bs4 import BeautifulSoup
from src.features.build_features import balance_articles_length, add_features
import pandas as pd
from pathlib import Path

project_path = str(Path(__file__).resolve().parents[1])

import pytest
@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

def test_add_features():
    # given input df
    df = pd.read_csv(f'{project_path}/data/interim/openmindfulness_contents.csv')
    
    # when I call add_features
    df = add_features(df)
    
    # then I expect all articles have a chapter number, a step number, a section number and a paragraph number where relevant
    assert (df[df.sort_chapter.isin([2,3,4])].sort_chapter.notnull()).all(), "All articles in chapters 2 to 4 should have a chapter number"
    assert (df[df.sort_chapter==3].sort_step_nb.notnull()).all(), "All articles in chapter 3 should have a step number"
    assert (df[df.sort_chapter.isin([2,3,4])].sort_section_nb.notnull()).all(), "All articles in chapters 2 to 4 should have a section number"
    assert (df.sort_paragraph_nb.notnull()).all(), "All articles should have a paragraph number"

def test_balance_articles_length():
    # given input df with added features
    df = pd.read_csv(f'{project_path}/data/interim/openmindfulness_contents.csv')
    df = add_features(df)
    
    # and given the current total contents length
    total_content_length = df.title.fillna("").apply(lambda x: len(x.split())).sum() + df.contents.apply(lambda x: len(x.split())).sum()

    # when I call balance_articles_length
    df = balance_articles_length(df)
    
    # then I expect no contents were lost
    assert total_content_length == df.contents_to_embed_length.sum(), "No contents should have been lost"
    
    # verify manually debug logs to check size of articles is improved