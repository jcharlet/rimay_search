import requests
from bs4 import BeautifulSoup
from src.data.scrap_website import scrap_page

# write function to test scrap_website function in src/data/scrap_website.py checking outputs for a given url
def test_scrap_page():
    url = "https://www.openmindfulness.net/1-introduction-e1/"
    # call scrap_page to get the output
    extracted_data = scrap_page(url)
    
    # check that the output is a dict
    assert isinstance(extracted_data, dict)
    
    assert "1-introduction-e1" in extracted_data["page_chapter"]
    
    assert isinstance(extracted_data["documents"], list)
    assert len(extracted_data["documents"]) > 0
    assert extracted_data["documents"][0]["title"] == "1. Mindfulness : Plénitude de l’instant présent"
    
    assert extracted_data["documents"][0]["paragraphs"].startswith("\nLa mindfulness* ou\xa0pleine présence* est d’abord un état, l’état de présence* à l’instant")