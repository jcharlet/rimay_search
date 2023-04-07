import requests
from bs4 import BeautifulSoup
import pandas as pd
import click
import logging
from pathlib import Path

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# FIXME I should scrap the pages and save them locally once and for all, and then run soup parser on the local files

def scrap_page(url):
    """
    Scrap a website page from openmindfulness.net and return the json document containing all contents in a structured format
    """
    # Make a GET request to the website
    response = requests.get(url)

    page_chapter = url.split("/")[-2]

    page_doc = {'url': url, 'page_chapter': page_chapter}
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")
    
    page_doc['page_title'] = extract_page_title(soup)
    
    documents = extract_documents_from_tag_type(soup, [], "h2")
    if documents == []:
        documents = extract_documents_from_tag_type(soup, documents, "h3")
        if documents == []:
            documents = extract_documents_from_wrapper_div(soup, documents)
    page_doc['documents'] = documents
    return page_doc

def extract_documents_from_wrapper_div(soup, documents: list[dict]):
    for element in soup.find_all("div",class_="wpb_wrapper"):
        contents = ""
        if element.name == "div":
            continue
        else:
            contents.append(element.text)
        documents.append({"contents": contents})
    return documents

def extract_page_title(soup):
    try:
        return soup.find_all("h2",class_="vcex-heading")[0].text
    except Exception as e:
        logger.error("No page title found",exc_info=e)
        return ""

def extract_documents_from_tag_type(soup, documents: list[dict], title_tag: str):
    # Find all title tags on the page
    for title_element in soup.find_all(title_tag):
        # Get the title text from the tag
        title = title_element.text

        # if element has any custom class, skip it
        if title_element.get("class"):
            continue

        # Initialize an empty string to hold the contents
        contents = ""

        # iterate over all elements that immediately follow the current tag
        for sibling in title_element.next_siblings:
            if sibling.name in ["p","h3", "h4", "h5","ul", "ol", "table"]:
                # Add the paragraph text to the string variable
                contents += sibling.text
            elif sibling == "\n":
                # Add the line break to the string variable
                contents += sibling
            # Check if the next sibling is the current title tag
            elif sibling.name == title_tag:
                break
            elif sibling.name == "div":
                # break all for loops
                break
            else:
                logger.warn(f"unmanaged sibling: {sibling.name}, {sibling.text}")
                break

        # # Print the title and contents
        # print(f"""## {title}
        # {contents}
        
        # """)
        documents.append({"title": title, "contents": contents})
        
    return documents


def collect_toc_links():
    """
    Collect all links from the Table of Contents of the website
    """
    url="https://www.openmindfulness.net/table-des-matieres/"

    links = []

    # Make a GET request to the website
    response = requests.get(url)

    page_chapter = url.split("/")[-2]

    page_doc = {'url': url, 'page_chapter': page_chapter}
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")


    # go through all links in the page
    for link in soup.find_all('a'):
        # get the link url
        link_url = link.get('href')
        if link_url == "https://www.openmindfulness.net/":
            continue
        elif link_url in ["https://www.openmindfulness.net/suivre-lentrainement-online/", 'https://www.openmindfulness.net/contact/', 'https://www.openmindfulness.net/les-seminaires-retraites-online-et-residentiel/']:
            continue
        # check if the link is a page from the website
        elif link_url.startswith("https://www.openmindfulness.net/"):
            links.append(link_url)
        # else:
        #     print("WARNING: link not from openmindfulness.net", link_url)

    # return list after removing duplicates
    return list(set(links))

def scrap_website():
    # sourcery skip: for-append-to-extend, inline-immediately-returned-variable, list-comprehension
    """
    Scrap the website openmindfulness.net and return a list of json documents containing all contents in a structured format
    """
    # collect all links from the Table of Contents of the website
    links = collect_toc_links()

    # scrap each page
    documents = []
    # counter = 0 # for debugging purposes
    for link in links:
        # if counter==10:
            # break
        documents.append(scrap_page(link))
        # counter+=1


    return documents


@click.command()
@click.argument('output_filepath', type=click.Path(), default="data/interim/openmindfulness_contents.csv")
def scrap_website_and_store_contents(output_filepath):
    """ Runs data processing scripts to scrap data from openmindfulness website and store them into
        a json file (saved in ../interim).
    """
    logger.info('scraping openmindfulness website')
    documents = scrap_website()
    logger.info('saving scraped data into json file')
    df = pd.DataFrame(documents).explode('documents').reset_index()
    df = pd.concat([df,pd.json_normalize(df['documents'])],axis=1)
    df.to_csv(output_filepath)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    scrap_website_and_store_contents()
