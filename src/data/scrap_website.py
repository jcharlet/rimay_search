import requests
from bs4 import BeautifulSoup


def scrap_page(url):
    """
    Scrap a website page from openmindfulness.net and return the json document containing all contents in a structured format
    """
    page_doc = {}
        
    # Make a GET request to the website
    response = requests.get(url)
    
    page_chapter = url.split("/")[-2]
    
    page_doc['url'] = url
    page_doc['page_chapter'] = page_chapter 
    

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")

    documents = []
    # Find all h2 tags on the page
    for h2 in soup.find_all("h2"):
        # Get the title text from the h2 tag
        title = h2.text
        
        # if element has any custom class, skip it
        if h2.get("class"):
            continue

        # Initialize an empty string to hold the paragraphs
        paragraphs = ""
        
        # iterate over all elements that immediately follow the current h2 tag
        for sibling in h2.next_siblings:
            if sibling.name in ["p","h3", "h4"]:
                # Add the paragraph text to the string variable
                paragraphs += sibling.text
            elif sibling == "\n":
                # Add the line break to the string variable
                paragraphs += sibling
            # Check if the next sibling is an h2 tag
            elif sibling.name == "h2":
                break
            elif sibling.name == "div":
                # break all for loops
                break
            else:
                print("WARNING unmanaged sibling: ", sibling.name, sibling.text)
                break

        # Print the title and paragraphs
        print(f"""## {title}
        {paragraphs}
        
        """)
        documents.append({"title": title, "paragraphs": paragraphs})
        
    page_doc['documents'] = documents
    return page_doc
