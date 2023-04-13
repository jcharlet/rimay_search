---
title: Rimay Search
emoji: 🧘
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.17.0
app_file: src/visualization/visualize.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

rimay_search
==============================

search over Rimay teachings using natural language


TODO
------------
- [X] collect data
    - [x] scrap website
    - [x] clean data (balanced articles in text length)
- [X] build qa search
    - [x] prototype
    - [x] applied to sample
    - [x] debug
      - [x] texts too long: need samples of max 1200 chars to follow openai
      This model's maximum context length is 4097 tokens, however you requested 9779 tokens (9523 in your prompt; 256 for the completion). Please reduce your prompt; or completion length
        - [NO] either reduce text size to 1200 chars as in langchain prompt (while I have a series of texts with 600 - 1200 tokens)
        - [x] or change architecture: QA to every match and combine, rather than QA on all matches in 1 prompt: map_reduce and map_ranks
      - [x] increase response size on demand
      - [X] track cost
      - [X] get url, chapter and everything from source returned into chromadb
      - [X] add title in response
    - [X] running on whole dataset
    - [ ] make sure to respond in the expected language (French)
    - [ ] break down texts longer than 1200 tokens so that we can get further details from chapter 1-2
  - [ ] FIX CLI CLICK not working with unit tests
- [X] build UI
    - [X] build it with streamlit
      - [X] introduction to service
      - [X] input text to run query
      - [X] show response
        - [X] model answer
        - [X][HIGH] links to sources
        - [X][HIGH] metadata
        - [X] query debug info and cost
      - [ ] additional 
        - [X] format cost display
        - [X] add openapi token, hidden as password
        - [X] improve UI: admin vs normal yogi user (hide sidebar, json, token metadata details)
          - [X] Rewrite interface in French  
        - [ ] select language: French and English - https://blog.devgenius.io/how-to-build-a-multi-language-dashboard-with-streamlit-9bc087dd4243
        - [ ] response length
- [ ] [HIGH] deploy
  - [ ] [HIGH] manage db on distant server 
    - [ ] explore hugging face spaces
    - [ ] share data folder on git repo? 
    - [ ] or enable to run embedding on server? (probably not)
  - [ ] [HIGH] deploy on hugging face spaces
- [ ] share and evaluate with karma ling
- [ ] write article 

Potential next steps
------------
- [ ] Provide dedicated examples for reader model in the prompt, adapted to the text length requested (using open mindfulness FAQ)
- [ ] train reader model on open mindfulness contents?
- [ ] experiment with different sizes of articles (smaller) 
- [ ] setup feedback collection
  - [ ] good / bad answer
  - [ ] better manual response - to finetune later
- [ ] setup logs collection (stacktrace) 

Documentation
------------
- Retrieval question answering with sources based on [langchain dedicated tutorial](https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa_with_sources.html)
- Explanation of different kinds of chain (stuffing, map_reduce, refine, map_rerank), to deal with long queries to open_ai: in [langchain](https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html), but also in this [excellent video](https://www.youtube.com/watch?v=), and this [cookbook](https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook.ipynb). Here we are using map_reduce since the stuffing method, which tries to put all candidate documents in prompt to answer question, results in too long prompts for openai (over 4k) 
- modifying prompts, from [langchain](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/getting_started.html), to make sure it always responds in the requested language (when asking for long answers, it sometimes responds in English, since the prompt and examples given are in English)



Setup
------------
You will need to create a .env file and put your openai token

Warning: the tests are not mocked! So running some of those can be quickly costly - a qa search query on the state of the union costs 6cents, a qa search query on the open mindfulness contents can cost 40cents

## debug streamlit with vscode
add to launch.json
```
{
    "name": "debug streamlit",
    "type": "python",
    "request": "launch",
    "program": "/home/jeremie/anaconda3/envs/rimay_search/bin/streamlit",
    "args": [
        "run",
        "/home/jeremie/Documents/workspace/rimay_search/src/visualization/visualize.py"],
    "console": "integratedTerminal",
    "justMyCode": true
}
```
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
