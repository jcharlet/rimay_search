---
title: Rimay Search
emoji: 🧘
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

rimay_search
==============================

search over Rimay teachings using natural language


TODO
------------

### Data preparation
  - [x] scrap website
  - [x] clean data (balanced articles in text length)

### Build search engine
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
  - [ ] [HIGH] break down texts longer than 1200 tokens so that we can get further details from chapter 1-2
  à quoi servent les sciences contemplatives? -> This model's maximum context length is 4097 tokens, however you requested 8818 tokens (8562 in your prompt; 256 for the completion).
- [ ] FIX CLI CLICK not working with unit tests

### Build UI with streamlit
  - [X] build it with streamlit
    - [X] introduction to service
    - [X] input text to run query
    - [X] show response
      - [X] model answer
      - [X] links to sources
      - [X] metadata
      - [X] query debug info and cost
    - [ ] additional 
      - [X] format cost display
      - [X] add openapi token, hidden as password
      - [X] improve UI: admin vs normal yogi user (hide sidebar, json, token metadata details)
        - [X] Rewrite interface in French  
      - [ ] select language: French and English - https://blog.devgenius.io/how-to-build-a-multi-language-dashboard-with-streamlit-9bc087dd4243
      - [ ] response length

### Deploy app
- [X] manage db on distant server 
  - [X] explore hugging face spaces
  - [X] share data folder on git repo? 
  - [ ] or enable to run embedding on server? (probably not)
- [X] deploy on hugging face spaces

### Evaluate, share
- [ ] share and evaluate with karma ling
- [ ] write article 

Next steps
------------

- [ ] Réduire le coût d'utilisation de l'application / la rendre plus rapide
   - [ ] utiliser un modèle plus petit (chatgpt ?)
     - [ ] changer de modèle
     - [ ] Fournir des exemples dédiés pour le modèle de lecteur dans l'invite, adaptés à la longueur du texte demandé (en utilisant la FAQ sur la pleine conscience ouverte)
   - [ ] Réduire la taille des documents
     - [ ] corrige le chapitre 1-2
     - [ ] Explorer la réduction des pages : décomposer les pages en paragraphes + inclure le résumé de la page / les métadonnées ?

- [ ] Évaluer les réponses de l'application
   - [ ] recueillir les questions/réponses
   - [ ] collecte de commentaires sur la configuration
     - [ ] bonne / mauvaise réponse
     - [ ] meilleure réponse manuelle - pour affiner plus tard
   - [ ] testez le modèle sur chacune de ces questions, évaluez les résultats, corrigez les résultats

- [ ] Améliorer la qualité des réponses
   - [ ] former le modèle de lecteur sur les contenus de pleine conscience ouverte ?
   - [ ] testez différentes tailles d'articles (plus petits)

- [ ] Industrialiser l'application
   - [ ] Déploiement sur serveur dédié
   - [ ] collecte des journaux de configuration (stacktrace)
   - [ ] À DÉFINIR

- [ ] Étendre la base de données de connaissances
   - [ ] wiki bouddha -> voir avec Tchamé = 1 semaine
   - [ ] questions réponses pendant les enseignements
   - [ ] sources externes
     - [ ] pages dédiées au bouddhisme/méditation sur wikipedia
     - [ ] recherche Google

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
