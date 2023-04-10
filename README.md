rimay_search
==============================

search over Rimay teachings using natural language


TODO
------------
- [ ] collect data
    - [x] scrap website
    - [x] clean data (balanced articles in text length)
- [ ] build qa search
    - [x] prototype
    - [x] applied to sample
        - [x] debug
          - [x] texts too long: need samples of max 1200 chars to follow openai
          This model's maximum context length is 4097 tokens, however you requested 9779 tokens (9523 in your prompt; 256 for the completion). Please reduce your prompt; or completion length
            - [NO] either reduce text size to 1200 chars as in langchain prompt (while I have a series of texts with 600 - 1200 tokens)
            - [x] or change architecture: QA to every match and combine, rather than QA on all matches in 1 prompt: map_reduce and map_ranks
        - [ ] evaluation
        - [ ] iteration 1
        - [ ] iteration 2
    - [ ] running on whole dataset
        - [ ] evaluation
- [ ] build UI
    - [ ] built it with streamlit
    - [ ] deploy on hugging face spaces
- [ ] write article 

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
