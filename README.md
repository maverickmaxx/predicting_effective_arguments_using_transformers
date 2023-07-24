Predicting Effective Arguments
==============================

MIDS w207 project
------------
Waqas Ali
Vivek Bhatnagar
Adam Childs

Getting Started
------------
After pulling, unzip the .zip file found in ./data/raw

If working locally, be sure 'LOCAL = True' is set in the notebook.  Set this False when running on Kaggle.

References
------------

Project Requirements
https://docs.google.com/document/d/1rfd54BVXDzj3awGkZU6HrT7UyTSfQs09zED-2W1oRoY/edit

Project Slides Intro
https://docs.google.com/presentation/d/1U2K5zC758AGo8IPGuEOsbO9HzYp78SHiZ298NhKo5WM/edit?usp=sharing

Feedback Prize - Predicting Effective Arguments | Kaggle. Kaggle.com. Published 2022. Accessed July 12, 2022. https://www.kaggle.com/competitions/feedback-prize-effectiveness/data‌.

argumentation_scheme_and_rubrics_kaggle.docx. argumentation_scheme_and_rubrics_kaggle.docx. Google Docs. Published 2022. Accessed July 12, 2022. https://docs.google.com/document/d/1G51Ulb0i-nKCRQSs4p4ujauy4wjAJOae/edit

Presentation on the Corpus from which the contest data is drawn (PERSUADE corpus):
https://www.youtube.com/watch?v=AETWJWL2M5Q


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

