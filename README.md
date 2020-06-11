#  Final Project Ironhack Data Bootcamp: Meme sentiment detection using Machine Learning
========================================================================================
Karla Vizcarra
<img src="./reports/figures/haters2.gif" />
*Data Part Time Barcelona Dic 2019*

## Content
- [Project Description](#project)
- [Dataset](#Dataset)
- [Workflow](#workflow)
- [Results](#results)

<a name="project"></a>

## Project Description

### Main Goal
The goal is to predict whether a meme is hateful or non-hateful. This is a binary classification problem with multimodal input data consisting of the the meme image itself (the image mode) and a string representing the text in the meme image (the text mode).

Given a meme id, meme image file, and a string representing the text in the meme image, the trained model should output the probability that the meme is hateful.

<a name="Datasets"></a>

## Dataset
The Hateful Memes Challenge is a dataset and benchmark created by Facebook AI to drive and measure progress on multimodal reasoning and understanding. In this model we will compare the multimodal and the unimodal aproach and the effectivity in the multimodal approach.

The input data contains the following files:

license.txt - the data set license
README.md - the data set readme
img/ - the directory of PNG images
train.jsonl - the training set
dev.jsonl - the development set
test.jsonl - the test set

*Image is a compilation of assets, including ©Getty Image.*

All the information about the dataset and the challenge can be found in: https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set

### Meme sentiment definition

Take an image, add some text: you've got a meme. Internet memes are often harmless and sometimes hilarious. However, by using certain types of images, text, or combinations of each of these data modalities, the seemingly non-hateful meme becomes a multimodal type of hate speech, a hateful meme.

At the massive scale of the internet, the task of detecting multimodal hate is both extremely important and particularly difficult. As the illustrative memes above show, relying on just text or just images to determine whether or not a meme is hateful is insufficient.

The owner of the data (Facebook) team defines hate speech as:

A direct or indirect attack on people based on characteristics, including ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, and disability or disease. We define attack as violent or dehumanizing (comparing people to non-human things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking hate crime is also considered hate speech.

Project Organization
------------

    ├── LICENSE
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
