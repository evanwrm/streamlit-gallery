# Streamlit Huggingface Pipeline Tests

This project tests out the features of [Streamlit](https://streamlit.io/), specifically we create a recommender system for Steam games using two datasets from [Kaggle](https://www.kaggle.com/), a [list of games](https://www.kaggle.com/nikdavis/steam-store-games) and their categories, and a [dataset of reviews](https://www.kaggle.com/najzeko/steam-reviews-2021). From this we'll apply a number of recsys models using [Microsoft recommenders](https://github.com/microsoft/recommenders).

## How to run

First install [Poetry](https://python-poetry.org/), and then install dependencies using

```sh
poetry install
```

Next, you'll need to install a version of [PyTorch](https://pytorch.org/get-started/locally). e.g. on Linux you can use

```sh
pip install pip3 install torch torchvision torchaudio transformers
```

Now we can run streamlit using

```sh
streamlit run streamlit_app.py
```

## Results

For a brief overview of some interesting results you can get
