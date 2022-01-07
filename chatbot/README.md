# Streamlit Huggingface Pipeline Tests

This project tests out the features of [Streamlit](https://streamlit.io/), specifically the aim is to test out [Huggingface Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines). We'll implement a standard conversational pipeline with input from an automatic speech recognition pipeline. Various models can be chosen for each task, and a spectogram view of the input audio can be toggled. Output can also synthesized for a text-to-speech component, a fully conversational AI!

## How to run

First install [Poetry](https://python-poetry.org/), and then install dependencies using

```sh
poetry install
```

Next, you'll need to install a version of [PyTorch](https://pytorch.org/get-started/locally), and the [Huggingface transformers library](https://huggingface.co/docs/transformers/master/en/quicktour#getting-started-on-a-task-with-a-pipeline). e.g. on Linux you can use

```sh
pip install pip3 install torch torchvision torchaudio transformers
```

Now we can run streamlit using

```sh
streamlit run streamlit_app.py
```

## Results

For a brief overview of some interesting results you can get

![Chatbot Preview](assets/chatbot-demo.png)
