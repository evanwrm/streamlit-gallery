# Streamlit Stable Diffusion Tests

This project is meant to test the [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2) models available on Hugging Face Hub. We explore the base model, and available image upscaling techniques.

## How to run

First install [Conda](https://docs.conda.io/en/latest/), or [Mamba](https://mamba.readthedocs.io/), and then install dependencies using

```sh
mamba env create -p ./.venv -f environment.yml
```

Now we can run streamlit using

```sh
streamlit run streamlit_app.py
```

## Results

Check the official Stable Diffusion [repository](https://github.com/Stability-AI/stablediffusion) for some example results.
