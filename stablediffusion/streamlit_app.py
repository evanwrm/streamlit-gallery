import streamlit as st
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel

st.set_page_config(
    page_title="Streamlit | Stable Diffusion",
    initial_sidebar_state="auto",
    layout="centered",
)
# Remove footer
style_streamlit = """<style>
    footer {visibility: hidden;}
</style>"""
st.markdown(style_streamlit, unsafe_allow_html=True)

with st.sidebar:
    st.header("Model Settings")
    model_id = st.selectbox(
        "Model",
        [
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-2",
            "stabilityai/stable-diffusion-2-base",
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
        ],
    )
    allow_explicit = st.checkbox("Allow explicit content", value=False)
    image_dim_cols = st.columns(2)
    width = image_dim_cols[0].number_input("Image width", 256, 2048, 512)
    height = image_dim_cols[1].number_input("Image height", 256, 2048, 512)
    num_images = st.slider("Number of images", 1, 9, 3)
    num_inference_steps = st.slider("Number of inference steps", 10, 1000, 100, 10)
    guidance_scale = st.slider("Guidance scale", 0.0, 10.0, 7.5, 0.5)

    # Storage
    st.header("Storage")
    if not "generations" in st.session_state:
        st.session_state.generations = []
    if st.button("Clear all images", help="Clear all generated images"):
        st.session_state.generations = []


def safety_checker(images, clip_input):
    return images, False


@st.cache(allow_output_mutation=True)
def load_model(model_id: str, allow_explicit: bool):
    if "stabilityai" in model_id:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
    else:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.float16
        )
        if allow_explicit:
            pipe.safety_checker = safety_checker
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_attention_slicing()
    return pipe


def generate_image(
    model: StableDiffusionPipeline,
    prompt: str,
    neg_prompt: str = None,
    num_images=1,
    num_inference_steps=100,
    guidance_scale=7.5,
    progress_bar=True,
):
    ptext = st.empty()
    pbar = st.progress(0)

    def progress_callback(i, *_):
        pbar.progress(i / num_inference_steps)
        ptext.text(f"Generating image [{int(i / num_inference_steps * 100)}%]")

    with torch.inference_mode():
        res = model(
            [prompt] * num_images,
            negative_prompt=[neg_prompt] * num_images if neg_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback=progress_callback if progress_bar else None,
        )
        images = res.images
    return images


def display_image_grid(images, cols=3):
    rows = (num_images + cols - 1) // cols
    for row in range(rows):
        with st.container():
            image_cols = st.columns(cols)
            for col, image_col in enumerate(image_cols):
                idx = row * cols + col
                if idx >= num_images:
                    break
                image = images[idx]
                with image_col:
                    st.image(image)  # caption=prompt


st.markdown(
    """
# Stable Diffusion Demo

Welcome to the Stable Diffusion demo! Choose a model and generate images. 
[Stable Diffusion](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) 
(accompanying [code](https://github.com/Stability-AI/stablediffusion)) 
is a text-to-image model trained on various [laion](https://laion.ai/) datasets.

#### Explcit Content
Older models (v1) will censor potentially explicit images, which can be disabled by checking the 
`Allow explicit content` box. Newer models (v2) will not censor images, but are also trained to 
discourage explicit content.

#### Memory Issues
If you're having memory issues (CUDA out of memory), try reducing the number of generated images.

#### Specific styles
A common way to enhance/alter the generated images is to use textual inversion, or model checkpoints.
A popular site to find these is [civitai](https://civitai.com/), and you can also find them on
Hugging Face Hub.

#### Other Dashboards
The [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is a great tool
to generate, and explore checkpoints/embeddings.
"""
)

prompt = st.text_area("Prompt", "a photo of an astronaut riding a horse on mars")
neg_prompt = st.text_area("Negative Prompt", "")
if st.button("Generate"):
    model = load_model(model_id, allow_explicit)
    images = generate_image(
        model,
        prompt,
        neg_prompt=neg_prompt,
        num_images=num_images,
        num_inference_steps=num_inference_steps,
    )
    st.session_state.generations.append((prompt, images))
    for gen, (prompt, images) in enumerate(st.session_state.generations[::-1]):
        if gen == 0:
            st.header(prompt)
            display_image_grid(images)
        else:
            with st.expander(prompt):
                display_image_grid(images)
