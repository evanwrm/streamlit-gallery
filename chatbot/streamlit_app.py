import time
import queue
import numpy as np

import pydub
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from transformers.pipelines.conversational import Conversation

from lib.pipeline import (
    AUTOMATIC_SPEECH_RECOGNITION_MODELS,
    CONVERSATIONAL_MODELS,
    automatic_speech_recognition_pipeline,
    conversational_pipeline,
)
from lib.audio import (
    ACTIVITY_TIME_THRESHOLD,
    DEAD_TIME_NOISE_THRESHOLD,
    DEAD_TIME_THRESHOLD,
    KEEP_FRAME_NOISE_THRESHOLD,
    MAX_FRAME_THRESHOLD,
    MIN_FRAME_THRESHOLD,
    audio_segment_from_frame,
    buffer_from_segment,
    get_spectogram_plot,
)


#
# Config
#

st.set_page_config(
    page_title="Streamlit | Stock Prediction",
    initial_sidebar_state="auto",
    layout="centered",
)
# Remove footer
style_streamlit = """<style>
    footer {visibility: hidden;}
</style>"""
st.markdown(style_streamlit, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def fetch_conversational_pipeline(model):
    return conversational_pipeline()


@st.cache(allow_output_mutation=True)
def fetch_asr_pipeline(model):
    return automatic_speech_recognition_pipeline()


#
# Introduction
#

st.title("Huggingface Chat Bot")
st.write(
    """
This project tests out the features of [Streamlit](https://streamlit.io/), specifically the aim is to test out [Huggingface Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines). 
We'll implement a standard conversational pipeline with input from an automatic speech recognition pipeline using the wav2vec encoder.
"""
)

#
# Sidebar
#

st.sidebar.subheader("Models")
conv_model = st.sidebar.selectbox("Conversational Model", CONVERSATIONAL_MODELS)
asr_model = st.sidebar.selectbox(
    "Automatic Speech Recognition Model", AUTOMATIC_SPEECH_RECOGNITION_MODELS
)
st.sidebar.write("---")
st.sidebar.subheader("Settings")
is_conversational = st.sidebar.checkbox("Conversational", True)
if is_conversational:
    speech_synthesis = st.sidebar.checkbox("Speech Synthesis", True)

st.sidebar.write("")
spectrogram_view = st.sidebar.checkbox("Spectogram View", True)

# Pipelines
conv_pipe = fetch_conversational_pipeline(model=conv_model)
asr_pipe = fetch_asr_pipeline(model=asr_model)

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.state.playing:
    status_indicator = st.empty()
    status_indicator.info("Loading...")

    if spectrogram_view:
        text_output, spectrogram_output = st.columns((1, 1))
        spectrogram_output = spectrogram_output.empty()
    else:
        text_output = st.container()
    conversation = Conversation()

    running_segment = pydub.AudioSegment.empty()
    dead_time_counter = 0
    dead_break = False
    activity_time_counter = 0
    was_activity = False

    if webrtc_ctx.audio_receiver:
        status_indicator.info("Running. Say something!")

    # Note: don't make any unnecessary calls to streamlit as it rerenders all interactive plots
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.warning("No frame arrived.")
                dead_break = True
                continue

            # add frames into the running segment
            audio_segment = audio_segment_from_frame(audio_frames)

            # calculate the noise to see if speech is present in the current frame
            buffer_segment = buffer_from_segment(audio_segment)
            noise = np.sum(np.linalg.norm(buffer_segment)) / len(audio_segment)

            if noise > KEEP_FRAME_NOISE_THRESHOLD:
                running_segment += audio_segment

            if noise < DEAD_TIME_NOISE_THRESHOLD:
                dead_time_counter += len(audio_segment)
                if dead_time_counter > DEAD_TIME_THRESHOLD:
                    dead_break = True
            else:
                dead_time_counter = 0
                dead_break = False
                activity_time_counter += len(audio_segment)
                if activity_time_counter > ACTIVITY_TIME_THRESHOLD:
                    was_activity = True

            # if segment is too long just discard (can't process this)
            # TODO: attempt to cut out dead parts of audio
            if len(running_segment) > MAX_FRAME_THRESHOLD:
                running_segment.empty()

            print(noise, dead_time_counter, dead_break)

            if (
                dead_break
                and was_activity
                and len(running_segment) > MIN_FRAME_THRESHOLD
            ):
                buffer = buffer_from_segment(running_segment)

                running_segment = running_segment.empty()
                dead_time_counter = 0
                activity_counter = 0
                dead_break = False
                was_activity = False

                result = asr_pipe(buffer)
                text = result["text"].lower()

                if text:
                    if "congratulations" in text:
                        st.balloons()

                    if spectrogram_view:
                        spectrogram_output.plotly_chart(get_spectogram_plot(buffer))

                    if is_conversational:
                        text_output.write(text)
                        text_output.caption("You")

                        # Get response from conversational pipline
                        conversation.add_user_input(text)
                        conv_pipe(conversation)

                        bot_text = conversation.generated_responses[-1]
                        text_output.write(bot_text)
                        text_output.caption("Bot")
                        if speech_synthesis:
                            components.html(
                                f"""
                                    <script>
                                        speechSynthesis.speak(new SpeechSynthesisUtterance("{bot_text}"))
                                    </script>
                                """,
                                width=0,
                                height=0,
                            )

                        text_output.write("---")
                    else:
                        text_output.write(f"**Text:** {text}")
        else:
            status_indicator.exception("AudioReceiver is not set. Abort.")
            break
