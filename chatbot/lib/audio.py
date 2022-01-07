import numpy as np
from scipy import signal
from typing import List, Union

import pydub
import streamlit as st
import plotly.graph_objects as go
from streamlit_webrtc.receive import FrameT


SAMPLING_RATE = 16_000
MAX_FRAME_THRESHOLD = 25000  # ms
MIN_FRAME_THRESHOLD = 300  # ms
DEAD_TIME_THRESHOLD = 750  # ms
ACTIVITY_TIME_THRESHOLD = 200  # ms
DEAD_TIME_NOISE_THRESHOLD = 500
KEEP_FRAME_NOISE_THRESHOLD = 100


def audio_segment_from_frame(audio_frames: List[FrameT]) -> Union[np.ndarray, None]:
    sound_chunk = pydub.AudioSegment.empty()
    for audio_frame in audio_frames:
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        sound_chunk += sound

    return sound_chunk


def buffer_from_segment(audio_segment: pydub.AudioSegment):
    if len(audio_segment) > 0:
        return np.array(
            audio_segment.set_channels(1)
            .set_frame_rate(SAMPLING_RATE)
            .get_array_of_samples()
        ).astype(np.float64)

    return None


def get_spectogram_plot(buffer: np.ndarray):
    frequencies, times, spectrogram = signal.spectrogram(buffer, SAMPLING_RATE)

    trace = [
        go.Heatmap(
            x=times,
            y=frequencies,
            z=10 * np.log10(spectrogram),
            colorscale="Jet",
        )
    ]
    layout = go.Layout(
        title="Spectrogram",
        yaxis=dict(title="Frequency (Hz)"),
        xaxis=dict(title="Time (s)"),
    )
    fig = go.Figure(data=trace, layout=layout)

    return fig


def audio_sample():
    return buffer_from_segment(pydub.AudioSegment.from_wav("test.wav"))
