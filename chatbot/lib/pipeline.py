from transformers import pipeline


CONVERSATIONAL_MODELS = [
    "facebook/blenderbot-400M-distill",
    "facebook/blenderbot-3B",
    "microsoft/DialoGPT-large",
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-small",
]
AUTOMATIC_SPEECH_RECOGNITION_MODELS = [
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec-large-robust-ft-libri-960h",
    "facebook/wav2vec-large-robust-ft-swbd-300h",
    "facebook/hubert-large-ls960-ft",
    "facebook/hubert-xlarge-ls960-ft",
]


def conversational_pipeline(model=CONVERSATIONAL_MODELS[0]):
    conv_pipeline = pipeline("conversational", model=model)

    return conv_pipeline


def automatic_speech_recognition_pipeline(model=AUTOMATIC_SPEECH_RECOGNITION_MODELS[0]):
    asr_pipeline = pipeline("automatic-speech-recognition", model=model)

    return asr_pipeline
