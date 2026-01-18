import os
import torch
import nltk
import numpy as np
import sounddevice as sd
from TTS.api import TTS
from dotenv import load_dotenv

nltk.download('punkt')
load_dotenv(dotenv_path="orian-development/app/.env")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Text_To_Speech:
    def __init__(self):
        self.sample_rate = eval(os.getenv('SAMPLE_RATE', 22050))
        self.chunk_ms = eval(os.getenv('CHUNK_MS', 40))
        self.samples_per_chunk = self.sample_rate * self.chunk_ms // 1000
        self.tts_model = os.getenv('TTS_MODEL', "tts_models/en/ljspeech/tacotron2-DDC")
        self.audio_prompt_path = os.getenv('SAMPLE_AUDIO_PATH', None)
        self.model = TTS(
            model_name=self.tts_model,
            progress_bar=False,
        ).to(device)
    
    def split_sentences(self, text: str):
        """
        NLTK based sentence splitter
        """
        return nltk.sent_tokenize(text)
    
    def chunk_audio(self, audio: np.ndarray):
        """
        Yield small chunks for real time illusion
        """
        for i in range(0, len(audio), self.samples_per_chunk):
            yield audio[i:i + self.samples_per_chunk]
    
    def generate_audio(self, sentence: str)->np.ndarray:
        """
        Docstring for generate_audio
        
        :param sentence: Generate audio for a single sentence
        :type sentence: str
        :return: np.ndarray
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        audio = self.model.tts(text = sentence, language = "en", speaker_wav=self.audio_prompt_path)
        return np.asarray(audio, dtype=np.float32)

    def speak(self, text: str):

        # Split the text to sentences
        sentences = self.split_sentences(text)

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.samples_per_chunk
        ) as stream:
            for sentence in sentences:
                # Generate audio for each sentence
                audio = self.generate_audio(sentence)

                for chunk in self.chunk_audio(audio):
                    stream.write(chunk.reshape(-1, 1))
