import os
import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS
from dotenv import load_dotenv
from scipy.signal import resample

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

        # Streaming
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.samples_per_chunk
        )
        self.stream.start() # Start the stream
    
    
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
        sentence = self.normalize_for_tts(sentence)
        audio = self.model.tts(text = sentence, language = "en", speaker_wav=self.audio_prompt_path)
        audio = np.asarray(audio, dtype=np.float32)
        audio = self.trim_silence(audio)
        audio = self.speed_up(audio, factor = 1.12)
        return audio

    def speak(self, sentence: str):
        # Generate audio for the sentence
        audio = self.generate_audio(sentence)
        for chunk in self.chunk_audio(audio):
            self.stream.write(chunk.reshape(-1, 1))

    # Some util functions
    def normalize_for_tts(self, text: str) -> str:
        """
        Normalize the text for efficient TTS
        """
        import re
        # Replace whitespaces
        text = text.replace("\n", " ").replace("\t", " ")
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove spaces before punctuation
        text = re.sub(r"\s+([?,.!])", r"\1", text)

        return text.strip()
    
    def speed_up(self, audio: np.ndarray, factor: float = 1.08):
        """
        High quality time compression using resampling.
        
        :param self: Description
        :param audio: Description
        :type audio: np.ndarray
        :param factor: Description
        :type factor: float

        factor > 1.08 -> faster speech\n
        factor < 1.08 -> slower speech
        """
        new_length = int(len(audio) / factor)
        return resample(audio, new_length).astype(np.float32)
    
    def trim_silence(self, audio, threshold=0.005, padding = 200):
        """
        Trim silence but keep safety padding (in samples)
        """
        mask = np.abs(audio) > threshold
        if not np.any(mask):
            return audio

        start = max(0, np.argmax(mask) - padding)
        end = min(len(audio), len(mask) - np.argmax(mask[::-1]) + padding)
        return audio[start:end]

    def pause_for_sentence(self,sentence: str):
        if sentence.endswith("?"):
            return 0.12
        if sentence.endswith("!"):
            return 0.10
        if sentence.endswith("."):
            return 0.08
        return 0.05
    
    def close(self):
        self.stream.stop()
        self.stream.close()