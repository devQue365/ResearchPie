import os
import nltk
import torch
import numpy as np
import asyncio
import sounddevice as sd
from TTS.api import TTS
from dotenv import load_dotenv

load_dotenv(dotenv_path="orian-development/app/.env")

device = "cuda" if torch.cuda.is_available() else "cpu"

nltk.download("punkt", quiet=True)

class OveroTTS:
    def __init__(self):
        # Sound Device Configs
        self.sample_rate = 22050
        self.chunk_ms = 100
        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)
        # TTS model configs
        self.tts_model = os.getenv('TTS_MODEL', "tts_models/en/ljspeech/tacotron2-DDC")
        self.audio_prompt_path = os.getenv('SAMPLE_AUDIO_PATH', "PATH_TO_YOUR_AUDIO_FILE")
        self.model = TTS(
            model_name=self.tts_model,
            progress_bar=False,
        ).to(device)

        # Streaming and Asynchronous Queues
        self.audio_q = asyncio.Queue(maxsize=10)
        self.sentence_q = asyncio.Queue()

        self.buffer_chunks = 4
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size
        )
        self.stream.start() # Start the stream
    
    async def process_request(self, token_stream):

        buffer = ""

        async for token in token_stream:
            buffer += token
            sentences = nltk.sent_tokenize(buffer)

            for sent_sentence in sentences[:-1]:
                await self.sentence_q.put(sent_sentence.strip())
        
            buffer = sentences[-1]

        if buffer.strip():
            await self.sentence_q.put(buffer.strip())

    async def audio_producer(self):
        """
        Generate and chunk audio tensor
        """

        while True:
            # Get the sentence from sentence queue
            sentence = await self.sentence_q.get()
    
            if sentence is None:
                break

            audio_tensor = self.model.tts(text = sentence, language = "en", speaker_wav=self.audio_prompt_path)
            audio_tensor = np.asarray(audio_tensor, dtype=np.float32)
            # Get the audio chunks
            chunks = [
                audio_tensor[i:i + self.chunk_size]
                for i in range(0, len(audio_tensor), self.chunk_size)
            ] 

            # Process each chunk -> Put in queue
            for chunk in chunks:
                await self.audio_q.put(chunk)
            
            # Add a small pause between sentences
            # pause = np.zeros(int(0.15 * self.sample_rate), dtype=np.float32)
            # await self.audio_q.put(pause)

    async def audio_consumer(self):
        """
        Play audio
        """

        silence = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        # Wait for buffer to load with chunks
        while(self.audio_q.qsize() < self.buffer_chunks):
            await asyncio.sleep(0.003)
        
        # Playback
        while True:
            try:
                chunk = await self.audio_q.get()
            except asyncio.QueueEmpty:
                self.stream.write(silence)
                continue
            if chunk is None:
                break
            
            self.stream.write(chunk)

        self.stream.stop()
        self.stream.close()

    async def shutdown(self):
        await self.sentence_q.put(None)
        await self.audio_q.put(None)
        