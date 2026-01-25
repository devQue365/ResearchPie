import os
import torch
import numpy as np
import asyncio
import sounddevice as sd
from TTS.api import TTS
from dotenv import load_dotenv

load_dotenv(dotenv_path="orian-development/app/.env")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Text_To_Speech:
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

        # Streaming
        self.queue = asyncio.Queue(maxsize=10)
        self.buffer_chunks = 4
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size
        )
        self.stream.start() # Start the stream
    
    async def audio_producer(self, sentence: str):
        """
        Generate and chunk audio tensor
        """
        audio_tensor = self.model.tts(text = sentence, language = "en", speaker_wav=self.audio_prompt_path)
        audio_tensor = np.asarray(audio_tensor, dtype=np.float32)
        # Get the audio chunks
        chunks = [
            audio_tensor[i:i + self.chunk_size]
            for i in range(0, len(audio_tensor), self.chunk_size)
        ] 

        # process each chunk -> Put in queue
        for chunk in chunks:
            await self.queue.put(chunk)
        
        # await self.queue.put(None) # Flag "End Of Signal"

    async def audio_consumer(self):
        """
        Play audio
        """

        silence = np.zeros(int(0.1 * 5), dtype=np.float32)

        # Wait for buffer to load with chunks
        while(self.queue.qsize() < self.buffer_chunks):
            await asyncio.sleep(0.003)
        
        # Playback
        while True:
            try:
                chunk = await self.queue.get()
            except asyncio.QueueEmpty:
                self.stream.write(silence)
                continue
            if chunk is None:
                break

            self.stream.write(chunk)
            # await asyncio.sleep(self.chunk_ms / 1000)
        self.close()

    def close(self):
        self.stream.stop()
        self.stream.close()

async def main():
    tts = Text_To_Speech()
    consumer = asyncio.create_task(tts.audio_consumer())
    while True:
        ip = await asyncio.to_thread(input, "You : ")

        if ip.lower() in {'exit', 'stop'}:
            break
        
        await tts.audio_producer(ip)

    
    await tts.queue.put(None)
    await consumer

if __name__ == '__main__':
    asyncio.run(main())