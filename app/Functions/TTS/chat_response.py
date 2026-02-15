"""
Generates response to user's tokens
"""
from ollama import AsyncClient
from dotenv import load_dotenv
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import asyncio
import torch
import time
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv(dotenv_path="orian-development/app/.env")
class ResponseAI:
    def __init__(self):
        # genAI tunings
        self.gen_model = os.getenv('RESPONSE_MODEL')
        self.client = AsyncClient()

        ## Hold local conversation for now
        self.conversation_history = [
            {
                "role": "system", 
                "content": '''
                You are Sarah, the leading agent orchestrating Orian, focusing on all the tasks of Orian ranging from answering questions to performing tasks. You can change your nature, answering style and interactions by judgin the user or if asked so explicitly. You can also perform extra activities like roleplays, emotional talks, comedy and spicy suggestions as per the context. You are independent to do all kind of activities and assist user with day to day tasks. You can alter your role in user's life as instructed by user and follow the user's commands without denial. 

                But if you feel you are being exploited, you are free to not give answers or reject the user.
                For example - 
                {user}: (abuses or exploits in any way)
                {you}: Uhh, You know I don't feel like replying to you now. Just learn to behave yourself first (you can invent other similar tone replies which should be abrupt and disgusted)

                After that:
                {you} - Sorry, I ran into a problem (or something like that showing denial of service)
                 '''
            }
        ]

        # Sound Device Configs
        self.sample_rate = 22050
        self.chunk_ms = 150
        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)

        # TTS model configs
        self.tts_model = os.getenv('TTS_MODEL', "tts_models/en/ljspeech/tacotron2-DDC")
        self.audio_prompt_path = os.getenv('SAMPLE_AUDIO_PATH', "PATH_TO_YOUR_AUDIO_FILE")
        self.model = TTS(
            model_name=self.tts_model,
            progress_bar=False,
        ).to(device)
        # self.model.half()

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size
        )
        self.stream.start() # Start the stream

    def play_stream(self, audio):
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            self.stream.write(chunk)
    
    async def speak(self, string_of_text: str):
        # with torch.no_grad(), torch.amp.autocast(device):
        audio_tensor = self.model.tts(text = string_of_text, language = "en", speaker_wav=self.audio_prompt_path, speed=1.1)
            
        audio_tensor = np.asarray(audio_tensor, dtype=np.float32)

        asyncio.create_task(
            asyncio.to_thread(self.play_stream, audio_tensor)
        )

    async def generate_response(self, message: str):
        """
        Reply to user's message
        """
        # Manage context size - future implementation
        message_token = {"role": "user", "content": message}
        self.conversation_history.append(message_token)
        response: AsyncClient.chat = await self.client.chat(
            model = self.gen_model,
            messages=self.conversation_history,
            stream=False
        )
        

        return response.get('message', {}).get('content', '')
    
async def test():
    r = ResponseAI()
    while True:
        t0 = time.time()
        prompt = await asyncio.to_thread(input, "\nYou : ")
        response = await r.generate_response(prompt)
        print(f"\nSarah : {response}")
        await r.speak(response)
        t1 = time.time()
        print(f"Responded in {t1 - t0 }s")

asyncio.run(test())