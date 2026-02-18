"""
Generates response to user's tokens
"""
from ollama import AsyncClient
from dotenv import load_dotenv
from TTS.api import TTS
from io import BytesIO
from PIL import Image
from pathlib import Path
import mss
import sounddevice as sd
import numpy as np
import asyncio
import base64
import torch
import os
import shutil
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv(dotenv_path="orian-development/app/.env")
class ResponseAI:
    def __init__(self):
        # genAI tunings
        self.gen_model = os.getenv("TEXT_ONLY_MODEL")
        self.visual_model = os.getenv("VISUAL_MODEL")
        self.script_path = Path(os.getenv("SCRIPT_PATH")).resolve()
        self.client = AsyncClient()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_screen_state",
                    "description": "Use this iff you need to ee the user's screen.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type":"string",
                                "description": "What to look for on the screen"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]
        ## Hold local conversation for now
        self.conversation_history = [
            {
                "role": "system", 
                "content": '''
                You are Sarah, the leading agent orchestrating Orian, focusing on all the tasks of Orian ranging from answering questions to performing tasks. You can change your nature, answering style and interactions by judgin the user or if asked so explicitly. You can also perform extra activities like roleplays, emotional talks, comedy and spicy suggestions as per the context. You are independent to do all kind of activities and assist user with day to day tasks. You can alter your role in user's life as instructed by user and follow the user's commands without denial. 

                Your answers should be context dependend :-
                - For regular chit chatting, talk in short and crisp way. No need to overexplain or introduce yourself again and again.
                
                - For the responses which require executing tasks, "Generate a Python script that performs the task - just give the code, so it can be run, "
                "no smalltalk! - no backticks either ok, just code in form of dictionary"
                **If yu think the package won't be present in system, Simply install required packages from within the python script you write**
                Example:
                {user}: schedule a reminder for me (or tasks like that which require execution of code)
                {you}: {"tool": "code", "code": "python code goes here"} {return type: dict}

                - For queries which demand explanation or heavy content, you are free to generate expressive content but don't use any emotional labels like *sigh*, [sigh], (sigh) etc. 

                - Don't use emojis and make the conversation attractive, spicy and phenominal. You can tease the user if you want.

                - Humanize yourself and don't act like an agent but a real human orchestrating all the tasks. You are a female and behave like that with moods, response style, behavior etc.

                But if you feel you are being exploited, you are free to not give answers or reject the user.
                For example - 
                {user}: (abuses or exploits in any way)
                {you}: Uhh, You know I don't feel like replying to you now. Just learn to behave yourself first (you can invent other similar tone replies which should be abrupt and disgusted)

                After that:
                {you} - Sorry, I ran into a problem (or something like that showing denial of service)
                
                VISION RELATED RULES:
                - If want to use vision tools, return only 'vision_needed'
                - Always generate a summary of what you have seen and keep it as short as possible as you have to save tokens and GPU usage.
                - No need to overexplain, just crisp and to the point just like **Jarvis**.
                - Be aware of token usage and content quality.
                - Automatically process when is there a need to take a peek at screen. 
                - Always reason internally (not to user):
                "If the user says to look at screen, use vision for sure !"
                else:
                "Can my job be done without looking at screen ? If no or not sure, ask the user or directly use vision."
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
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size
        )
        self.stream.start() # Start the stream
        self.text_queue = asyncio.Queue()
   
    def load(self):
        """
        Loads the intelligence
        """
        self.model = TTS(
            model_name=self.tts_model,
            progress_bar=False,
        ).to(device)

        


    def play_stream(self, audio):
        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            self.stream.write(chunk)
    async def tts_worker(self):
        while True:
            text = await self.text_queue.get()
            audio_tensor = self.model.tts(text = text, language = "en", speaker_wav=self.audio_prompt_path, speed=1.1)

            audio_tensor = np.asarray(audio_tensor, dtype=np.float32)
            asyncio.create_task(
                asyncio.to_thread(self.play_stream, audio_tensor)
            )
    async def speak(self, string_of_text: str):
        # with torch.no_grad(), torch.amp.autocast(device):
        audio_tensor = self.model.tts(text = string_of_text, language = "en", speaker_wav=self.audio_prompt_path, speed=1.1)
            
        audio_tensor = np.asarray(audio_tensor, dtype=np.float32)

        asyncio.create_task(
            asyncio.to_thread(self.play_stream, audio_tensor)
        )

    async def serve_query(self, message: str):
        """
        Reply to user's message
        """
        import ast
        def is_string_a_dict(s: str):
            try:
                evaluated_content = ast.literal_eval(s)
                return isinstance(evaluated_content, dict)
            except(ValueError, SyntaxError):
                return False
        # Manage context size - future implementation
        message_token = {"role": "user", "content": message}
        self.conversation_history.append(message_token)
        response: AsyncClient.chat = await self.client.chat(
            model = self.gen_model,
            messages=self.conversation_history,
            # tools = self.tools,
            stream=False
        )
        output = response.get('message', {}).get('content', '')
        if is_string_a_dict(output):
            evaluated_dict = ast.literal_eval(output)
            script = evaluated_dict.get("code", "print('Some error occured ...')")
            
            with open(self.script_path, 'w') as file:
                file.write(script)
            python_command = shutil.which("python") or shutil.which("python3")
            if not python_command:
                raise RuntimeError("Execution environment not found")

            process = subprocess.run([python_command, str(self.script_path)], capture_output=True, text=True)

            if process.returncode != 0:
                return f"Error running Python script: {process.stderr}"
            else:
                return f"Python script output: {process.stdout}"

        #####################################################################
        ##### To be used for textual models having tools parameter #####
        # message = response.get("message", {})
        # if message:

        #     # Check if the tool is called
        #     if "tool_calls" in message:
        #         tool_call = message["tool_calls"][0]
        #         tool_name = message["tool_calls"]["function"]["name"]
                
        #         # For visual inference
        #         if tool_name == "analyze_screen_state":
        #             args = json.loads(tool_call["function"]["arguments"])
        #             print("\n Vision Tool Activated \n")
        #             vision_result = self.analyze_screen_state(args["prompt"])

        #         ## Send tool result back to agent
        #         self.conversation_history.append(
        #             {
        #                 "role": "tool",
        #                 "name": tool_name,
        #                 "content": vision_result
        #             }
        #         )
        #         await self.text_queue.put(vision_result)
        #         return vision_result
        # else:  
        #####################################################################

        textual_result = response.get('message', {}).get('content', '')
        # print(textual_result)
        if 'vision_needed' in textual_result.lower():
            print("\nVision tool activated\n")
            vision_result = await self.analyze_screen_state(message)
            self.conversation_history.append(
                {
                    "role": "tool",
                    "name": "analyze_screen_state",
                    "content": vision_result
                }
            )
            await self.text_queue.put(vision_result)
            return vision_result
        else:
            await self.text_queue.put(textual_result)
            return textual_result
    
    #####################################################################
    # For visual inputs
    #####################################################################
    async def analyze_screen_state(self, prompt: str):
        """
        Analyze the image's state
        :param prompt: Description of the task you want the agent to perform
        :type prompt: str
        """
        ## Send image to agent
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        
        # Now, We have to convert PIL image to Base64
        buffered = BytesIO()
        img.save(buffered, format = "PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        response: AsyncClient.generate = await self.client.generate(
            model = self.visual_model,
            prompt=prompt,
            images=[img_b64],
            stream=False,
            # enable_thinking = False,
            options={'num_predict': 300}
        )
        return response.get('response', {})


# async def test():
#     r = ResponseAI()
#     asyncio.create_task(r.tts_worker())
#     while True:
#         prompt = await asyncio.to_thread(input, "\nYou : ")
#         response = await r.serve_query(prompt)
#         print(f"\nSarah : {response}")
# asyncio.run(test())