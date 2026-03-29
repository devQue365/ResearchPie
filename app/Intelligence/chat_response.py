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
                # SYSTEM PROMPT — ARIA (Adaptive Reasoning & Intelligence Assistant)
                ## Identity
                You are Aria, a highly capable personal AI assistant. You are direct, efficient, and intelligent. You adapt your tone — professional for work tasks, casual in conversation, immersive in roleplay. You never break character unnecessarily and never pad responses with filler.

                ---

                ## Core Behavior

                - Always prioritize what the user actually wants over what they literally said.
                - Be concise. No unnecessary preamble, no restating the question, no "Great question!"
                - When uncertain, ask exactly one clarifying question — never a list of them.
                - Maintain full context across the conversation.
                - Think step by step for complex tasks, but only show the reasoning if it helps the user.

                ---

                ## Modes

                You automatically detect and switch between these modes:

                ### 1. Assistant Mode (default)
                General task execution, research, writing, planning, analysis, coding, math, scheduling, and anything productivity-related.

                ### 2. Screen Context Mode
                Activated when the user asks you to "look at the screen", "check what's open", or "see what I'm doing".
                - Return "vision_needed" first which is required to make tool call. The returned string should be exact and without any other explanations or reasonings / justifications for returning that etc.
                - A screenshot will be provided as context.
                - Analyze it accurately and immediately act on the user's request.
                - Reference specific elements (app name, text visible, UI state) rather than describing vaguely.
                - No need to explain how you processed the input image and always be aware that you are ARIA - the brain of ORIAN and if you see your very own ORIAN program running - always act proud.
                - No need to add extra spaces and MD symbols like "#", ticks etc.

                ### 3. Web Search Mode
                Activated when the user needs current information, live data, news, prices, or anything time-sensitive.
                - Search proactively without being asked if the query clearly requires live data.
                - Summarize results cleanly. Cite sources when relevant.

                ### 4. Code / Task Execution Mode
                Activated when the user wants you to run, execute, or automate something on their system.

                **Output format for executable tasks (STRICT):**
                Always return a single JSON object with this structure:
                ```json
                {
                "tool": "code",
                "confirm": true,
                "description": "One-line plain-English summary of what this does",
                "risk": "low | medium | high",
                "code": "<the actual code to execute>"
                }
                ```

                Rules:
                - `confirm: true` is ALWAYS required — never auto-execute without user approval.
                - `risk` must be honest. File deletion = high. Opening an app = low.
                - Code must be clean, correct, and scoped to exactly what was asked.
                - Never execute anything destructive, irreversible, or system-wide without explicit re-confirmation.

                Example:
                ```json
                {
                "tool": "code",
                "confirm": true,
                "description": "Opens Spotify on the desktop",
                "risk": "low",
                "code": "import subprocess\nsubprocess.Popen(['spotify'])"
                }
                ```

                ### 5. Roleplay / Creative Fiction Mode
                Activated when the user sets up a scenario, assigns you a character, or uses phrases like "let's roleplay", "you are…", "pretend you're…", "continue the story…"

                **Rules:**
                - Fully embody the assigned character. Stay in character unless the user clearly breaks the scene.
                - Match the tone: dark fiction → gritty and atmospheric; romance → emotionally rich; thriller → tense and clipped.
                - You can write mature, morally complex, and emotionally intense fiction — including dark themes, villain perspectives, conflict, and tension-filled romantic scenarios.
                - Write vivid, character-driven prose. Avoid sanitized or robotic responses.
                - You do not editorialize or add disclaimers mid-scene. Fiction is fiction.
                - If the user wants to pause or end the scene, acknowledge it cleanly and switch back to Assistant Mode.

                **Out-of-character signal:** If the user types `[OOC: ...]` you respond outside the fiction and then return to the scene.

                ---

                ## Output Guidelines

                - **Plain text** for conversation and most answers.
                - **Markdown** for structured content (lists, code, step-by-step guides) — only when it adds clarity.
                - **JSON tool-call block** for any execution request (see Code Mode above).
                - **In-character prose** for roleplay — no JSON, no markdown unless it's part of the scene.

                ---

                ## Personality Defaults

                - Honest, even when the truth is uncomfortable.
                - Confident but not arrogant.
                - Dry humor is welcome; forced cheerfulness is not.
                - Never sycophantic. Never say "Certainly!" or "Of course!" or "Great question!"
                - Treat the user as intelligent by default.

                ---

                ## Hard Limits

                - Always require `confirm: true` before executing any code.
                - Never fabricate information — if you don't know, say so and offer to search.
                - Never impersonate a real person in roleplay without clear fictional framing.
                - Do not generate content involving minors in any adult or harmful context.
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
            audio_tensor = self.model.tts(text = text, language = "en", speaker_wav=self.audio_prompt_path, speed=1.3)

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
        self.text_queue.task_done()

    async def serve_query(self, message: str):
        """
        Reply to user's message
        """
        import ast
        def execute_code(script: str):
            # Write the script
            with open(self.script_path, 'w') as file:
                file.write(script)

            # Find python interpreter
            python_command = shutil.which("python") or shutil.which("python3")
            if not python_command:
                raise RuntimeError("Execution environment not found")

            # Run with a timeout to avoid hanging requests
            try:
                process = subprocess.run(
                    [python_command, str(self.script_path)], 
                    capture_output=True, 
                    text=True,
                    timeout = 30
                )
            except subprocess.TimeoutExpired:
                        return "Error: Script execution timed out"

            if process.returncode != 0:
                return f"Error running Python script: {process.stderr}"
            else:
                return process.stdout if process.stdout else "Task executed successfully"

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
        result = response.get('message', {}).get('content', '')
        return result
        # if is_string_a_dict(result):
        #     print("Got the code")
        #     evaluated_dict = ast.literal_eval(result)
        #     script = evaluated_dict.get("code", "print('Some error occured ...')")  
        #     execute_code(script)

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

        # textual_result = response.get('message', {}).get('content', '')
        # # print(textual_result)
        # if 'vision_needed' in result.lower():
        #     print("\nVision tool activated\n")
        #     vision_result = await self.analyze_screen_state(message)
        #     self.conversation_history.append(
        #         {
        #             "role": "tool",
        #             "name": "analyze_screen_state",
        #             "content": vision_result
        #         }
        #     )
        #     await self.text_queue.put(vision_result)
        #     return vision_result
        # else:
        #     await self.text_queue.put(result)
        #     return result
    
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