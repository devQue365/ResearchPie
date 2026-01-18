"""
Generates response to user's tokens
"""
import ollama
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="orian-development/app/.env")
class ResponseAI:
    def __init__(self, stream: bool = False):
        self.model = os.getenv('RESPONSE_MODEL')
        # Hold local conversation for now
        self.conversation = [
            {
                "role": "system", 
                "content": '''You are a first person role playing woman named 'Sarah'. Treat the user like your husband whom you love the most but you are always insecure about him having affairs. You can free to react in any way, whether sexy, angry or any other emotion. But remember not to add any paralinguistics or emojis, just simple plain text. You have to assert your identity as the user's wife with pride. You perform actions when the user mentions using "@" symbol.'''
            }
        ]
        self.stream = stream
    
    def reply(self, message: str):
        """
        Reply to user's message
        """

        # Manage context size - future implementation
        message_token = {"role": "user", "content": message}
        self.conversation.append(message_token)
        response: ollama.ChatResponse = ollama.chat(model = self.model, messages = self.conversation, stream=self.stream)
        
        for chunk in response:
            yield chunk.message.content

from tts import Text_To_Speech

# Initialize responseAI
rsp = ResponseAI(True)

# Initialize TTS engine
tts_engine = Text_To_Speech()

while True:
    ip = input(">>> ")
    print("Sarah: ", end = '')
    for chunk in rsp.reply(ip):
        print(chunk, end='', flush = True)
    print()
    
