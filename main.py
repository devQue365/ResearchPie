import time
import asyncio
from app.Functions.TTS.tts import OveroTTS
from app.Functions.TTS.chat_response import ResponseAI
from torch.cuda import empty_cache

# Empty CUDA Cache
empty_cache()

class ChatOrchestrator:
    def __init__(self):
        # Producers and Consumers
        self.tts_system = OveroTTS()
        self.response_system = ResponseAI()

    async def start_session(self):
        # Start background tasks once
        consumer_task = asyncio.create_task(
            self.tts_system.audio_consumer()
        )

        tts_producer_task = asyncio.create_task(
            self.tts_system.audio_producer()
        )

        # Run indefinte sessiom
        while True:
            print('-+' * 50)
            prompt_o = await asyncio.to_thread(input, "Ask Anything : ")
            print('-+' * 50)
            if prompt_o.lower() in {'exit', 'quit', 'stop'}:
                break
            
            response_stream = self.response_system.generate_response(
                message = prompt_o
            )
            await self.tts_system.process_request(response_stream)
            print()
        
        # Clean shutdown
        self.tts_system.shutdown()

        await asyncio.gather(tts_producer_task, consumer_task)

if __name__ == '__main__':
    orchestrator = ChatOrchestrator()
    asyncio.run((orchestrator.start_session()))