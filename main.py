import sys
import nltk
import threading
import queue
from app.Functions.TTS.tts import Text_To_Speech
from app.Functions.TTS.chat_response import ResponseAI

nltk.download("punkt", quiet=True)

tts_queue = queue.Queue()

def tts_worker(tts: Text_To_Speech):
    while True:
        sentence = tts_queue.get()

        if sentence is None:
            break
        tts.speak(sentence)
        tts_queue.task_done()

def main():

    # TTS background thread
    tts = Text_To_Speech()
    worker = threading.Thread(
        target = tts_worker,
        args = (tts, ),
        daemon = True
    )
    worker.start()

    response_system = ResponseAI(True)

    try:
        while True:
            user_input = input(">> ").strip()
            if user_input.lower() in ('exit', 'quit'):
                break

            print("Sarah : ", end = '', flush = True)
            response_buffer = ""

            for token in response_system.reply(user_input):
                sys.stdout.write(token)
                sys.stdout.flush()

                # Accumulate text in buffer
                response_buffer += token

                # Detect sentence
                sentences = nltk.sent_tokenize(response_buffer)

                if(len(sentences) > 1):
                    for sentence in sentences[:-1]:
                        tts_queue.put(sentence)
                    
                    response_buffer = sentences[-1]
            # Speak the remaining text
            if response_buffer.strip():
                tts_queue.put(response_buffer)
            
            print()
    finally:
        tts_queue.put(None)
        worker.join()
        tts.close()

if __name__ == '__main__':
    main()