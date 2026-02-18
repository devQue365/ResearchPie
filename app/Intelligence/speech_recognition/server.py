from flask import Flask, request, jsonify
from flask_cors import CORS
from queue import Queue
import threading

app = Flask(__name__)
CORS(app)  

speech_queue = Queue()

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    user_text = data.get("message")

    print("Received from browser:", user_text)

    speech_queue.put(user_text)
    return jsonify({"status": "received"})

def speech_input(prompt=""):
    print(prompt, end="", flush=True)
    return speech_queue.get()

def main_program():
    while True:
        user_text = speech_input("You said: ")
        print(user_text)

threading.Thread(target=main_program, daemon=True).start()

if __name__ == "__main__":
    app.run(port=5000)
