import os
import sys
import pyaudio
import wave
from enum import Enum
from dotenv import load_dotenv
from ..logs.settings import logger


# Load the environment variables
load_dotenv()


class Color(Enum):
    """
    Define basic color schemes
    """
    SUCCESS = "\033[1m\033[32m"
    ERROR = "\033[1m\033[31m"
    WARNING = "\033[1m\033[33m"
    INFO = "\033[1m\033[35m"
    DEFAULT = "\033[0m"

    def __str__(self):
        return self.value

class VoiceRecorder(object):
    """
    Record audio for cloning
    """
    def __init__(self, format = pyaudio.paInt16, channels = 1, rate = 44100, chunk = 1024, output_file = "recorded_audio"):
        """
        - format - Signifies the integer size we are using (here 16 bit integers) to represent audio data samples. => Determines quality and precision of audio
        - Channels - Defines whether the audio is mono (one channel) or stereo (two channels).
        - rate - The speed at which the audio is captured. Higher sample rate means higher quality audio but also more data.
        - chunk - This determines how much audio data we capture in one iteration. Higher chunks can improve efficiency but might introduce latency.
        """
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.output_file = output_file + ".wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format = self.format, channels = self.channels, rate = self.rate, input = True,  frames_per_buffer = self.chunk)
        self.frames = [] # Store the audio data into frames array
        

    def record(self):
        """
        Record input audio data
        """
        logger.info(f"{Color.SUCCESS}Recording started ...{Color.DEFAULT}")
        try:
            while True:
                # Capture the audio data in manageable chunks
                data = self.stream.read(self.chunk)
                self.frames.append(data)

        except KeyboardInterrupt:
            # Ctrl + F2
            logger.warning(f"{Color.WARNING}Encountered a keyboard interrupt. Stopping recording ...{Color.DEFAULT}")
        
        except Exception as e:
            logger.exception(f"{Color.ERROR}Exception - {str(e)}{Color.DEFAULT}")

        # Save the file
        self.save_audio()
        logger.info(f"{Color.SUCCESS} Successfully saved the audio file as \"{self.output_file}\"!{Color.DEFAULT}")

        # Perform Cleanup
        self.perform_cleanup()

    def save_audio(self):
        """
        Save audio recording (.wav file)
        """
        os.makedirs("audio_samples", exist_ok=True)
        with wave.open("app/audio_samples/" + self.output_file, mode='wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

    def perform_cleanup(self):
        """
        Performs cleanup
        """
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

recorder = VoiceRecorder()
recorder.record()

