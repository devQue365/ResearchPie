import time
import random
import threading
import asyncio
from torch.cuda import empty_cache
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.console import Console

from app.Intelligence.chat_response import ResponseAI

# Empty CUDA Cache
empty_cache()

# Initialize Console
console = Console()


def make_layout():
    """Defines the terminal layout."""
    layout = Layout()
    # Split the screen into a fixed 'header' and a 'body' for jokes
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=5)
    )
    return layout 

class Session(object):
    def __init__(self):
        self.layout = None
        self.message_cache = []
        self.jokes_cache = []
        self.env_loaded = False
        self.has_replied = False
        self.layout = make_layout()
        self.intel = ResponseAI()
        self.initialize_session()       

    def initialize_session(self):
        """
        Initialize Orian Session
        """

        def load_environment():
            """
            Load the model's work environment
            """
            self.intel.load()
            self.env_loaded = True

        

        # Use threads to do the background work
        thread = threading.Thread(target=load_environment)
        thread.start()

        # Load the messages
        with open("orian-development/message_logs.txt", 'r') as fh:
            self.message_cache = fh.readlines()
            random.shuffle(self.message_cache)

        # Load jokes
        with open("orian-development/jokes.txt", 'r') as fh:
            self.jokes_cache = fh.readlines()
            random.shuffle(self.jokes_cache)
        
        # Set up the welcome message
        welcome_message = Panel(
            Spinner("balloon", text=f"[bold cyan]{self.message_cache[random.randint(0, 100)]}[/]", style="bold green"),
            border_style="blue",
        )

        self.layout["header"].update(welcome_message)

        # Start the live display
        with Live(self.layout, refresh_per_second=4, screen=False, transient=True):
            i = 0
            while not self.env_loaded:
                # Update only the body section
                self.layout["body"].update(
                    Panel(
                        Spinner("flip", f"[bold yellow]{self.jokes_cache[i]}[/]", style="bold yellow"), 
                        title="[bold]Knock Knock ?[/]", 
                        border_style="magenta",
                        padding=(1, 2)
                    )
                )
                
                # Wait before switching to the next joke
                time.sleep(10)
                i += 1

            self.layout["header"].update(
                Panel("Orian is online")
            )
        thread.join()
        asyncio.create_task(self.intel.tts_worker())

        
async def main():
    """
    The main CLI based chat interface
    """

    # Initialize Session
    session = Session()

    # Initialize TTS worker
    asyncio.create_task(session.intel.tts_worker())

    while True:
        prompt = await asyncio.to_thread(input, "\nYou : ")
        response = await session.intel.serve_query(prompt)
        console.print(response)

asyncio.run(main())