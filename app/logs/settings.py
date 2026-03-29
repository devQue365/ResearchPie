import logging
import sys
from enum import Enum


"""
Highlight Rules :-
1. Green - Info Level (Program works as intended)
2. Red - Error /Exception encountered
3. Red (Underlined) - Critical error
4. Yellow - Warnings
"""


ESC = "\x1b"
''' 
Used a Custom Formatter_
- Colorful Terminal logging
- Plain file-based logging
'''
class Color(Enum):
    """
    Logging color scheme
    """
    GREEN = f"{ESC}[0;102m"
    YELLOW = f"{ESC}[0;103m"
    RED_1 = f"{ESC}[0;101m" # High Intensity
    RED_2 = f"{ESC}[41m" # Low Intensity
    RED_UNDERLINE = f"{ESC}[4;31m" # Underlined
    DEFAULT = "\033[0m"
    def __str__(self):
        return self.value


# Create a custom color formatter
class Color_Formatter(logging.Formatter):
    """
    Custom color formatter
    """
    LEVEL_COLORS = {
        logging.INFO: Color.GREEN,
        logging.ERROR: Color.RED_2,
        logging.WARNING: Color.YELLOW,
        logging.CRITICAL: Color.RED_UNDERLINE,
    }

    def format(self, record):
        """
        Perform formatting
        """
        log_msg = super().format(record)

        # Only colorize if the output is a terminal
        if sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno)
            if color:
                return f"{color}{log_msg}{Color.DEFAULT}"
        return log_msg

# Logging Configurations
logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)

''' Console Handler '''
console_handler = logging.StreamHandler(stream = sys.stdout)

console_handler.setFormatter(Color_Formatter(
    "[%(name)s] - %(asctime)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s", datefmt = "%d/%m/%Y %I:%M:%S %p"
))

''' File Handler '''
file_handler = logging.FileHandler("./main.log")
file_handler.setFormatter(
    logging.Formatter(
        "[%(name)s] - %(asctime)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s", datefmt = "%d/%m/%Y %I:%M:%S %p"
    )
)

## Configure logger with handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
