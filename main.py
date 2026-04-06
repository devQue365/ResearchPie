import json
import sys
import os
from PyQt5.QtWidgets import QApplication

from app_main import MainWindow
from dotenv import load_dotenv
# Import resources to make icons available
import static.resource_rc


load_dotenv(dotenv_path="./app/.env")

if __name__ == '__main__':
    flag = os.path.exists(os.getenv("DB_PATH"))
    print(flag)

    if not flag:
        os.mkdir(os.getenv("DB_PATH"))
        with open(os.getenv("DB_PATH") + "/data.json", "w") as f:
            f.write(json.dumps([]))

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())