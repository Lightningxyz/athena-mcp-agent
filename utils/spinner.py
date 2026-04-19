import sys
import time
import threading

class Spinner:
    def __init__(self, message="Thinking..."):
        self.message = message
        self.running = False
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        i = 0
        while self.running:
            # Add padding to clear previous longer messages
            sys.stdout.write(f"\r\033[96m{self.spinner_chars[i]}\033[0m {self.message:<60}")
            sys.stdout.flush()
            time.sleep(0.08)
            i = (i + 1) % len(self.spinner_chars)

    def __enter__(self):
        self.running = True
        self.thread.start()
        return self

    def update(self, new_message):
        self.message = new_message

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
