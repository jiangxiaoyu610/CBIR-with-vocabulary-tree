import sys


class Logger:
    def __init__(self, log_path):
        self.logger = open(log_path, 'wb', buffering=0)
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message + '\n')
        self.logger.write(message.encode('utf-8') + b"\n")
        self.logger.flush()

    def close(self):
        self.logger.close()