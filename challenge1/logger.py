import sys
import os
import time

class Logger(object):
    def __init__(self, filename="mylog.log", mode='w'):

        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.filename = filename
        self.file = open(filename, mode=mode)
        self.time = time.time() - 5
        sys.stdout = self
        sys.stderr = self
        self.previous_message = ''

    def open(self):
        self.file = open(self.filename, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        if message != '\n':
            self.file.write('%s  %s' \
                            % (time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),
                            message))
            self.stdout.write(message)
        elif self.previous_message != '\n':
            self.file.write('\n')
            self.stdout.write('\n')
        self.previous_message = message

        if time.time() - self.time > 5:
            self.file.close()
            self.open()
            self.time = time.time()

    def flush(self):
        self.stdout.flush()
        self.stderr.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.stderr != None:
            sys.stderr = self.stderr
            self.stderr = None

        if self.file != None:
            self.file.close()
            self.file = None
