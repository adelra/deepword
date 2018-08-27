#!/usr/bin/python
from subprocess import Popen
import time

while True:
    print("\nStarting " + "lstm_seq2seq.py")
    p = Popen("python " + "lstm_seq2seq.py", shell=True)
    time.sleep(14400)
    p.kill()
    print("killed")