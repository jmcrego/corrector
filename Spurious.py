import random
from collections import defaultdict

class Spurious():
    def __init__(self,f):
        self.spurious = defaultdict(int)
        with open(f,'r') as fd:
            for l in fd:
                self.spurious[l.rstrip()] = 1 #initially all equally likely

    def __call__(self):
        tok = random.choices(list(self.spurious.keys()), list(self.spurious.values()), k=1)[0]        
        return tok

