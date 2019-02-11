import json
from collections import namedtuple

with open("test.json", "r") as file:
    x = json.load(file, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))
    print(x.matches)