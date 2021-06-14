import json
import pandas as pd
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import colorsys
import random
import glob


def load_json(filepath):
    file = open(filepath, 'rb')
    data = json.load(file)
    return data


def write_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)
