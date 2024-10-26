# music generation with transformer using pytorch and maestro Dataset
# 1: setup Environment and import libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import music21
import os
import glob
import random

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")