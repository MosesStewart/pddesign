import torch, pandas as pd, numpy as np, sys, os, re
from matplotlib import pyplot as plt
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]))
from derived.simulation import *
from rddesign.main import *