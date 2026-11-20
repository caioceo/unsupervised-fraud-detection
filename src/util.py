from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


PALETTE_MAIN = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
COLOR_HIGHLIGHT = "#0072B2"
COLOR_ALERT = "#D55E00"
COLOR_NEUTRAL = "#94A3B8"
COLOR_DARK = "#1E293B"


