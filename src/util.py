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


def configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": COLOR_DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": COLOR_DARK,
        "ytick.color": COLOR_DARK,
        "text.color": COLOR_DARK,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_data(file_path: str = "data/raw/base04_cartao_alta_dimensao.csv") -> pd.DataFrame:
    file = Path(file_path)
    if file.exists():
        return pd.read_csv(file)
    raise FileNotFoundError(f"Arquivo nao encontrado: {file.resolve()}")


def show_info(df: pd.DataFrame) -> None:
    print(f"Dimensoes da Base: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Total de Valores Ausentes (Nulos): {int(df.isnull().sum().sum())}")
    print("\nContagem de Tipos de Variaveis:")
    print(df.dtypes.value_counts().to_string())
    print("\nValores Nulos por Coluna:")
    print(df.isnull().sum().to_string())
    print("\nEstrutura Detalhada (info):")
    df.info()


