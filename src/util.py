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


def plot_hist_and_box(
    df: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    save_path: str | None = None,
) -> None:
    configure_plot_style()
    n_rows = len(specs)
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 3.4 * n_rows), gridspec_kw={"width_ratios": [2.2, 1.3]})
    if n_rows == 1:
        axs = np.array([axs])

    for i, (col, title, xlabel) in enumerate(specs):
        series = df[col]
        med = series.median()
        q75 = series.quantile(0.75)

        axs[i, 0].hist(series, bins=35, color=COLOR_HIGHLIGHT, edgecolor="white", alpha=0.88)
        axs[i, 0].axvline(med, color=COLOR_ALERT, linestyle="-", linewidth=1.8, label=f"Mediana: {med:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
        axs[i, 0].axvline(q75, color=COLOR_DARK, linestyle="--", linewidth=1.4, label=f"Q3 (75%): {q75:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
        axs[i, 0].set_title(title, loc="left", fontsize=11)
        axs[i, 0].set_xlabel(xlabel)
        axs[i, 0].set_ylabel("Clientes")
        axs[i, 0].legend(frameon=False, loc="upper right", fontsize=9)
        axs[i, 0].grid(axis="y", alpha=0.4)

        axs[i, 1].boxplot(
            series,
            vert=False,
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#BAE6FD", "edgecolor": COLOR_HIGHLIGHT, "linewidth": 1.2},
            medianprops={"color": COLOR_ALERT, "linewidth": 2.0},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLOR_NEUTRAL, "markeredgecolor": "none", "alpha": 0.5},
        )
        axs[i, 1].set_title(f"Dispersao e cauda: {col}", loc="left", fontsize=10)
        axs[i, 1].set_xlabel(xlabel)
        axs[i, 1].set_yticks([])
        axs[i, 1].grid(axis="x", alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.show()


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    extra_count: int | None = None,
) -> dict[str, float | int | str]:
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))
    n_noise = int((labels == -1).sum())

    if n_clusters > 1:
        sil = round(float(silhouette_score(X[valid_mask], labels[valid_mask])), 4)
        db = round(float(davies_bouldin_score(X[valid_mask], labels[valid_mask])), 4)
        ch = round(float(calinski_harabasz_score(X[valid_mask], labels[valid_mask])), 1)
    else:
        sil, db, ch = 0.0, 0.0, 0.0

    return {
        "algoritmo": model_name,
        "n_clusters": int(n_clusters),
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "ruidos_ou_transicao": int(extra_count if extra_count is not None else n_noise),
    }


def evaluate_dbscan_metrics(
    X: np.ndarray,
    dbscan_model,
    min_samples: int = 15,
) -> dict[str, float | int]:
    labels = dbscan_model.labels_
    valid_mask = labels != -1
    noise_mask = labels == -1
    n_total = len(labels)
    n_clusters = len(set(labels[valid_mask]))
    n_noise = int(noise_mask.sum())
    n_core = len(dbscan_model.core_sample_indices_)
    n_border = int(valid_mask.sum()) - n_core

    sil_clusters = round(float(silhouette_score(X[valid_mask], labels[valid_mask])), 4) if n_clusters > 1 else 0.0
    sil_global = round(float(silhouette_score(X, labels)), 4) if len(set(labels)) > 1 else 0.0
    db_clusters = round(float(davies_bouldin_score(X[valid_mask], labels[valid_mask])), 4) if n_clusters > 1 else 0.0
    ch_clusters = round(float(calinski_harabasz_score(X[valid_mask], labels[valid_mask])), 1) if n_clusters > 1 else 0.0

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dists = dists[:, min_samples - 1]
    mean_kdist_normal = float(k_dists[valid_mask].mean())
    mean_kdist_noise = float(k_dists[noise_mask].mean()) if n_noise > 0 else 0.0
    isolation_ratio = round(mean_kdist_noise / mean_kdist_normal, 2) if mean_kdist_normal > 0 else 0.0

    return {
        "n_clusters_densos": n_clusters,
        "qtd_ruidos_anomalias": n_noise,
        "taxa_ruido_pct": round((n_noise / n_total) * 100, 2),
        "qtd_core_points": n_core,
        "pct_core_points": round((n_core / n_total) * 100, 2),
        "qtd_border_points": n_border,
        "silhouette_nucleos_densos": sil_clusters,
        "silhouette_global_com_ruido": sil_global,
        "davies_bouldin_nucleos": db_clusters,
        "calinski_harabasz_nucleos": ch_clusters,
        "dist_media_k_vizinho_normal": round(mean_kdist_normal, 4),
        "dist_media_k_vizinho_anomalia": round(mean_kdist_noise, 4),
        "razao_isolamento_anomalia": isolation_ratio,
    }


def evaluate_gmm_metrics(
    X: np.ndarray,
    gmm_model,
    labels: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float | int]:
    eps_prob = np.clip(probs, 1e-12, 1.0)
    entropy_per_sample = -np.sum(eps_prob * np.log(eps_prob), axis=1)
    mean_entropy = float(np.mean(entropy_per_sample))
    max_probs = probs.max(axis=1)

    return {
        "n_componentes": int(gmm_model.n_components),
        "bic": round(float(gmm_model.bic(X)), 1),
        "aic": round(float(gmm_model.aic(X)), 1),
        "log_likelihood_media": round(float(gmm_model.score(X)), 4),
        "entropia_media_shannon": round(mean_entropy, 4),
        "probabilidade_max_media": round(float(max_probs.mean()), 4),
        "clientes_alta_certeza_ge_80": int((max_probs >= 0.80).sum()),
        "clientes_transicao_lt_80": int((max_probs < 0.80).sum()),
        "taxa_transicao_pct": round(float((max_probs < 0.80).mean() * 100), 2),
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
    }
