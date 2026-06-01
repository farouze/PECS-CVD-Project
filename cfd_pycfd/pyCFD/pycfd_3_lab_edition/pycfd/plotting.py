from pathlib import Path
import matplotlib.pyplot as plt

def plot_cfd_matrix(report, path=None, show=True):
    matrix = report.matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title("CFD Complementarity Matrix")
    fig.colorbar(im, ax=ax, label="CFD Index")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig

def plot_cfd_ranking(report, path=None, show=True):
    ranking = report.ranking.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ranking["combination"].astype(str), ranking["cfd_index"])
    ax.set_ylabel("CFD Index")
    ax.set_title("CFD Combination Ranking")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
