# src/eval_utils.py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
    title: str,
    save_path: str,
):
    """
    Plot and save a confusion matrix using matplotlib (no seaborn).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    labels : list of label names, e.g. ["negative", "positive"]
    title : figure title
    save_path : where to save the PNG figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 在每个格子里标数字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to: {save_path}")
