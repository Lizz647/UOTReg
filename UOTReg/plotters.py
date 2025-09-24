import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot_G_loss(G_loss_history):
    """
    Plot metrics during training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), dpi=200)
    axes[0].set_title('G Loss (Regression), log10', fontsize=12)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].plot(np.log10(np.array(G_loss_history)), label="G Loss")
    axes[0].legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)