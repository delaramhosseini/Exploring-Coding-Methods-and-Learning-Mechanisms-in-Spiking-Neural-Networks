from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial.distance import cosine


def memoized_number():
    dic = [0]

    def inside():
        dic[0] += 1
        return f"plot({dic[0]}):"

    return inside


def plot(
    net,
    title=None,
    ngs=[],
    sgs=[],
    scaling_factor=3,
    label_font_size=8,
    print_sum_activities=False,
    recorder_index=13,
    env_recorder_index=14,
):
    n = len(ngs)
    fig_counter = memoized_number()
    fig, axd = plt.subplot_mosaic(
        """
        BBCC
        BBCC
        FFHH
        FFHH
        """,
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(12 * scaling_factor, 6 * scaling_factor),
    )

    fig.suptitle(
        title or "Plot",
        fontsize=(label_font_size + 4) * scaling_factor,
    )
    sg = sgs[0]

    axd["B"].scatter(
        ngs[0][env_recorder_index, 0].variables["spike"][:, 0].cpu(),
        ngs[0][env_recorder_index, 0].variables["spike"][:, 1].cpu(),
        label=f"{ngs[0].tag}",
    )
    axd["B"].set_ylabel(
        "spike (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["B"].set_xlim(0, net.iteration)
    axd["B"].set_ylim(-1, ngs[0].size)
    axd["B"].set_title(
        "1st Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    )
    axd["B"].set_xlabel(
        f"{fig_counter()} time ({ngs[0].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["B"].grid()

    axd["C"].scatter(
        ngs[1][env_recorder_index, 0].variables["spike"][:, 0].cpu(),
        ngs[1][env_recorder_index, 0].variables["spike"][:, 1].cpu(),
        label=f"{ngs[1].tag}",
    )
    axd["C"].set_ylabel(
        "spike (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["C"].set_xlim(0, net.iteration)
    axd["C"].set_ylim(-1, ngs[1].size)

    axd["C"].set_title(
        "2nd Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    )
    axd["C"].set_xlabel(
        f"{fig_counter()} time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["C"].grid()

    axd["F"].plot(
        sg[recorder_index, 0].variables["W"][:, -1, :].cpu(),
    )
    axd["F"].set_ylabel("Weight", fontsize=label_font_size * scaling_factor)
    axd["F"].set_xlim(0, net.iteration)
    # axd["F"].set_title("Scatter-Plot", fontsize=(label_font_size + 1) * scaling_factor)
    axd["F"].set_xlabel(
        f"{fig_counter()} time (input neuron number {ngs[0].size-1})",
        fontsize=label_font_size * scaling_factor,
    )

    axd["H"].plot(
        sg[recorder_index, 0].variables["W"][:, :, -1].cpu(),
    )
    axd["H"].set_ylabel("Weight", fontsize=label_font_size * scaling_factor)
    axd["H"].set_xlim(0, net.iteration)
    # axd["H"].set_title("Scatter-Plot", fontsize=(label_font_size + 1) * scaling_factor)
    axd["H"].set_xlabel(
        f"{fig_counter()} time (output neuron number {ngs[1].size-1})",
        fontsize=label_font_size * scaling_factor,
    )
    for key, ax in axd.items():
        ax.xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
        ax.yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    fig.show()
