from matplotlib import pyplot as plt
import torch

# Customization variables


def print_plot(
    net,
    title=None,
    scaling_factor=0.7,
    ngs=[],
    recorder_index=13,
    event_index=14,
    print_sum_activities=False,
    raster_plot=False,
):
    font_size = 30
    PLOT_WIDTH = int(48 * scaling_factor)
    PLOT_HEIGHT_PER_ROW = int(10 * scaling_factor)
    NUM_SUBPLOTS_PER_ROW = 3
    LINESTYLES = ["-", "--", "-.", ":"]
    COLORS = ["blue", "orange", "red", "pink"]

    n = len(ngs)
    k = n + 1 if n < 3 else n + 2
    num_rows = k if k != n else n

    # Create a grid of subplots
    fig, axs = plt.subplots(
        num_rows,
        NUM_SUBPLOTS_PER_ROW,
        figsize=(PLOT_WIDTH, PLOT_HEIGHT_PER_ROW * num_rows),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    av_current = [ngs[i][recorder_index, 0].variables["I"] for i in range(n)]
    for i in range(n):
        # Plot 1 - Voltage and Current
        axs[i, 0].plot(ngs[i][recorder_index, 0].variables["u"].cpu())
        axs[i, 0].plot(
            av_current[i].cpu() - torch.max(av_current[i]) - 80,
            linestyle=LINESTYLES[i % len(LINESTYLES)],
        )  # Adjust linestyle for current
        axs[i, 0].set_xlabel(f"time {ngs[i].tag}", fontsize=font_size * scaling_factor)
        axs[i, 0].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[i, 0].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)

        axs[i, 0].legend(
            ["Voltage (top)", "Current (bottom)"],
            loc="lower right",
            bbox_to_anchor=(1, 1),
            fontsize=font_size * scaling_factor,
        )

        # Plot 2 - Current
        axs[i, 1].plot(ngs[i][recorder_index, 0].variables["I"].cpu())
        axs[i, 1].set_xlim(0, ngs[0].network.iteration)
        axs[i, 1].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[i, 1].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[i, 1].set_ylabel(f"Current", fontsize=font_size * scaling_factor)
        axs[i, 1].set_xlabel(f"time {ngs[i].tag}", fontsize=font_size * scaling_factor)

        # Plot 3 - activity
        axs[i, 2].plot(ngs[i][recorder_index, 0].variables["T"].cpu())
        axs[i, 2].set_xlim(0, ngs[0].network.iteration)
        axs[i, 2].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[i, 2].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[i, 2].set_ylabel(f"activity", fontsize=font_size * scaling_factor)
        axs[i, 2].set_xlabel(f"time {ngs[i].tag}", fontsize=font_size * scaling_factor)

    # Plot 4 - Spike

    if k != n:
        colors = ["blue", "orange", "red", "pink"]
        if n == 3:
            for i in range(n):
                axs[k - 2, i].scatter(
                    ngs[i][event_index, 0].variables["spike"][:, 0].cpu(),
                    ngs[i][event_index, 0].variables["spike"][:, 1].cpu()
                    + sum([ng.size for ng in ngs[:i]]),
                    color=COLORS[i % len(COLORS)],
                    label=f"{ngs[i].tag}",
                )
                axs[k - 2, i].set_ylabel(
                    "spike (neuron number)", fontsize=font_size * scaling_factor
                )

                axs[k - 2, i].xaxis.set_tick_params(
                    labelsize=(font_size - 2) * scaling_factor
                )
                axs[k - 2, i].yaxis.set_tick_params(
                    labelsize=(font_size - 2) * scaling_factor
                )
                axs[k - 2, i].set_xlim(0, ngs[0].network.iteration)
                axs[k - 2, i].set_xlabel(
                    f"time {ngs[i].tag}",
                    fontsize=font_size * scaling_factor,
                )
        for i in range(n):
            axs[k - 1, 0].scatter(
                ngs[i][event_index, 0].variables["spike"][:, 0].cpu(),
                ngs[i][event_index, 0].variables["spike"][:, 1].cpu()
                + sum([ng.size for ng in ngs[:i]]),
                color=COLORS[i % len(COLORS)],
                label=f"{ngs[i].tag}",
            )
            axs[k - 1, 1].plot(
                torch.sum(ngs[i][recorder_index, 0].variables["I"], axis=1)
                / ngs[i].size,
                color=COLORS[i % len(COLORS)],
                label=f"{ngs[i].tag}",
            )
            axs[k - 1, 2].plot(
                ngs[i][recorder_index, 0].variables["T"],
                color=COLORS[i % len(COLORS)],
                label=f"{ngs[i].tag}",
            )

        ## waited sum of all activities
        if print_sum_activities:
            axs[k - 1, 2].plot(
                sum(
                    [
                        ngs[i][recorder_index, 0].variables["T"] * ngs[i].size
                        for i in range(n)
                    ]
                )
                / sum([ngs[i].size for i in range(n)]),
                color="black",
                label=f"overal activity",
                linewidth=4.0,
            )

        axs[k - 1, 0].set_xlim(0, ngs[0].network.iteration)
        axs[k - 1, 0].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 0].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 0].set_ylabel("spike", fontsize=font_size * scaling_factor)
        axs[k - 1, 0].set_xlabel("time", fontsize=font_size * scaling_factor)
        axs[k - 1, 0].legend(
            loc="lower right",
            bbox_to_anchor=(1, 1),
            fontsize=font_size * scaling_factor,
        )

        axs[k - 1, 1].set_xlim(0, ngs[0].network.iteration)
        axs[k - 1, 1].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 1].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 1].set_ylabel(
            "average of current", fontsize=font_size * scaling_factor
        )
        axs[k - 1, 1].set_xlabel("time", fontsize=font_size * scaling_factor)
        axs[k - 1, 1].legend(
            loc="lower right",
            bbox_to_anchor=(1, 1),
            fontsize=font_size * scaling_factor,
        )

        axs[k - 1, 2].set_xlim(0, ngs[0].network.iteration)
        axs[k - 1, 2].yaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 2].xaxis.set_tick_params(labelsize=(font_size - 2) * scaling_factor)
        axs[k - 1, 2].set_ylabel("activity", fontsize=font_size * scaling_factor)
        axs[k - 1, 2].set_xlabel("time", fontsize=font_size * scaling_factor)
        axs[k - 1, 2].legend(
            loc="lower right",
            bbox_to_anchor=(1, 1),
            fontsize=font_size * scaling_factor,
        )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
