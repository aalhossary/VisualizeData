import typing
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, FixedFormatter

from data import NUM_LEVELS_3D, cmap2, norm2, intact_limbs, diseased_limbs
from spmclient import consts

# global all_data_scalar
global all_data_3d


def show_data():
    # highest_scalar = np.max(all_data_scalar, -1)
    # print(highest_scalar.shape)
    # highest_3d = np.max(all_data_3d, -1)
    # print(highest_3d.shape)
    draw_plot(all_data_3d, norm2, 1, intact_limbs, 'Moments Healthy Limbs (Histo)')
    draw_plot(all_data_3d, norm2, 1, diseased_limbs, 'Moments Affected Limbs (Histo)')
    draw_plot(all_data_3d, norm2, 0, intact_limbs, 'Kinematics Healthy Limbs (Histo)')
    draw_plot(all_data_3d, norm2, 0, diseased_limbs, 'Kinematics Affected Limbs (Histo)')

    draw_plot(all_data_3d, norm2, 1, intact_limbs, 'Moments Healthy Limbs (Histo)%', percent=True)
    draw_plot(all_data_3d, norm2, 1, diseased_limbs, 'Moments Affected Limbs (Histo)%', percent=True)
    draw_plot(all_data_3d, norm2, 0, intact_limbs, 'Kinematics Healthy Limbs (Histo)%', percent=True)
    draw_plot(all_data_3d, norm2, 0, diseased_limbs, 'Kinematics Affected Limbs (Histo)%', percent=True)


def draw_plot(data: np.ndarray, norm, i_measurement, limbs_to_draw, title: str, percent: bool = False):
    # global all_data_scalar

    fig: Figure = plt.figure()  # figsize=(10, 25), dpi=300)
    fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')

    data_selected_measurement_and_limb = data[np.arange(data.shape[0]), i_measurement, limbs_to_draw, ...]

    data_max_per_joint = np.max(np.abs(data_selected_measurement_and_limb), -1, keepdims=False)
    data_max_per_patient_limb = np.max(np.abs(data_selected_measurement_and_limb), (-1, -2), keepdims=False)

    ax = fig.add_subplot(2, 1, 1)
    ax = typing.cast(Axes, ax)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1. * NUM_LEVELS_3D / 10))
    draw_hist(ax, data_max_per_patient_limb, norm, "At least one joint", percent)
    # ax.set_title(title)

    first: Optional[Axes] = None
    for i_joint in range(3):
        if i_joint == 0:
            first = ax = fig.add_subplot(2, 3, 4 + i_joint)
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1. * NUM_LEVELS_3D / 10))
        else:
            ax = fig.add_subplot(2, 3, 4 + i_joint, sharey=first)
        ax = typing.cast(Axes, ax)
        draw_hist(ax, data_max_per_joint[:, i_joint], norm, consts.joint[i_joint], percent)
    # ax = fig.add_subplot(2, 1, 2)
    # if percent:
    #     ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # draw_hist(ax, data_max_per_joint, norm, "All joints", percent)

    fig.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
    plt.close(fig)


def draw_hist(ax, data_max_per_patient_limb, norm, title, percent: bool = False):

    bin_heights, bin_starts, patches = ax.hist(data_max_per_patient_limb.clip(norm.vmin, norm.vmax), bins=NUM_LEVELS_3D,
                                               # weights=np.ones(data_max_per_patient_limb.shape)*NUM_LEVELS_3D,
                                               rwidth=0.9, align='left',
                                               range=(norm.vmin, norm.vmax),
                                               density=percent
                                               )
    # hist, bin_edges = np.histogram(data_max_per_patient_limb.clip(norm.vmin, norm.vmax), bins=NUM_LEVELS_3D,
    #                                range=(norm.vmin, norm.vmax))
    # bin_heights, bin_starts, patches = ax.hist(bin_edges[:-1], bin_edges, weights=hist, rwidth=0.9, align='left', density=percent)

    ax.set_title(title)
    # print('return', bin_heights, bin_starts, patches)
    # Now, we'll loop through our objects and set the color of each accordingly
    for bin_start, patch in zip(bin_starts, patches):
        color = cmap2(norm(bin_start))
        patch.set_facecolor(color)
    if NUM_LEVELS_3D == 4:
        # ax.set_xticks((.0, .25, .50, .75))
        ax.set_xticks(bin_starts[:-1])
        ax.set_xticklabels(('Mild', 'Mod', 'Sev', 'Xtreme'))


# ################################


def main():
    ## global all_data_scalar
    global all_data_3d
    # g3d.summarize_data()

    # all_data_scalar = analyze_data_scalar()
    # np.save('./data.npy', all_data_scalar)
    # all_data_scalar = np.load('./data.npy')

    # all_data_3d = analyze_data_3d()
    # np.save('./data_3d.npy', all_data_3d)
    all_data_3d = np.load('./data_3d.npy')

    show_data()


if __name__ == '__main__':
    main()


