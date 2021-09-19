from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data import analyze_data_3d, norm2, intact_limbs, diseased_limbs, cmap2
from spmclient import consts

global all_data_3d


def show_data():
    draw_plot(all_data_3d, norm2, 1, intact_limbs, 'Overview Moments Healthy Limbs')
    draw_plot(all_data_3d, norm2, 1, diseased_limbs, 'Overview Moments Affected Limbs')
    draw_plot(all_data_3d, norm2, 0, intact_limbs, 'Overview Kinematics Healthy Limbs')
    draw_plot(all_data_3d, norm2, 0, diseased_limbs, 'Overview Kinematics Affected Limbs')


def draw_plot(data: np.ndarray, norm, i_measurement, limbs_to_draw, title: str, percent: bool = False):
    # global all_data_scalar

    fig: Figure = plt.figure()  # figsize=(10, 25), dpi=300)
    fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')
    data_len: int = len(data)  # 12 or 20
    data_selected_measurement_and_limb = data[np.arange(data.shape[0]), i_measurement, limbs_to_draw, ...]
    ################################
    first: Optional[Axes] = None
    for i_patient, limb_data in enumerate(data_selected_measurement_and_limb):
        for i_joint in range(3):
            ax: Optional[Axes] = None
            if i_joint == 0:
                first = ax = fig.add_subplot(data_len, 3, 3 * i_patient + i_joint + 1)
                ax.set_ylabel(f"Pt {i_patient+1}", rotation=0, va='center', ha='right')
            else:
                ax = fig.add_subplot(data_len, 3, 3 * i_patient + i_joint + 1, sharey=first)
            if i_patient == len(data_selected_measurement_and_limb) - 1:
                ax.set_xlabel(consts.joint[i_joint])
            else:
                ax.set_xticks([])
                # ax.set_xticklabels([])
            ax.grid(False)
            ax.set_yticks([])
            # ax.set_yticklabels([])

            joint_data: np.ndarray = np.abs(limb_data[i_joint])
            # print(joint_data.shape)
            z_array = np.stack((joint_data, joint_data), axis=0)
            # print(z_array.shape)

            ax.imshow(z_array, interpolation='nearest', cmap=cmap2, aspect='auto', norm=norm)
            # ax.autoscale(enable=True, axis='both', tight=True)

    fig.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ################################


def main():
    # global all_data_scalar
    global all_data_3d
    # g3d.summarize_data()

    # all_data_scalar = g3d.analyze_data_scalar()
    # np.save('./data.npy', all_data_scalar)
    # all_data_scalar = np.load('./data.npy')

    all_data_3d = analyze_data_3d()
    np.save('./data_3d.npy', all_data_3d)
    all_data_3d = np.load('./data_3d.npy')

    # g3d.show_data()
    show_data()


if __name__ == '__main__':
    main()


