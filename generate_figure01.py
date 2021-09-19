from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from data import cmap1, norm1, VMIN_SCALAR, VMAX_SCALAR, diseased_limbs, intact_limbs, summarize_data, \
    analyze_data_scalar, analyze_data_3d, cmap2, norm2
from spmclient import consts


global all_data_scalar  # : np.ndarray
global all_data_3d  # : np.ndarray


def draw_test_plot(i_measurement, limbs_to_draw, title: str):
    global all_data_scalar
    global all_data_3d

    data_selected_measurement_and_limb = all_data_3d[np.arange(all_data_3d.shape[0]), i_measurement, limbs_to_draw, ...]
    ################################

    fig: Figure = plt.figure(figsize=(24, 8), dpi=150)
    # fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')

    first: Optional[Axes] = None

    selected_ids = [0, 11]
    for order, i_individ in enumerate(selected_ids):
        i_side = limbs_to_draw[i_individ]
        # for i_individ, i_side in enumerate(limbs_to_draw):
        for i_joint in range(len(consts.joint)):

            ax: Optional[Axes] = None
            if i_joint == 0:
                first = ax = fig.add_subplot(4, 3*2, 3 * order + i_joint + 1)
                # ax.set_ylabel(f"Pt {i_individ+1}", rotation=0, va='center', ha='right')
            else:
                ax = fig.add_subplot(4, 3*2, 3 * order + i_joint + 1, sharey=first)
            # ax.set_xlabel(consts.joint[i_joint])
            # ax.set_xticks([])
            # ax.set_xticklabels([])
            ax.grid(False)
            ax.set_yticks([])
            # ax.set_yticklabels([])

            limb_data = data_selected_measurement_and_limb[i_individ]
            joint_data: np.ndarray = np.abs(limb_data[i_joint])
            # print(joint_data.shape)
            z_array = np.stack((joint_data, joint_data), axis=0)
            # print(z_array.shape)

            ax.imshow(z_array, interpolation='nearest', cmap=cmap2, aspect='auto', norm=norm2)
            # ax.autoscale(enable=True, axis='both', tight=True)
            #  ################################################################
            AZIMUTH = -80
            joint_mat = np.abs(all_data_scalar[i_individ, i_measurement, i_side, i_joint, :, :])
            print(i_individ, i_measurement, i_side, i_joint, joint_mat.shape)
            x = [range(joint_mat.shape[1])]
            ax = fig.add_subplot(2, 3*2, 6 + order * 3 + i_joint + 1, projection='3d', azim=AZIMUTH, elev=45)

            for i_dimension in range(len(consts.dim)):
                # joint_mat = all_data_3d[i_individ, i_measurement, i_side, i_joint]
                y = [i_dimension, i_dimension + 1]
                x, y = np.meshgrid(x, y, sparse=True)
                # print(x,y,X,Y)

                surf = ax.plot_surface(x, y, np.array([joint_mat[i_dimension], joint_mat[i_dimension]]),
                                       # rstride=1, cstride=1,
                                       cmap=cmap1, norm=norm1, linewidth=0, antialiased=True)
                ax.set_zlim(0, VMAX_SCALAR + (VMAX_SCALAR-VMIN_SCALAR)/2)

            # if i_joint == 0:
            #     # ax.set_zlabel(f'Pt {i_individ + 1}')
            #     zlim = ax.get_zlim()
            #     ax.text3D(-50, 0, np.average(zlim), f'Pt {i_individ + 1}', zdir='z')
            if order == len(selected_ids):
                ax.set_xlabel(['Hip', 'Knee', 'Ankle'][i_joint])
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['Flex/ext.', 'Abd/add.', 'Int/ext rot.'], rotation=(-90-AZIMUTH), va='center', ha='left')

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    # fig.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()
    plt.close(fig)


def show_data():

    # # draw kinematics affected
    # draw_plot(0, diseased_limbs, 'Kinematics Affected Limbs')
    # draw_unified_plot(0, diseased_limbs, 'Kinematics Affected Limbs Combined')
    #
    # # draw kinematics healthy
    # draw_plot(0, intact_limbs, 'Kinematics Healthy Limbs')
    # draw_unified_plot(0, intact_limbs, 'Kinematics Healthy Limbs Combined')
    #
    # # draw moments affected
    # draw_plot(1, diseased_limbs, 'Moments Affected Limbs')
    # draw_unified_plot(1, diseased_limbs, 'Moments Affected Limbs Combined')
    #
    # # draw moments healthy
    # draw_plot(1, intact_limbs, 'Moments Healthy Limbs')
    # draw_unified_plot(1, intact_limbs, 'Moments Healthy Limbs Combined')

    draw_test_plot(1, intact_limbs, 'Moments Healthy Limbs Fig 01')


def main():
    global all_data_scalar
    global all_data_3d
    # summarize_data()
    # all_data_scalar = analyze_data_scalar()
    # np.save('./data.npy', all_data_scalar)
    all_data_scalar = np.load('./data.npy')

    # all_data_3d = analyze_data_3d()
    # np.save('./data_3d.npy', all_data_3d)
    all_data_3d = np.load('./data_3d.npy')

    show_data()


if __name__ == '__main__':
    main()
