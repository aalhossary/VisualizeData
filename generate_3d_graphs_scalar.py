import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from data import cmap1, norm1, VMIN_SCALAR, VMAX_SCALAR, diseased_limbs, intact_limbs, summarize_data, \
    analyze_data_scalar
from spmclient import consts


global all_data_scalar  # : np.ndarray


def draw_plot(i_measurement, limbs_to_draw, title: str):
    global all_data_scalar

    AZIMUTH = -80

    fig: Figure = plt.figure(figsize=(12, 40), dpi=200)
    fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')
    for i_individ, i_side in enumerate(limbs_to_draw):
        for i_joint in range(len(consts.joint)):
            joint_mat = np.abs(all_data_scalar[i_individ, i_measurement, i_side, i_joint, :, :])
            print(i_individ, i_measurement, i_side, i_joint, joint_mat.shape)
            x = [range(joint_mat.shape[1])]
            ax = fig.add_subplot(12, 3, i_individ * 3 + i_joint + 1, projection='3d', azim=AZIMUTH, elev=45)

            for i_dimension in range(len(consts.dim)):
                # joint_mat = all_data_3d[i_individ, i_measurement, i_side, i_joint]
                y = [i_dimension, i_dimension + 1]
                x, y = np.meshgrid(x, y, sparse=True)
                # print(x,y,X,Y)

                surf = ax.plot_surface(x, y, np.array([joint_mat[i_dimension], joint_mat[i_dimension]]),
                                       # rstride=1, cstride=1,
                                       cmap=cmap1, norm=norm1, linewidth=0, antialiased=True)
                ax.set_zlim(0, VMAX_SCALAR + (VMAX_SCALAR-VMIN_SCALAR)/2)

            if i_joint == 0:
                # ax.set_zlabel(f'Pt {i_individ + 1}')
                zlim = ax.get_zlim()

                ax.text3D(-50, 0, np.average(zlim), f'Pt {i_individ + 1}', zdir='z')
            if i_individ == 11:
                ax.set_xlabel(['Hip', 'Knee', 'Ankle'][i_joint])
            if i_joint == 2:
                ax.set_yticks([1, 2, 3])
                ax.set_yticklabels(['Flex/ext.', 'Abd/add.', 'Int/ext rot.'], rotation=(-90-AZIMUTH), va='center', ha='left')
            else:
                ax.set_yticklabels([])

    fig.tight_layout(pad=0.1, w_pad=0.08)
    plt.savefig(f"{title}.png")  # , bbox_inches='tight')
    plt.show()
    plt.close(fig)


def draw_test_plot(i_measurement, limbs_to_draw, title: str):
    global all_data_scalar

    AZIMUTH = -80

    fig: Figure = plt.figure(figsize=(12, 8), dpi=150)
    # fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')

    selected_ids = [0, 11]
    for order, i_individ in enumerate(selected_ids):
        i_side = limbs_to_draw[i_individ]
        # for i_individ, i_side in enumerate(limbs_to_draw):
        for i_joint in range(len(consts.joint)):
            joint_mat = np.abs(all_data_scalar[i_individ, i_measurement, i_side, i_joint, :, :])
            print(i_individ, i_measurement, i_side, i_joint, joint_mat.shape)
            x = [range(joint_mat.shape[1])]
            ax = fig.add_subplot(len(selected_ids), 3, order * 3 + i_joint + 1, projection='3d', azim=AZIMUTH, elev=45)

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
    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()
    plt.close(fig)


def draw_unified_plot(i_measurement, limbs_to_draw, title: str):
    global all_data_scalar
    fig: Figure = plt.figure(figsize=(10, 10), dpi=400)
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-45, elev=45)

    # ax.invert_xaxis()
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')
    strip_thinkness = 4
    strip_strip_dist = 2
    num_dim = 3
    patient_thinkness = num_dim * strip_thinkness + (num_dim - 1) * strip_strip_dist
    patient_patient_dist = (strip_thinkness + strip_strip_dist) * 10
    # x = np.array(range(3 * 120))  # joints * time
    # y = np.array(range(10 * 12))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 12  # ndimensions * n_individuals
    # z = np.ma.masked_where(True, np.empty(shape=(len(y), len(x)), dtype=float), False)
    # # z = np.array(np.empty(shape=(len(y), len(x)), dtype=float))
    # z.fill(np.nan)
    for i_individ, i_side in enumerate(limbs_to_draw):
        for i_joint in range(len(consts.joint)):
            joint_mat = np.abs(all_data_scalar[i_individ, i_measurement, i_side, i_joint, :, :])
            print(i_individ, i_measurement, i_side, i_joint, joint_mat.shape)
            for i_dimension in range(len(consts.dim)):
                x_offset = 120 * i_joint
                y_offset = ((patient_thinkness + patient_patient_dist) * i_individ) + (strip_thinkness * i_dimension)
                x = np.arange(x_offset, x_offset + 101)
                y = np.arange(y_offset, y_offset + strip_thinkness)
                x, y = np.meshgrid(x, y, sparse=True)
                z = np.array([joint_mat[i_dimension]] * strip_thinkness)
                # joint_mat = all_data_3d[i_individ, i_measurement, i_side, i_joint]
                # print(x,y,X,Y)
                surf = ax.plot_surface(x, y, z,
                                       # rstride=1, cstride=1,
                                       cmap=cmap1, norm=norm1, linewidth=0, antialiased=True)

    a, b, c = 50, 150, 250
    ax.set_xticks((a, b, c))
    ax.set_xticklabels(('Hip', 'Knee', 'Ankle'))

    ax.set_yticks([i * (patient_thinkness + patient_patient_dist) for i in range(12)])
    ax.set_yticklabels([f'Pt {i + 1}' for i in range(12)])

    ax.set_zlim(0, VMAX_SCALAR + (VMAX_SCALAR-VMIN_SCALAR)/2)

    # fig.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
    plt.close(fig)


def show_data():

    # draw kinematics affected
    draw_plot(0, diseased_limbs, 'Kinematics Affected Limbs')
    draw_unified_plot(0, diseased_limbs, 'Kinematics Affected Limbs Combined')

    # draw kinematics healthy
    draw_plot(0, intact_limbs, 'Kinematics Healthy Limbs')
    draw_unified_plot(0, intact_limbs, 'Kinematics Healthy Limbs Combined')

    # draw moments affected
    draw_plot(1, diseased_limbs, 'Moments Affected Limbs')
    draw_unified_plot(1, diseased_limbs, 'Moments Affected Limbs Combined')

    # draw moments healthy
    draw_plot(1, intact_limbs, 'Moments Healthy Limbs')
    draw_unified_plot(1, intact_limbs, 'Moments Healthy Limbs Combined')

    draw_test_plot(1, intact_limbs, 'Moments Healthy Limbs Fig 01')


def main():
    global all_data_scalar
    # summarize_data()
    all_data_scalar = analyze_data_scalar()
    np.save('./data.npy', all_data_scalar)
    all_data_scalar = np.load('./data.npy')
    show_data()


if __name__ == '__main__':
    main()
