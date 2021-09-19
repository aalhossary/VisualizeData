from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

import spm1d

####################################################
infolder_mask = '{measurement}/{side}_{joint}{dimension}_{measurementSFX}_{subjname}.csv'
outfolder_mask = infolder_mask
SIDE_RIGHT = 'R'
SIDE_LEFT = 'L'
side = [SIDE_RIGHT, SIDE_LEFT]
joint = ['Hip', 'Knee', 'Ankle']
dim = ['X', 'Y', 'Z']
# measurement_suffix = ['Ang', 'moment']
measurementSFX = dict([('kinematic', 'Ang'), ('moment', 'moment')])

SUFFIX = 'SUFFIX'  # TODO @deprecated
SUBJECT = 'SUBJECT'
SIDE = 'SIDE'
JOINT = 'JOINT'
DIMENSION = 'DIMENSION'
alpha = 0.05  # TODO import value from MovementRx
roi = np.array([True] * 101)
HOTELLINGS_2 = 'hotellings2'
HOTELLINGS = 'hotellings'

VMIN_3D = 1
VMAX_3D = 11
NUM_LEVELS_3D = 4
under_color = (0.5, 0.5, 0.5)
cmap2 = cm.get_cmap('coolwarm', NUM_LEVELS_3D)
cmap2.set_under(color=under_color)
norm2 = Normalize(vmin=VMIN_3D, vmax=VMAX_3D)

####################################################


def read_individual_data_simplified(path: str, individual_name: str) -> List[np.ndarray]:
    parent_folder = Path(path, individual_name)
    ret: List[np.ndarray] = []

    measurement = 'moment'  # Moment
    s = side[0]  # right
    j = joint[0]  # hip

    for i_dimension in range(3):
        d = dim[i_dimension]
        print(measurement, s, j, d)
        file_name = infolder_mask \
            .replace('{measurement}', measurement).replace('{side}', s) \
            .replace('{joint}', j).replace('{dimension}', d) \
            .replace('{measurementSFX}', measurementSFX[measurement]) \
            .replace('{subjname}', individual_name)
        dof = np.loadtxt(Path(parent_folder, file_name), delimiter=',', dtype=float)  # shape = (n, 101)
        print(dof.shape)
        ret.append(dof)
    return ret  # List[ndarray(shape=(n, 101)] for 3 dimensions


def analyze_data_3d_simplified(): #returns np.ndarray(shape=(101,))
    ref = read_individual_data_simplified('./', 'refData')   #

    id_individual = 11  # patient No. 12
    individual = read_individual_data_simplified('./', f'TTA{(id_individual + 1):03}')
    # test_name = HOTELLINGS_2
    test_name = HOTELLINGS
    singlejoint_data_3d = eval_individual_vs_ref_simplified(alpha, individual, ref, roi, test_name)

    return singlejoint_data_3d  #np.ndarray(shape=(3, 101), dtype=float)


def analyze_bootstrapped_data_3d_simplified():
    base_ref: List[np.ndarray] = read_individual_data_simplified('./', 'refData')

    id_individual = 5  # subject No. 6
    individual: List[np.ndarray] = []
    ref: List[np.ndarray] = []

    for i_d in range(3):  # len(base_ref)):
        arr_to_split = base_ref[i_d]
        splits = np.split(arr_to_split, [id_individual, id_individual+1])
        individual.append(np.vstack([splits[1], splits[1]]))  # at least 2 sets of readings per joint per dimension
        ref.append(np.vstack((splits[0], splits[2])))

    # test_name = HOTELLINGS_2
    test_name = HOTELLINGS
    singlejoint_data_3d = eval_individual_vs_ref_simplified(alpha, individual, ref, roi, test_name)

    return singlejoint_data_3d


def eval_individual_vs_ref_simplified(alpha, individual, ref, roi, test_name):
    data_ya, data_yb = None, None
    for i_d in range(3):  # i_d stand for dimension_i not *ID* (This is i_d not id)
        # data_ya is the reference and data_yb is the subject
        if i_d == 0:
            data_ya = np.ndarray(shape=(*ref[i_d].shape, 3))
            data_yb = np.ndarray(shape=(*individual[i_d].shape, 3))
        data_ya[:, :, i_d] = ref[i_d]
        data_yb[:, :, i_d] = individual[i_d]

    spm_t = None
    test = eval('spm1d.stats.' + test_name)

    if test_name == HOTELLINGS_2:
        spm_t = test(data_ya, data_yb, roi=roi)
    elif test_name == HOTELLINGS:
        spm_t = test(data_ya, mu=data_yb.mean(axis=0), roi=roi)

    spmi_t = spm_t.inference(alpha)
    z = (spmi_t.z / spmi_t.zstar)
    print(z)
    return z


def draw_plot_simplified(data: np.ndarray, norm, cmap, title: str):

    fig: Figure = plt.figure()
    fig.suptitle(title)
    # fig.supxlabel('Joints')
    # fig.supylabel('Patients')
    data_len: int = 1  # only one patient here. Normally, it should be 12 or 20
    ################################
    first: Optional[Axes] = None
    # for i_patient, limb_data in enumerate(data):
    #     for i_joint in range(3):

    ax: Optional[Axes] = None
    first = ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel(f"Pt {11+1}", rotation=0, va='center', ha='right')

    ax.grid(False)
    ax.set_yticks([])
    # ax.set_yticklabels([])

    joint_data: np.ndarray = np.abs(data)
    # print(joint_data.shape)
    z_array = np.stack((joint_data, joint_data), axis=0)  # imshow takes *TWO dimensional* array
    # print(z_array.shape)

    ax.imshow(z_array, interpolation='nearest', cmap=cmap, aspect='auto', norm=norm)
    # ax.autoscale(enable=True, axis='both', tight=True)

    # -----------------------------------------

    ax = fig.add_subplot(2, 1, 2, sharex=first)
    # t2i.plot(ax=ax)
    ax.plot(joint_data)  # or data
    for i in np.arange(VMIN_3D, VMAX_3D, (VMAX_3D-VMIN_3D)/NUM_LEVELS_3D):
        ax.axhline(i, color=cmap(norm(i)))

    ax.set_xlabel(joint[0])

    fig.tight_layout()
    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()
    plt.close(fig)


###################################

if __name__ == '__main__':
    patient_12_rt_hip_moment_data = analyze_data_3d_simplified()
    draw_plot_simplified(patient_12_rt_hip_moment_data, norm=norm2, cmap=cmap2, title='Patient #12 rt hip moment')

    normal_6_rt_hip_moment_data = analyze_bootstrapped_data_3d_simplified()
    draw_plot_simplified(normal_6_rt_hip_moment_data, norm=norm2, cmap=cmap2, title='Normal #6 rt hip moment')
