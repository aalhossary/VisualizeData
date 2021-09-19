import csv
from pathlib import Path

import spm1d
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# from spmclient import consts


TTEST_2 = 'ttest2'
TTEST = 'ttest'
TTEST_PAIRED = 'ttest_paired'
HOTELLINGS_2 = 'hotellings2'
HOTELLINGS = 'hotellings'
HOTELLINGS_PAIRED = 'hotellings_paired'
SIDE_RIGHT = 'R'
SIDE_LEFT = 'L'
all_side = [SIDE_RIGHT, SIDE_LEFT]
all_joint = ['Hip', 'Knee', 'Ankle']
all_dim = ['X', 'Y', 'Z']

measurement_folder = ['kinematic', 'moment']
measurementSFX = dict([('kinematic', 'Ang'), ('moment', 'moment')])
under_color = (0.5, 0.5, 0.5)
in_folder = 'C:\\Users\\Amr\\Google Drive\\Share\\MovementRX_dist\\Amr data\\TTA 1-13'
out_folder = './'
infolder_mask = '{measurement}/{side}_{joint}{dimension}_{measurementSFX}_{subjname}.csv'
outfolder_mask = infolder_mask
NUM_LEVELS_SCALAR = 4
VMIN_SCALAR = 1
VMAX_SCALAR = 4

cmap1 = cm.get_cmap('coolwarm', NUM_LEVELS_SCALAR)
cmap1.set_under(color=under_color)
norm1 = Normalize(vmin=VMIN_SCALAR, vmax=VMAX_SCALAR)

VMIN_3D = 1
VMAX_3D = 11
# VMAX_3D = 50
NUM_LEVELS_3D = 4
cmap2 = cm.get_cmap('coolwarm', NUM_LEVELS_3D)
cmap2.set_under(color=under_color)
norm2 = Normalize(vmin=VMIN_3D, vmax=VMAX_3D)

all_limbs = np.ones((12,), dtype=int)
# TODO put them in a file and generate the list automatically
# 1: Right
# 2: Right
# 3: Right
# 4: Left
# 5: Left
# 6: Right
# 7: Right
# 8: Right
# 9; Left
# 10: Right
# 11: Right
# 12: Left
diseased_limbs = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1], dtype=int)
intact_limbs = (all_limbs - diseased_limbs).astype(np.int)

global all_data_scalar
global all_data_3d


def analyze_data_3d():
    global all_data_3d
    # (patient id, measurement, side, joint, dimension, time series)
    all_data_3d = np.empty(shape=(12, 2, 2, 3, 101), dtype=float)
    alpha = 0.05  # TODO import value from MovementRx
    ref = read_individual_data('./', 'refData')
    roi = np.array([True] * 101)
    # test_name = HOTELLINGS_2
    test_name = HOTELLINGS
    for id_individual in range(12):
        individual = read_individual_data('./', f'TTA{(id_individual + 1):03}')
        eval_individual_vs_ref(all_data_3d, alpha, id_individual, individual, ref, roi, test_name)
    return all_data_3d


def analyze_bootstrapped_data_3d():
    # global all_bootstrapped_data_3d
    # (patient id, measurement, side, joint, dimension, time series)
    all_bootstrapped_data_3d = np.ones(shape=(20, 2, 2, 3, 101), dtype=float)
    alpha = 0.05  # TODO import value from MovementRx
    base_ref = read_individual_data('./', 'refData')
    roi = np.array([True] * 101)
    # test_name = HOTELLINGS_2
    test_name = HOTELLINGS

    for id_individual in range(20):
        # individual = read_individual_data('./', f'TTA{(id_individual + 1):03}')
        temp: np.ndarray = base_ref.copy()
        temp = np.reshape(temp, 36)  # Linear indexing easier
        individual = np.empty(temp.shape, dtype=np.ndarray)
        ref = np.empty(temp.shape, dtype=np.ndarray)

        for i in range(len(temp)):
            arr_to_split = temp[i]
            splits = np.split(arr_to_split, [id_individual, id_individual+1])
            individual[i] = np.vstack([splits[1], splits[1]])
            ref[i] = np.vstack((splits[0], splits[2]))

        # Reshape back
        individual = np.reshape(individual, base_ref.shape)
        ref = np.reshape(ref, base_ref.shape)

        eval_individual_vs_ref(all_bootstrapped_data_3d, alpha, id_individual, individual, ref, roi, test_name)

    return all_bootstrapped_data_3d


def eval_individual_vs_ref(out, alpha, id_individual, individual, ref, roi, test_name):
    for i_measurement, measurement in enumerate(measurement_folder):
        for i_side in range(len(all_side)):
            for i_joint in range(len(all_joint)):
                data_ya, data_yb = None, None
                for i_d in range(len(all_dim)):
                    # data_ya is the reference and data_yb is the subject
                    # spm_t = App.do_spm_test(data_ya, data_yb, test_name, roi=roi)
                    if i_d == 0:
                        data_ya = np.ndarray(shape=(*ref[i_measurement, i_side, i_joint, i_d].shape, 3))
                        data_yb = np.ndarray(shape=(*individual[i_measurement, i_side, i_joint, i_d].shape, 3))
                    data_ya[:, :, i_d] = ref[i_measurement, i_side, i_joint, i_d]
                    data_yb[:, :, i_d] = individual[i_measurement, i_side, i_joint, i_d]

                test = eval('spm1d.stats.' + test_name)
                spm_t = None
                if test_name == HOTELLINGS_2:
                    spm_t = test(data_ya, data_yb, roi=roi)
                elif test_name == HOTELLINGS:
                    spm_t = test(data_ya, mu=data_yb.mean(axis=0), roi=roi)
                spmi_t = spm_t.inference(alpha)
                z = (spmi_t.z / spmi_t.zstar)
                # print(z)
                out[id_individual, i_measurement, i_side, i_joint] = z


def summarize_data():
    individual_folder = Path('C:/Users/Amr/PycharmProjects/MovementRx/spmclient/res/refData')
    subjname = 'Ref'
    for measurement in measurement_folder:
        for i_side in range(len(all_side)):
            for i_joint in range(len(all_joint)):
                for i_dimension in range(len(all_dim)):
                    summarize_individual(individual_folder, subjname, measurement, i_side, i_joint, i_dimension)
    in_folder_path = Path(in_folder)
    subjname = 'subj1'
    for individual_folder in in_folder_path.iterdir():
        if not individual_folder.is_dir():
            continue
        for measurement in measurement_folder:
            for i_side in range(len(all_side)):
                for i_joint in range(len(all_joint)):
                    for i_dimension in range(len(all_dim)):
                        summarize_individual(individual_folder, subjname, measurement, i_side, i_joint, i_dimension)


def read_individual_data(path: str, individual_name: str) -> np.ndarray:
    parent_folder = Path(path, individual_name)
    # ret = np.empty(shape=(2, 2, 3, 3, 101), dtype=float)
    ret = np.empty(shape=(2, 2, 3, 3), dtype=np.ndarray)

    for i_measurement, measurement in enumerate(measurement_folder):
        for i_side in range(len(all_side)):
            for i_joint in range(len(all_joint)):
                for i_dimension in range(len(all_dim)):
                    side = all_side[i_side]
                    joint = all_joint[i_joint]
                    dim = all_dim[i_dimension]
                    print(measurement, side, joint, dim)
                    file_name = infolder_mask \
                        .replace('{measurement}', measurement).replace('{side}', side) \
                        .replace('{joint}', joint).replace('{dimension}', dim) \
                        .replace('{measurementSFX}', measurementSFX[measurement]) \
                        .replace('{subjname}', individual_name)
                    dof = np.loadtxt(Path(parent_folder, file_name), delimiter=',', dtype=float)
                    print(dof.shape)
                    ret[i_measurement, i_side, i_joint, i_dimension] = dof
    return ret


def analyze_data_scalar():
    global all_data_scalar
    # patient id, measure, side, joint, dimension, time series
    all_data_scalar = np.empty(shape=(12, 2, 2, 3, 3, 101), dtype=float)
    roi = np.array([True] * 101)
    # test_name = TTEST_2
    test_name = TTEST
    alpha = 0.05
    ref = read_individual_data('./', 'refData')
    for id_individual in range(12):
        individual = read_individual_data('./', f'TTA{id_individual + 1:03}')
        for i_measurement, measurement in enumerate(measurement_folder):
            for i_side in range(len(all_side)):
                for i_joint in range(len(all_joint)):
                    for i_dimension in range(len(all_dim)):
                        # data_ya is the reference and data_yb is the subject
                        # spm_t = App.do_spm_test(data_ya, data_yb, test_name, roi=roi)

                        # spm_t = App.do_spm_test(ref[i_measurement, i_side, i_joint, i_dimension],
                        #                         individual[i_measurement, i_side, i_joint, i_dimension],
                        #                         test_name, roi=roi)
                        test = eval('spm1d.stats.' + test_name)
                        spm_t = None
                        if test_name == TTEST_2:
                            spm_t = test(ref[i_measurement, i_side, i_joint, i_dimension],
                                         individual[i_measurement, i_side, i_joint, i_dimension],
                                         roi=roi)
                        elif test_name == TTEST:
                            spm_t = test(ref[i_measurement, i_side, i_joint, i_dimension],
                                         individual[i_measurement, i_side, i_joint, i_dimension].mean(axis=0),
                                         roi=roi)

                        spmi_t = spm_t.inference(alpha)
                        z = (spmi_t.z / spmi_t.zstar)
                        # print(z)
                        all_data_scalar[id_individual, i_measurement, i_side, i_joint, i_dimension] = z
    return all_data_scalar


def summarize_individual(individual: Path, subjname: str, measurement: str, i_side: int, i_joint: int,
                         i_dimension: int):
    side = all_side[i_side]
    joint = all_joint[i_joint]
    dim = all_dim[i_dimension]
    print(measurement, side, joint, dim)
    file_name = infolder_mask \
        .replace('{measurement}', measurement).replace('{side}', side) \
        .replace('{joint}', joint).replace('{dimension}', dim) \
        .replace('{measurementSFX}', measurementSFX[measurement]) \
        .replace('{subjname}', subjname)

    with open((individual / file_name), 'r') as infile:
        reader = csv.reader(infile, delimiter=',')
        headers = next(reader)
        # headers = [_.strip() for _ in headers]

        data = np.array(list(reader)).astype(float)

        file_name = file_name.replace(f"{subjname}.csv", f"{individual.name}.csv")

        outpath = Path(out_folder, individual.name) / file_name
        out_folder_path = outpath.parent
        if not out_folder_path.exists():
            out_folder_path.mkdir(parents=True)
        print(out_folder_path)

        with open(outpath, 'w') as outfile:
            # Header not needed here
            # print(', '.join(headers), file=outfile)

            # data = np.matrix(np.mean(data, 0))

            # print(data.shape)
            np.savetxt(outfile, data, fmt='%.9f', delimiter=',', newline='\n')


def main():
    global all_data_scalar
    global all_data_3d
    summarize_data()
    all_data_scalar = analyze_data_scalar()
    np.save('./data.npy', all_data_scalar)
    all_data_scalar = np.load('./data.npy')

    all_data_3d = analyze_data_3d()
    np.save('./data_3d.npy', all_data_3d)
    all_data_3d = np.load('./data_3d.npy')


if __name__ == '__main__':
    main()
