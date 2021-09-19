from typing import Optional

import numpy as np
from data import analyze_data_3d, norm2, analyze_bootstrapped_data_3d
from generate_overview_of_3d import draw_plot
from spmclient import consts

# global all_data_3d
global all_bootstrapped_data_3d


def show_data():
    right_limbs: np.ndarray = np.zeros((20,), dtype=int)  # TODO replace the 20 with len(data)
    left_limbs: np.ndarray = np.ones((20,), dtype=int)
    draw_plot(all_bootstrapped_data_3d, norm2, 1, right_limbs, 'Overview Normal Moments Right Limbs')
    draw_plot(all_bootstrapped_data_3d, norm2, 1, left_limbs, 'Overview Normal Moments Left Limbs')
    draw_plot(all_bootstrapped_data_3d, norm2, 0, right_limbs, 'Overview Normal Kinematics Right Limbs')
    draw_plot(all_bootstrapped_data_3d, norm2, 0, left_limbs, 'Overview Normal Kinematics Left Limbs')

# ################################


def main():
    # global all_data_scalar
    global all_bootstrapped_data_3d
    # g3d.summarize_data()

    # all_data_scalar = g3d.analyze_data_scalar()
    # np.save('./data.npy', all_data_scalar)
    # all_data_scalar = np.load('./data.npy')

    all_bootstrapped_data_3d = analyze_bootstrapped_data_3d()
    np.save('./data_bootstrapped_3d.npy', all_bootstrapped_data_3d)
    all_bootstrapped_data_3d = np.load('./data_bootstrapped_3d.npy')

    # g3d.show_data()
    show_data()


if __name__ == '__main__':
    main()


