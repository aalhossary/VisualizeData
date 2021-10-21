import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.scale import LogScale

import spm1d


all_limbs = np.ones((12,), dtype=int)
diseased_limbs = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1], dtype=int)
intact_limbs = (all_limbs - diseased_limbs).astype(np.int)
dir0         = os.path.dirname( __file__ )
measurement  = 'moment'
# limb         = ['R', 'L']
VMIN_3D = 20
VMAX_3D = 300
NUM_LEVELS_3D = 4
cmap2 = cm.get_cmap('coolwarm', NUM_LEVELS_3D)
cmap2.set_under(color='0.5')
norm2 = Normalize(vmin=VMIN_3D, vmax=VMAX_3D)


###################################

def read_indiv_data(dir0, indiv_name, measurement, limb, joint):
    fpath = lambda x: os.path.join( dir0, indiv_name, measurement, f'{limb}_{joint}{x}_{measurement}_{indiv_name}.csv')
    return np.dstack(   [np.loadtxt( fpath(c) , delimiter=',', dtype=float)  for c in ['X','Y','Z']]   )


def calc_plots(dir, measurement, limbs, joint, remove_patient_6: bool = False):
    # Load data:
    yr = read_indiv_data(dir, 'refData', measurement, 'R', joint)  # reference data (one observation per subject)
    if remove_patient_6:
        yr = np.delete(yr, 5, axis=0)  # optionally remove one subject from the reference dataset;  this subject exhibits substantial deviation from the mean
        limbs = np.delete(limbs, 5)
    Y = [read_indiv_data(dir, f'TTA{i+1:03}', measurement, limbs[i], joint) for i in range(len(limbs))]  # all TTA data
    ytta = np.array([yy.mean(axis=0) for yy in Y])  # TTA data (one observation --- i.e., one mean --- per subject)
    # Calculate T2 statistic for each TTA subject:
    ztta = np.array([spm1d.stats.hotellings(yr, mu=yy).z for yy in ytta])
    # Calculate T2 statistic for each reference subject:
    zr = []
    for i, yy in enumerate(yr):
        yrr = np.delete(yr, i, axis=0)  # reduced reference data
        zr.append(spm1d.stats.hotellings(yrr, mu=yy).z)
    zr = np.array(zr)
    return zr, ztta


def plot_fig(ax: Axes, limb_indices_to_draw: List, title: str, joint, exclude_patient_6 = False):
    limbs = ['R' if l == 0 else 'L' for l in limb_indices_to_draw]
    zr, ztta = calc_plots(dir0, measurement, limbs, joint, remove_patient_6=exclude_patient_6)
    h1 = ax.plot(ztta.T, color='r')[0]
    h0 = ax.plot(zr.T, color='g')[0]
    ax.legend([h0, h1], ['Control Ref', 'TTA'], fontsize=10, loc='upper right')
    ax.set_xlabel('Time  (%)')
    ax.set_title(title)
    ax.set_ylabel('$T^2$ value')

    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), size=8)
    ax.set_ylim(top=1000 if exclude_patient_6 else 1750, bottom=0)
    for i in np.arange(VMIN_3D, VMAX_3D, (VMAX_3D - VMIN_3D) / NUM_LEVELS_3D):
        ax.axhline(i, color=cmap2(norm2(i)))


#####################################


for exclude_patient_6 in [True, False]:
    for joint in ['Knee', 'Hip']:
        # plt.close('all')
        fig: Figure = plt.figure()  # figsize=(10, 25), dpi=300)
        name = f"TTA vs. Control Ref ({joint}{', excluding #6' if exclude_patient_6 else ''})"
        fig.suptitle(name)
        for i, tta_limbs, title in [(0, intact_limbs, 'Intact limbs'), (1, diseased_limbs, 'Diseased limbs')]:
            ax = fig.add_subplot(1, 2, i+1)
            plot_fig(ax, tta_limbs, title, joint, exclude_patient_6)

        fig.tight_layout()
        plt.savefig(f"{name}.png", bbox_inches='tight')
        plt.show()
        plt.close(fig)

