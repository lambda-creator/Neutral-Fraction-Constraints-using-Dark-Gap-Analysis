
import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pandas import read_csv
import os

# --------------------------------------------------------------------------
#                           PLOT SETTINGS:
# --------------------------------------------------------------------------
plt.rcParams.update({
    'figure.figsize': (12, 12),
    # 'figure.facecolor': 'white',
    # 'figure.edgecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'legend.numpoints': 1,

    # Axes and ticks
    'axes.linewidth': 2.5,
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    

    'xtick.major.size': 12,
    'xtick.major.width': 2.5,
    'xtick.minor.size': 8,
    'xtick.minor.width': 2.5,

    'ytick.major.size': 14,
    'ytick.major.width': 2.5,
    'ytick.minor.size': 8,
    'ytick.minor.width': 2.5,

    # Padding
    'xtick.major.pad': 7,
    'xtick.minor.pad': 7,

    # Font sizes
    'font.size': 35,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,

    # Font family and LaTeX
    'font.family': 'serif',
    'font.sans-serif': ['Georgia'],
    'text.usetex': False,
    'mathtext.fontset':'cm',

    # Image and lines
    'image.cmap': 'jet',
    'lines.linewidth': 2,
    'lines.markersize': 8,

    # Savefig options
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.dpi': 200,

    # Legend
    'legend.fancybox': True,
    'legend.frameon': True,
    'legend.edgecolor': "black"
})


'''
This code will calculate the F10 Statistics to give us more sensitivity to neutral patches rather than the opacity fluctuations.

F10 Statistics is a cdf of the dark gaps having length more than 10 h^-1 Mpc ; we use this as an "eyetest" for whether 
our simulation is able to reproduce the curve or not. 

As we will see in the results, we will be able to see that all models irrespective of their z_50 are able to produce this curve. 
This acts as a consistency check, as we don't have the error bounds on the F10 curve.


'''



# ----------------------------------------------------------------------------
#                       COSMOLOGICAL PARAMETER
# ---------------------------------------------------------------------------
c_km_s = 299792.458  # speed of light in km/s
c_cgs=c_km_s *1e5
omega_matter=0.302
omega_lambda=0.698
H0 = 2.1972e-18                         # s^-1
h = 0.678
cm_to_Mpc = 3.240779289e-25

L_thresh=10   #h^-1 Mpc


# -----------------------------------------------------------------
#                           dz => COMOVING LENGTH :
#------------------------------------------------------------------


def E_of_z(z):
    return np.sqrt(omega_matter*(1+z)**3 + omega_lambda)  # add omega_r, omega_k if needed


conversion_factor= cm_to_Mpc *h


def comoving_gap(z1, z2):
    # integral over 1/E(z)
    integral, _ = quad(lambda zz: 1.0/E_of_z(zz), z1, z2, epsabs=1e-10, epsrel=1e-10)
    return (c_cgs / H0) * integral *conversion_factor

# ---------------------------------------------------------------------------------------
#                           ZHU DATA : F10 STATISTICS
# ---------------------------------------------------------------------------------------
zhu_data = read_csv('gap_data/zhu_ly_alpha_dark_gaps.csv')
zhu_start = zhu_data.iloc[:, 2]
zhu_end = zhu_data.iloc[:, 3]


zhu_l = [comoving_gap(z1, z2) for z1, z2 in zip(zhu_start, zhu_end)]
zhu_l_arr = np.asarray(zhu_l)


z_range = np.arange(5.0, 6.0 + 0.02, 0.02)
gaps_2d = np.column_stack((zhu_start, zhu_l_arr))


gaps_2d_sorted = gaps_2d[np.argsort(gaps_2d[:, 0])]
mask_long = gaps_2d_sorted[:, 1] > L_thresh


cdf_long_zhu, bin_edges_zhu = np.histogram(gaps_2d_sorted[:, 0][mask_long], bins=z_range)
cdf_long_zhu = np.cumsum(cdf_long_zhu)
cdf_long_zhu = cdf_long_zhu / cdf_long_zhu[-1]   # Normalizing to 1

z_mid_zhu = 0.5 * (bin_edges_zhu[:-1] + bin_edges_zhu[1:])


# ---------------------------------------------------------------------------------------------
#                       SIMULATION F10 STATISTICS
# ---------------------------------------------------------------------------------------------

model_indices = [0, 12, 25]
model_names = ["Ultra Late", "Late", "Early"]
gamma_scale_values = [0.3, 0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
n_mocks = 45

gamma_scale_reduced=[0.3,0.6,3.0]


fig, axes = plt.subplots(1, 3, figsize=(40,9), sharey=True)
for m_index, model_idx in enumerate(model_indices):
    ax = axes[m_index]
    
    for gamma_val in gamma_scale_reduced:
        gamma_str = str(gamma_val).replace('.', 'pt')

        all_cdfs = []

        for j in range(n_mocks):
            filepath = f"model_{model_idx}/gamma_{gamma_str}/neutral_gaps_z_mock_{j}.npz"
            sim_data = np.load(filepath)
            z_start = sim_data["z_start_array"]
            z_end = sim_data["z_end_array"]


            sim_length = np.array([comoving_gap(z1, z2) for z1, z2 in zip(z_start, z_end)])
            sim_gaps_2d = np.column_stack((z_start, sim_length))
            sim_gaps_2d_sorted = sim_gaps_2d[np.argsort(sim_gaps_2d[:, 0])]

        
            mask_long = sim_gaps_2d_sorted[:, 1] > L_thresh
            counts_long, _ = np.histogram(sim_gaps_2d_sorted[:, 0][mask_long], bins=z_range)
            cdf_long=np.cumsum(counts_long)
            cdf_sim=cdf_long/cdf_long[-1]


            all_cdfs.append(cdf_sim)

        all_cdfs = np.array(all_cdfs)
        mean_cdf = np.nanmean(all_cdfs, axis=0)
        std_cdf = np.nanstd(all_cdfs, axis=0)

        z_centers = 0.5 * (z_range[1:] + z_range[:-1])

        ax.plot(z_centers, mean_cdf, label=fr"$f_{{\Gamma}}$={gamma_val}")
       
        # ax.fill_between(z_centers, mean_cdf - std_cdf, mean_cdf + std_cdf, alpha=0.2)
    ax.plot(z_mid_zhu, cdf_long_zhu, 'k--', linewidth=3, label='Zhu Data')


    # ax.set_title(f"{model_names[m_index]}")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(rf"$F_{{L}}$")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=20)
# plt.savefig(f"F30_plots/simulation_F_{L_thresh}.png")
plt.tight_layout()
plt.show()








