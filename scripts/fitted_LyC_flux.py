import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

# --- MNRAS-style formatting ---
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

"""
Plotting the fitted profile for sanity check and observing the fit for true mean free path value.
This code will plot the flux vs mean free path values and will give the true mean free path based on the fitting values.
"""


# -------------------------
# Settings 
# -------------------------
z_mean = 5.412723                   #change accordingly;
outdir = f"fitted_z_{z_mean:.2f}"
h = 0.678
L_box = 80.0  # h^-1 cMpc

lambda_0_to_plot = -1.4       # can be chosen from the Gamma list
Gamma_index_to_plot = 50     # index into mean_Gamma_vals (0-99), mid-range

mean_Gamma_vals = np.logspace(-15.0, -10.0, 100)
mean_Gamma_pick = mean_Gamma_vals[Gamma_index_to_plot]

# -------------------------
# Two-parameter fit 
# -------------------------

def lyc_profile(x, F0, lambda_mfp):
    return F0 * np.exp(-x / lambda_mfp)

# -------------------------
# Load flux profile
# -------------------------
fname = f'{outdir}\\flux_profile_lambda_{lambda_0_to_plot:.3f}_Gamma_{mean_Gamma_pick:.3e}.txt'

if not os.path.exists(fname):
    raise FileNotFoundError(f"Profile file not found: {fname}\nMake sure you saved profiles in the MPI code.")

data = np.loadtxt(fname)
x    = data[:, 0]   # h^-1 cMpc
F    = data[:, 1]   # mean flux

# -------------------------
# Two-parameter fit 
# -------------------------
mask = F > 1e-6
x_fit = x[mask]
F_fit = F[mask]

try:
    popt2, _ = curve_fit(
        lyc_profile, x_fit, F_fit,
        p0=[1.0, L_box / 2.0],
        bounds=([0.0, 0.01], [1.1, 1e4]),
        maxfev=10000
    )
    F0_fit, lmfp_fit = popt2
    fit_success = True
except Exception as e:
    print(f"2-param fit failed: {e}")
    fit_success = False


x_model = np.linspace(x[0], x[-1], 500)

plt.plot(x, F, 'k.', ms=3, alpha=0.7, label='Stacked flux profile')

if fit_success:
        plt.plot(x_model, lyc_profile(x_model, *popt2), 'r-', lw=2,
                label=f'2-param fit: $F_0$={F0_fit:.3f}, $\\lambda_{{mfp}}$={lmfp_fit:.2f} $h^{{-1}}$cMpc')

plt.xlabel(r'$x \; [h^{-1} \, \mathrm{cMpc}]$', fontsize=13)
plt.ylabel(r'$\langle F_\mathrm{Lyc} \rangle$', fontsize=13)
plt.title(f'$\\lambda_0$ = {lambda_0_to_plot:.2f},  '
                 f'$\\langle\\Gamma\\rangle$ = {mean_Gamma_pick:.2e} s$^{{-1}}$\n', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, alpha=0.3)

plt.suptitle(f'     Ly-continuum Transmission Profile  |  z = {z_mean:.3f}', fontsize=26)
plt.tight_layout()
plt.savefig(f'{outdir}\\flux_fit_lambda{lambda_0_to_plot:.3f}_Gamma{Gamma_index_to_plot}.png', dpi=150)
plt.show()
print("Plot saved.")