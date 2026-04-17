
import numpy as np 
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.integrate import quad
from matplotlib import rcParams
from scipy.stats import ks_2samp
from itertools import combinations

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
This code plots the CDF to get an idea about whether the models are able to produce the observations from Zhu'21 et al or not.
The CDF is used as a simple diagnostic tool to get an idea about how different models fare with each other.
Steps: 
1) Define CDF as P(L<=L_i) i.e. CDF= (#dark gaps with L<=L_i)/ N  where N=total number of dark gaps. 
2) We will try to use KS test, D,p-distribution tests since the sample size of the dark gaps is very large and KS-2 samp test will always 
give very less p-values, so we want to understand how the p-/D- distribution looks like.

"""
c_cgs = 2.99792458e10
H0 = 67.7 * 1e5 / (3.086e24)

omega_matter = 0.302
omega_lambda = 0.698
h = 0.678
cm_to_Mpc = 3.24078e-25
conversion_factor = cm_to_Mpc * h

model_indices = [0, 12, 25]
n_mocks=45
def get_redshift_bin_mask(z_start,z_end,zmin,zmax,dz_min=0.00315):
    z_mid=0.5*(z_start+z_end)
    dz = z_end - z_start
    return (z_mid>=zmin) & (z_mid<zmax) & (dz >= dz_min)

def E_of_z(z):
    return np.sqrt(omega_matter*(1+z)**3 + omega_lambda)

def comoving_gap(z1, z2):
    integral, _ = quad(lambda zz: 1.0/E_of_z(zz), z1, z2, epsabs=1e-10, epsrel=1e-10)
    return (c_cgs / H0) * integral * conversion_factor


def empirical_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf



filename='gap_data/zhu_ly_alpha_dark_gaps.csv'
data=read_csv(filename)
zhu_start=data.iloc[:, 2]
zhu_end=data.iloc[:, 3]
z_minimum=0.00315
zhu_dz = zhu_end - zhu_start
mask_min = zhu_dz >= z_minimum
zhu_dz = zhu_dz[mask_min]
zhu_start = zhu_start[mask_min]
zhu_end = zhu_end[mask_min]
z_mid = 0.5*(zhu_start + zhu_end)

# Zhu data length
lzhu = [comoving_gap(z1, z2) for z1, z2 in zip(zhu_start, zhu_end)]

#1. for 5.0<z<6.0:


# Bin 1: z = [5.0100, 5.3440], z_center = 5.202, Δz = 0.334, N = 246
# Bin 2: z = [5.3440, 5.5652], z_center = 5.457, Δz = 0.221, N = 246
# Bin 3: z = [5.5652, 5.9725], z_center = 5.711, Δz = 0.407, N = 245

zmin=[5.0100, 5.3440, 5.5652]
zmax=[5.3440, 5.5652, 5.9725]
mask=[]
z_edges = np.array([5.0100, 5.3440, 5.5652, 5.9725])
z_mid=0.5*(zhu_start+zhu_end)
fig, axes = plt.subplots(1,3, figsize=(18,6), sharey=True)

model_labels = ["Ultra Late", "Late", "Early"]
bin_colors = ["tab:red","tab:blue","tab:green"]

gamma_val = 1.0
gamma_str = str(gamma_val).replace('.', 'pt')

zhu_cdf_bins = []
zhu_lengths_bins=[]
for ibin in range(3):   #ZHU FOR LOOP

    mask_bin = get_redshift_bin_mask(
        zhu_start, zhu_end,
        z_edges[ibin], z_edges[ibin+1],
        dz_min=z_minimum
    )

    zhu_start_bin = zhu_start[mask_bin]
    zhu_end_bin   = zhu_end[mask_bin]

    lengths = [
        comoving_gap(z1, z2)
        for z1, z2 in zip(zhu_start_bin, zhu_end_bin)
    ]
    print(f"Zhu Data Points in redshift bins {z_edges[ibin]} to {z_edges[ibin+1]}:",len(lengths))
    lx, ly = empirical_cdf(lengths)

    zhu_cdf_bins.append((lx, ly))
    zhu_lengths_bins.append(np.array(lengths))


# store p-value distributions
pvals_all = {label: [[] for _ in range(3)] for label in model_labels}
Dvals_all= {label: [[] for _ in range(3)] for label in model_labels}
for imodel, m in enumerate(model_indices):                                          #SIMULATION FOR BLOCK

    ax = axes[imodel]

    for ibin in range(3):
        lxzhu, lyzhu = zhu_cdf_bins[ibin]

        all_cdfs = []
        p_values=[]
        mock_lengths=[]
        D_values=[]
        for mock_idx in range(n_mocks):

            file = f"gamma_binary_closed_rebinned_{gamma_str}/model_{m}/neutral_gaps_z_mock_{mock_idx}.npz"

            sim_data = np.load(file)

            z_start = sim_data["z_start_array"]
            z_end   = sim_data["z_end_array"]

            sim_dz = z_end - z_start

            mask_min = sim_dz >= z_minimum

            z_start = z_start[mask_min]
            z_end   = z_end[mask_min]

            sim_mask = get_redshift_bin_mask(
                z_start, z_end, z_edges[ibin], z_edges[ibin+1]
            )

            lengths = [
                comoving_gap(z1,z2)
                for z1,z2 in zip(z_start[sim_mask], z_end[sim_mask])
            ]
            if len(lengths) > 0:
                mock_lengths.append(np.array(lengths))

            if len(lengths)==0:
                continue
            sim_lengths=np.array(lengths)
            zhu_lengths=zhu_lengths_bins[ibin]
            lx, ly = empirical_cdf(lengths)
            D, p = ks_2samp(zhu_lengths, sim_lengths)
            p_values.append(p)
            D_values.append(D)
            pvals_all[model_labels[imodel]][ibin].append(p)
            Dvals_all[model_labels[imodel]][ibin].append(D)
            all_cdfs.append((lx,ly))

        if len(p_values) > 0:
            median_p = np.median(p_values)
            frac_small = np.mean(np.array(p_values) < 0.05)

            print(f"Model {model_labels[imodel]}, Bin {ibin}")
            print(f"Median p-value = {median_p:.3f}")
            print(f"Fraction p<0.05 = {frac_small:.3f}")
        if len(all_cdfs)>0:

            L_grid = np.linspace(0,100,300)

            cdf_stack = []

            for lx,ly in all_cdfs:
                interp = np.interp(L_grid, lx, ly, left=0, right=1)
                cdf_stack.append(interp)

            cdf_stack = np.array(cdf_stack)
            median = np.median(cdf_stack, axis=0)
            low = np.percentile(cdf_stack,16,axis=0)
            high = np.percentile(cdf_stack,84,axis=0)
            ax.step(
                L_grid,
                median,
                '--',
                lw=1.5,
                color=bin_colors[ibin],
                label=f"{z_edges[ibin]:.3f} < z < {z_edges[ibin+1]:.3f}"
                )
           
            ax.step(
            lxzhu,
            lyzhu,
            '-',
            lw=2.5,
            color=bin_colors[ibin],
            label=f"Zhu {z_edges[ibin]:.2f}-{z_edges[ibin+1]:.2f}" if imodel==0 else None
            )
            ax.fill_between(
                L_grid,
                low,
                high,
                color=bin_colors[ibin],
                alpha=0.25
            )


    ax.set_title(model_labels[imodel])
    ax.set_xlim(0,100)
    ax.set_xlabel(r"$L\ (h^{-1}\mathrm{Mpc})$")
    ax.legend()

axes[0].set_ylabel(r"$CDF$")

plt.tight_layout()
plt.savefig(f"CDF_comparison.png")
plt.show()
fig, axes = plt.subplots(3,3, figsize=(12,12), sharex=True, sharey=True)
# p_min=1e-4
# p_max=1.0
# bins=np.logspace(np.log10(p_min),np.log10(p_max),40)


#============================================================
#                   Various Diagnostic tests
#=============================================================

for imodel, label in enumerate(model_labels):   #not matching , very low p-values.
    for ibin in range(3):

        ax = axes[imodel, ibin]
        pvals = np.array(pvals_all[label][ibin])
        # pvals=pvals[pvals>0]
        ax.hist(pvals,bins=21, range=(0,1), alpha=0.8)

        # ax.axvline(0.05, color="red", linestyle="--")

        ax.set_title(f"{label}, z-bin {ibin+1}")
        # ax.set_xlim(0,1)
        ax.set_xscale('log')

axes[2,1].set_xlabel("log(p-values)")
axes[1,0].set_ylabel("Number of mocks")
plt.tight_layout()
plt.savefig("pvalue_distribution.png")
plt.show()


#we need to compute Dn,m= supremum(CDF1-CDF2)
fig, axes = plt.subplots(3,3, figsize=(12,12), sharex=True, sharey=True)
D_min=1e-4
D_max=1.0
bins=np.logspace(np.log10(D_min),np.log10(D_max),40)

for imodel, label in enumerate(model_labels):
    for ibin in range(3):

       
        ax = axes[imodel, ibin]
        Dvals = np.array(Dvals_all[label][ibin])
        Dvals=Dvals[Dvals>0]
        ax.hist(Dvals, bins=bins,density=True, alpha=0.8)

        # ax.axvline(0.05, color="red", linestyle="--")

        ax.set_title(f"{label}, z-bin {ibin+1}")
        # ax.set_xlim(0,1)
        ax.set_xscale('log')

axes[2,1].set_xlabel("D_values")
axes[1,0].set_ylabel("Number of mocks")
plt.tight_layout()
plt.savefig("Dvalue_distribution.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,12))

x = np.arange(3)  # 3 redshift bins

for imodel, label in enumerate(model_labels):

    medians = []
    lows = []
    highs = []

    for ibin in range(3):
        Dvals = np.array(Dvals_all[label][ibin])

        median = np.median(Dvals)
        low = np.percentile(Dvals, 16)
        high = np.percentile(Dvals, 84)

        medians.append(median)
        lows.append(median - low)
        highs.append(high - median)

    yerr = [lows, highs]

    ax.errorbar(
        x,
        medians,
        yerr=yerr,
        
        fmt='o',
        capsize=4,
        elinewidth=2,
        label=label
    )

# Labels
ax.set_xticks(x)
ax.set_xticklabels([
    f"{z_edges[i]:.2f}-{z_edges[i+1]:.2f}" for i in range(3)
])

ax.set_ylabel("D_values")
ax.set_xlabel("Redshift bin")

ax.legend()
plt.tight_layout()
plt.savefig("Dmedian_vs_redshift.png")
plt.show()



fig, axes = plt.subplots(1,3, figsize=(12,4))

for imodel, label in enumerate(model_labels):   #not following diagnol : not matching 

    ax = axes[imodel]

    pvals = np.concatenate(pvals_all[label])
    p_sorted = np.sort(pvals)
    ecdf = np.arange(1,len(p_sorted)+1)/len(p_sorted)

    ax.step(p_sorted, ecdf)
    ax.plot([0,1],[0,1],'k--')

    ax.set_title(label)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

plt.tight_layout()
plt.show()


pvals_mockmock = []
Dvals_mockmock= []
for i, j in combinations(range(len(mock_lengths)), 2):

    D, p = ks_2samp(mock_lengths[i], mock_lengths[j])
    pvals_mockmock.append(p)
    Dvals_mockmock.append(D)

pvals_mockmock = np.array(pvals_mockmock)
Dvals_mockmock= np.array(Dvals_mockmock)




p_sorted = np.sort(pvals_mockmock)
ecdf = np.arange(1, len(p_sorted)+1)/len(p_sorted)

plt.figure()
# no bias in the mock-mock comparisons
plt.step(p_sorted, ecdf, label="Mock vs Mock")
plt.plot([0,1],[0,1],'k--', label="Uniform")

plt.xlabel("p-value")
plt.ylabel("ECDF")

plt.legend()
plt.show()