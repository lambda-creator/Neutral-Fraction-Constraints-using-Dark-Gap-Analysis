import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from pandas import read_csv
import os
#calling function:

"""
This code will calculate the PDF using KDE for different redshift bins as a function of mean photoionization rate.

To get a quantitative view of the dark gaps, we quantify PDF as our next tool: 

Steps to define the PDF:
    1) computing the KDE based on an extended Scott Rule to account for broadness of the dark gaps in higher z.
    2) Assuming only intermmediate scales : we do a masking of pdf_zhu/ observations < 0.1
    3) We finally store all the chi^2 values at the bin centers which will be further used in \chi^2 minimization.
    4) Interpolate the \Gamma values to get an optimized value for the minimization \chi^2 curve.

"""

def compute_kde(arr, bw, x_grid=None):  #computing the kde
    x = np.log10(arr)
    x = x[np.isfinite(x)]
    n = len(x)

    # Safety: KDE needs at least 2 points
    if n < 2:
        if x_grid is None:
            x_grid = np.linspace(-5, 1, 300)
        return x_grid, np.full_like(x_grid, np.nan, dtype=float)

    # grid 
    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 300)

    # bandwidth choice 
    if bw == "scott_3pt5":
        bw_factor = 3.5 * n**(-1/3)
        kde = gaussian_kde(x, bw_method=bw_factor)
    elif bw == "scott":
        kde = gaussian_kde(x, bw_method="scott")
    elif bw == "silverman":
        kde = gaussian_kde(x, bw_method="silverman")
    else:
        kde = gaussian_kde(x, bw_method=bw)

    kde_pdf = kde(x_grid)
    kde_pdf /= np.trapezoid(kde_pdf, x_grid)   #normalization

    return x_grid, kde_pdf


def get_redshift_bin_mask(z_start, z_end, bin_start, bin_end,dz_min=0.00315):  #redshift bin 
    z_mid = 0.5 * (z_start + z_end)
    dz = z_end - z_start
    return (z_mid >= bin_start) & (z_mid <= bin_end) & (dz >= dz_min) 



def chi2_func(pdf_simulation,pdf_obs,sigma, sigma_mask):
        residual=pdf_simulation[sigma_mask]-pdf_obs[sigma_mask]
        chi2=np.sum(residual**2 /(sigma[sigma_mask]**2))
        dof=np.sum(sigma_mask) -1 
        return chi2,dof
#parameters:

f_gamma=[0.3,0.4,0.5,0.6,1.0,1.5,2.0,3.0]
model_names=["Ultra Late Model","Late Model","Early Model"]
model_indices=[0,12,25]
z_minimum= 0.00314884   # based on the minimum criteria of dz> 150 km/s at z0=5.67 i.e 1h^-1 Mpc
n_mocks=45
n_bins=3

#observation file:

zhu_obs=read_csv('gap_data/zhu_ly_alpha_dark_gaps.csv')
zhu_z_start= zhu_obs.iloc[:,2]
zhu_z_end=zhu_obs.iloc[:,3]
zhu_dz=zhu_z_end-zhu_z_start

mask_min= zhu_dz >= z_minimum   #bool dz > 1h^-1Mpc  

zhu_dz= zhu_dz[mask_min]
zhu_start=zhu_z_start[mask_min]
zhu_end=zhu_z_end[mask_min]


x_obs,zhu_kde=compute_kde(zhu_dz,bw="scott_3pt5")  # x_obs is in log scale
x_obs1,zhu_kde1=compute_kde(zhu_dz,bw="silverman")
x_obs2,zhu_kde2=compute_kde(zhu_dz,bw="scott")
plt.plot(x_obs,zhu_kde,label="3pt5")
plt.plot(x_obs1,zhu_kde1,label="silverman")
plt.plot(x_obs2,zhu_kde2,label="scott")
plt.legend()  
plt.show()

#masking parameters:

mask_chi2= (zhu_kde >= 0.1)

# Bin 1: z = [5.0100, 5.3440], z_center = 5.202, Δz = 0.334, N = 246
# Bin 2: z = [5.3440, 5.5652], z_center = 5.457, Δz = 0.221, N = 246
# Bin 3: z = [5.5652, 5.9725], z_center = 5.711, Δz = 0.407, N = 245

z_edges = np.array([5.0100, 5.3440, 5.5652, 5.9725])

results={}
for ibin in range(n_bins):
    zhu_dz_bin=zhu_dz[get_redshift_bin_mask(zhu_start,zhu_end,z_edges[ibin],z_edges[ibin+1])]
    x_common_grid,zhu_kde_bin= compute_kde(zhu_dz_bin,'scott')
    
   

    for gamma_idx,gamma_val in enumerate(f_gamma):
       
        gamma_str = str(gamma_val).replace('.', 'pt')
        file_template=f"gamma_binary_closed_rebinned_{gamma_str}/model_{{}}/neutral_gaps_z_mock_{{}}.npz"
        for model_idx, model_val in enumerate(model_indices):
            pdf_arr = np.full((n_mocks, len(x_common_grid)), np.nan)

            # x_arr = np.zeros((n_mocks, len(x_common_grid)), dtype=int)
            for mock_idx in range(n_mocks):

                path=file_template.format(model_val,mock_idx)
                data=np.load(path)
                z_start=data["z_start_array"]
                z_end=data["z_end_array"]
                dz=z_end-z_start

                sim_mask_min = dz >= z_minimum
                dz = dz[sim_mask_min]
                z_start = z_start[sim_mask_min]
                z_end = z_end[sim_mask_min]



                sim_mask=get_redshift_bin_mask(z_start,z_end,z_edges[ibin],z_edges[ibin+1])
                dz=dz[sim_mask]
                _,kde_sim=compute_kde(dz,"scott_3pt5",x_grid=x_common_grid)
                pdf_arr[mock_idx,:]=kde_sim
                # x_arr[mock_idx,:]=x_common_grid

            mean_pdf=np.nanmean(pdf_arr,axis=0)
            std_pdf=np.nanstd(pdf_arr,axis=0,ddof=1)
            
            results.setdefault(ibin, {})
            results[ibin].setdefault(gamma_idx, {})

            results[ibin][gamma_idx][model_val]={
                    "pdf_arr":pdf_arr,
                    "std_dev":std_pdf,
                    "mean_pdf": mean_pdf,
                    "x_grid": x_common_grid

                }

                
chi2_table = []   # store results rows


for ibin in range(n_bins):
    zhu_dz_bin = zhu_dz[get_redshift_bin_mask(zhu_start, zhu_end, z_edges[ibin], z_edges[ibin+1])]
    x_grid, pdf_obs = compute_kde(zhu_dz_bin, "scott_3pt5")

    # hard tail-cut
    mask = pdf_obs >= 0.1

    for gamma_idx, gamma_val in enumerate(f_gamma):
        for model_val in model_indices:
            pdf_mod   = results[ibin][gamma_idx][model_val]["mean_pdf"]
            sigma_mod = results[ibin][gamma_idx][model_val]["std_dev"]

            # avoid division by zero
            sigma = np.maximum(sigma_mod, 1e-6)

            chi2, dof = chi2_func(pdf_mod, pdf_obs, sigma, mask_chi2)
            chi2_table.append((ibin, gamma_val, model_val, chi2, dof, chi2/dof))
for ibin in range(n_bins):
    for model_val in model_indices:
        rows = [r for r in chi2_table if r[0] == ibin and r[2] == model_val]
        best = min(rows, key=lambda r: r[3])
        print(f"bin {ibin+1}, model {model_val}: best gamma={best[1]}, chi2/dof={best[5]:.3f}")



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

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os

# Using log gamma interpolation

def interp_pdf_over_gamma(gammas, pdf_stack, gamma_fine, kind="linear"):
    """
    gammas: (Ng,)
    pdf_stack: (Ng, Nx)  [pdf values at each gamma]
    gamma_fine: (Ngfine,)
    """
    xg = np.log10(gammas)
    xg_fine = np.log10(gamma_fine)

    f = interp1d(xg, pdf_stack, axis=0, kind=kind, bounds_error=False)
    return f(xg_fine)  # shape (Ngfine, Nx)

def chi2_from_pdf(pdf_mod, pdf_obs, sigma, mask):
    good = mask & np.isfinite(pdf_mod) & np.isfinite(pdf_obs) & np.isfinite(sigma) & (sigma > 0)
    chi2 = np.sum(((pdf_mod[good] - pdf_obs[good]) / sigma[good])**2)
    dof = np.sum(good) - 1
    return chi2, dof


# output directory

outdir = "chi2_from_pdf_interp"
os.makedirs(outdir, exist_ok=True)


model_label = {0: "Ultra Late", 12: "Late", 25: "Early"}

# fine gamma grid in log space :

gamma_fine = np.logspace(np.log10(min(f_gamma)), np.log10(max(f_gamma)), 250)

for ibin in range(n_bins):

    # --- obs PDF for this bin ---
    zhu_dz_bin = zhu_dz[get_redshift_bin_mask(zhu_start, zhu_end, z_edges[ibin], z_edges[ibin+1])]
    x_grid, pdf_obs = compute_kde(zhu_dz_bin, "scott_3pt5")

    # hard-tail cut mask 
    mask = pdf_obs >= 0.1

    plt.figure()

    for model_val in model_indices:

        # stack PDFs across discrete gammas
        gammas = np.array(f_gamma, dtype=float)

        pdf_stack = np.array([results[ibin][gidx][model_val]["mean_pdf"]
                              for gidx in range(len(f_gamma))])   # (Ng, Nx)

        sig_stack = np.array([results[ibin][gidx][model_val]["std_dev"]
                              for gidx in range(len(f_gamma))])   # (Ng, Nx)

      
        pdf_fine = interp_pdf_over_gamma(gammas, pdf_stack, gamma_fine, kind="linear")
        sig_fine = interp_pdf_over_gamma(gammas, sig_stack, gamma_fine, kind="linear")

        chi2_fine = np.zeros(len(gamma_fine))
        dof_fine  = np.zeros(len(gamma_fine))

        for k in range(len(gamma_fine)):
            sigma = np.maximum(sig_fine[k], 1e-6)  # safety floor
            chi2_fine[k], dof_fine[k] = chi2_from_pdf(pdf_fine[k], pdf_obs, sigma, mask)

        chi2_red_fine = chi2_fine / dof_fine

        kbest = np.argmin(chi2_fine)
        best_gamma = gamma_fine[kbest]

      
        plt.plot(gamma_fine, chi2_red_fine,
                 label=f"{model_label.get(model_val, model_val)} best fΓ={best_gamma:.2f}")

    plt.xscale("log")
    plt.xlabel(r"$f_\Gamma$")
    plt.ylabel(r"$\chi^2/\mathrm{dof}$")
    plt.yscale('log')
    plt.title(f"{z_edges[ibin]:.3f} < z < {z_edges[ibin+1]:.3f}")
    plt.legend()


    plt.savefig(f"{outdir}/chi2red_fromPDFinterp_bin{ibin+1}_pt1.pdf")
    plt.show()
