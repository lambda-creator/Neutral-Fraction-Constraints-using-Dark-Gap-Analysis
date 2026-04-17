from mpi4py import MPI
import numpy as np
import os
from scipy.optimize import curve_fit

"""
The role of this code is to find out the true mean free path in the IGM at various redshift bins.

As mentioned previously, we cover a 40 Mpc interval of the redshifts 
from z=5.0 to z= 6.0: (5.110, 5.256,5.413,5.579,5.756,5.946)
This code uses 108 x 100 \Gamma-\lambda_0 fluctuation models:
Steps to find out the true mean free path for each model and redshift bin: 

1)Shoot a large number of sightlines and find out the mean photoionization rate across all the sightlines.
2)Use the mean photoionization rate to find the Lyman continuum optical depth by following formula:
\tau_{LC}= \int n_{HI} \sigma_{HI} dx ; where dx is the physical length; \sigma_{HI}= 6.34e-18 cm^2 and
n_{HI}= \alpha n_e n_H^2/ (\Gamma_{mean}).
3) Find the LyC flux as <F> = <np.exp(-1.0 * \tau_{LC})> 
4)Take the average of the LyC flux along all the sightlines. This will be similar 
as stacking the profiles at 912 \AA.
5) Fit the stacked flux with an exponential to obtain the true mean free path as: 
F=F0 np.exp(-dx/ (\lambda_{mfp})); 
6) But if flux is non-zero only for pixels< 5: consider using the first value of F= F0/(np.e)
 point as the true mfp.

Note: This code is run per redshift, change the redshift accordingly.
"""

# -------------------------
# MPI setup
# -------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------
# Load data (all ranks)
# -------------------------
z_mean = 5.109781      #Change the redshift accordingly;

delta_path = f"\\density_fields\\z-{z_mean}\\GRIDS\\N_GRID-512\\d3d_gas_512_z{z_mean:.3f}.npy"  
delta_field = np.load(delta_path)

outdir = f"fitted_z_{z_mean:.2f}"

if rank == 0:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"Output directory: {outdir}")
    print(f"Directory exists: {os.path.exists(outdir)}")

comm.Barrier()  # ensure directory exists before any rank tries to write

# -------------------------
# Constants
# -------------------------

h        = 0.678
L_box    = 80.0
Ng       = 512
Y        = 0.24
ne_factor = 1 + Y / (4 * (1 - Y))
sigma_HI  = 6.34e-18
dx_com    = L_box / Ng
dx_phys   = (dx_com / h) * (1 / (1 + z_mean)) * 3.086e24

mean_Gamma_vals = np.logspace(-15.0, -10.0, 100)
no_of_skewers   = 2048


lambda_0_values = [-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,
                   -0.8,-0.7,-0.6,-0.5,
                   -0.4,-0.3,-0.2,-0.1,0.0,
                    0.1,0.2,0.3,0.4,0.5,0.6,
                    0.7,0.8,0.9,1.0,1.1,1.2,1.3,
                    1.4,1.5,1.6,1.7,1.8,
                    1.9,2.0,2.1,2.2,2.3,2.4,2.5]

low_temp_threshold = 10**(-13.6)
s_comoving_hinvMpc = np.arange(Ng) * dx_com

def lyc_profile(x, F0, lambda_mfp):
    return F0 * np.exp(-x / lambda_mfp)

# -------------------------
# Split skewer workload
# -------------------------

skewer_indices = np.arange(no_of_skewers)
local_indices  = np.array_split(skewer_indices, size)[rank]

# -------------------------
# Main loop
# -------------------------

for lambda_0 in lambda_0_values:

    lambda_0_str = f"{lambda_0:.3f}"

    if rank == 0:
        print(f"Processing lambda_0 = {lambda_0:.3f}...")

    gamma_path = (
        f"\\Gamma_fluctuations\\z-5.109794"
        f"\\N_Grid_Gamma_HI-512\\MFP_Log_Lambda_0-{lambda_0_str}\\Gamma_XI_Field_3D.npy"
    )

    gamma_field = np.load(gamma_path)
    gamma_field /= np.mean(gamma_field)

    lambda_mfp_results = []

    for gamma_idx, mean_Gamma in enumerate(mean_Gamma_vals):   # enumerate here

        F_local = np.zeros(Ng)

        for i in local_indices:

            x = np.random.randint(0, Ng)
            y = np.random.randint(0, Ng)

            Gamma_skewer = mean_Gamma * gamma_field[x, y, :]
            Gamma_skewer = np.maximum(Gamma_skewer, 1e-20)

            Delta_skewer = delta_field[x, y, :]
            nH = 2e-7 * Delta_skewer * (1 + z_mean)**3

            T0 = np.where(Gamma_skewer >= low_temp_threshold, 10000.0, 100.0)
            recombination_rate = 4.36e-10 * T0**(-0.7)

            xHI = recombination_rate * ne_factor * nH / Gamma_skewer
            xHI = np.minimum(xHI, 1.0)

            nHI = xHI * nH

            tau = np.cumsum(nHI * sigma_HI * dx_phys)
            F   = np.exp(-tau)

            F_local += F

        # -------------------------
        # Reduce across ranks
        # -------------------------
        F_total = np.zeros_like(F_local)
        comm.Reduce(F_local, F_total, op=MPI.SUM, root=0)

        if rank == 0:

            F_mean = F_total / no_of_skewers

            # --- Save flux profile every 10th Gamma ---
            if gamma_idx % 10 == 0:
                np.savetxt(
                    f'{outdir}\\flux_profile_lambda_{lambda_0:.3f}_Gamma_{mean_Gamma:.3e}.txt',
                    np.column_stack((s_comoving_hinvMpc, F_mean)),
                    header='x[h^-1_cMpc]  F_mean',
                    fmt='%.6e'
                )

            # --- Fitting ---

            mask  = F_mean > 1e-6
            s_fit = s_comoving_hinvMpc[mask]
            F_fit = F_mean[mask]

            if len(s_fit) < 5:
                # 1/e method
                F0_raw = F_mean[0] if F_mean[0] > 1e-6 else np.nan
                if np.isnan(F0_raw):
                    lambda_mfp_results.append(np.nan)
                else:
                    threshold = F0_raw / np.e
                    below = np.where(F_mean <= threshold)[0]
                    if len(below) == 0 or below[0] == 0:
                        lambda_mfp_results.append(np.nan)
                    else:
                        idx = below[0]
                        x0, x1 = s_comoving_hinvMpc[idx-1], s_comoving_hinvMpc[idx]
                        f0, f1 = F_mean[idx-1], F_mean[idx]
                        x_cross = x0 + (threshold - f0) * (x1 - x0) / (f1 - f0)
                        lambda_mfp_results.append(x_cross)
            else:
                try:
                    p0     = [1.0, L_box / 2.0]
                    bounds = ([0.0, dx_com], [1.0, 1e4])
                    popt, _ = curve_fit(
                        lyc_profile, s_fit, F_fit,
                        p0=p0, bounds=bounds, maxfev=10000
                    )
                    lambda_mfp_results.append(popt[1])

                except (RuntimeError, ValueError):
                    # 1/e fallback
                    F0_raw = F_fit[0]
                    threshold = F0_raw / np.e
                    below = np.where(F_mean <= threshold)[0]
                    if len(below) == 0 or below[0] == 0:
                        lambda_mfp_results.append(np.nan)
                    else:
                        idx = below[0]
                        x0, x1 = s_comoving_hinvMpc[idx-1], s_comoving_hinvMpc[idx]
                        f0, f1 = F_mean[idx-1], F_mean[idx]
                        x_cross = x0 + (threshold - f0) * (x1 - x0) / (f1 - f0)
                        lambda_mfp_results.append(x_cross)

    # -------------------------
    # Save mfp_vs_Gamma (only rank 0)
    # -------------------------
    if rank == 0:
        lambda_mfp_arr = np.array(lambda_mfp_results)
        np.savetxt(
            f'{outdir}\\mfp_vs_Gamma_lambda_{lambda_0:.3f}_z{z_mean:.3f}_large.txt',
            np.column_stack((mean_Gamma_vals, lambda_mfp_arr)),
            header='mean_Gamma[s^-1]  lambda_mfp[h^-1_cMpc]',
            fmt='%.6e'
        )

# -------------------------
# Done
# -------------------------
if rank == 0:
    print("All done!")