import numpy as np
import matplotlib.pyplot as plt
from read_spec_ewald_script import spectra
from scipy import special
from astropy import constants as const
import astropy.units as u
import h5py
import time
from mpi4py import MPI


"""
This code uses general  ly alpha optical depth generation code to generate sightlines across 80 h^-1 Mpc box and 2048 pixels.
This code is a specialized optical depth code which considers the fluctuations of the photoionization and evolving mean free path.

"""

# =========================
# MPI setup
# =========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =========================
# File paths
# =========================
filename_los = r"\data\curie3\PRACE_ra1865\planck1_40_2048\los\los2048_n5000_z5.200.dat"
filename_tau = r"\data\curie3\PRACE_ra1865\planck1_40_2048\los\tau2048_n5000_z5.200.dat"

spec_obj = spectra("se_onthefly", filename_los, taufilename=filename_tau)
# Note:
# External module for reading simulation data (LOS, spectra, etc.).
# Not included in this repository.

# =========================
# Cosmology parameters
# =========================
L_box = 80.0  # h^-1 Mpc
N_pixels = 2048

omega_matter = spec_obj.om
omega_energy = spec_obj.ol
omega_bar = spec_obj.ob
h = spec_obj.h
H0 = spec_obj.H0

Mpc2cm = (1 * u.Mpc).to(u.cm).value
H_mpc = 67.8  # km/s/Mpc

# =========================
# Physical constants
# =========================
mp = const.m_p.cgs.value
kb = const.k_B.cgs.value
c = const.c.cgs.value
c_kms = 2.998e5

I_alpha = 4.45e-18
lamda_alpha = 1215.67e-8
Gamma_lya = 6.265e8  # damping constant for Ly-alpha line

# =========================
# HDF5 LOS fields
# =========================
filename = r"\Common_LOS_Fields_For_Optical_Depth.hdf5"   #los information

delta_path = "/N_GRID_LOS_TAU-2048/Delta_Field_For_Optical_Depth"
vel_path = "/N_GRID_LOS_TAU-2048/Velocity_Field_For_Optical_Depth"
redshift_path = "/N_GRID_LOS_TAU-2048/Redshift_For_Optical_Depth"

with h5py.File(filename, "r") as f:
    vel = f[vel_path][:]
    redshift = f[redshift_path][:]
    delta = f[delta_path][:]

    if rank == 0:
        print("Shape of delta field:", delta.shape)
        print("Shape of redshift array:", redshift.shape)
        print("Shape of velocity field:", vel.shape)

loglambda = np.array([
    -1.4, -1.3, -1.2, -1.1, -1.0, -0.9,
    -0.8, -0.7, -0.6, -0.5,
    -0.4, -0.3, -0.2, -0.1, 0.0,
     0.1,  0.2,  0.3,  0.4,  0.5,  0.6,
     0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,
     1.4,  1.5,  1.6,  1.7,  1.8,
     1.9,  2.0,  2.1,  2.2,  2.3,  2.4,  2.5
], dtype=float)

# =========================
# Redshift grid
# =========================
zmean = 5.1835

# =========================
# Photoionization rate and mean free path
# =========================
mfp = 41.759
J = 5.9799e-13

idx = np.abs(loglambda - np.log10(mfp)).argmin()
nearest_value = loglambda[idx]
lambda_0_str = f"{nearest_value:.3f}"
print(lambda_0_str)
z1=5.109794
z2 = 5.256697

Gamma_fluctuations_path_lower = (
    rf"\Gamma_fluctuations\z-{z1}"
    rf"\N_Grid_Gamma_HI-512\MFP_Log_Lambda_0-{lambda_0_str}\Gamma_XI_Field_3D.npy"
)


Gamma_3D_lower = np.load(Gamma_fluctuations_path_lower).astype(np.float32)
Gamma_3D_lower /= np.mean(Gamma_3D_lower)  # global normalization

Gamma_fluctuations_path_upper = (
    rf"\Gamma_fluctuations\z-{z2}"
    rf"\N_Grid_Gamma_HI-512\MFP_Log_Lambda_0-{lambda_0_str}\Gamma_XI_Field_3D.npy"
)


Gamma_3D_upper = np.load(Gamma_fluctuations_path_upper).astype(np.float32)
Gamma_3D_upper /= np.mean(Gamma_3D_upper)  # global normalization

# =========================
# Interpolating Gamma fluctuations in redshift (log space)
# =========================


z_target = zmean

w = (z_target - z1) / (z2 - z1)

Gamma_3D_lower = np.log(Gamma_3D_lower)
Gamma_3D_upper = np.log(Gamma_3D_upper)

# interpolation
Gamma_3D = (1 - w) * Gamma_3D_lower + w * Gamma_3D_upper

# freeing memory early
del Gamma_3D_lower
del Gamma_3D_upper

# exponentiate
Gamma_3D = np.exp(Gamma_3D)

# normalization
Gamma_3D /= np.mean(Gamma_3D)

if rank == 0:
    print(f"Interpolated Gamma field at z = {z_target:.3f}")


# =========================
# Temperature
# =========================

T0 = 11000.0
gamma = 1.2

def Hubble_mpc(z):
    return H_mpc * np.sqrt(omega_matter * (1 + z)**3 + omega_energy)

delta_z = (Hubble_mpc(zmean) / c_kms) * (L_box / h)
print("The redshift interval between the boxes is", delta_z)

z_low = zmean - delta_z / 2
z_high = zmean + delta_z / 2

zidx_low = np.argmin(np.abs(redshift - z_low))    #extracting that section of the redshift which spans from z_low, z_high
zidx_high = np.argmin(np.abs(redshift - z_high))

z_grid = redshift[zidx_low:zidx_high]
dz = z_grid[1] - z_grid[0]

# =========================
# Extract LOS data
# =========================
vel_grid = vel[zidx_low:zidx_high, 0:1980] * 1e5
delta_grid = delta[zidx_low:zidx_high, 0:1980]

n_grids = vel_grid.shape[0]
n_los = vel_grid.shape[1]

# Coordinate grids for interpolation
x_gamma = np.linspace(0.0, L_box, Gamma_3D.shape[0])
x_los = np.linspace(0.0, L_box, n_grids)

# =========================
# Temperature grid
# =========================

T = T0 * np.power(delta_grid, gamma - 1.0)

# =========================
# Doppler parameter
# =========================

b_grid = np.sqrt(2.0 * kb * T / mp)

# =========================
# Hydrogen / ionization parameters
# =========================

Y = 0.24
ue = 4.0 / (4.0 - 3.0 * Y)
up = 2.0 / (2.0 - Y)
alpha = 4.36e-10 * T**(-0.7)
ne_factor = 1 + Y / (4 * (1 - Y))   # consistent with MFP code

# =========================
# Cosmology function
# =========================
def H(z):
    return H0 * np.sqrt(omega_matter * (1 + z)**3 + omega_energy)

# =========================
# Voigt profile
# =========================

def voigt(delta_z, z_abs, b, vpec):
    B = ((delta_z) / (1 + z_abs) * c / b) + (vpec / b)
    A = (lamda_alpha * Gamma_lya) / (4.0 * np.pi * b)
    return np.real(special.wofz(B + 1j * A))

# =========================
# Optical depth integrand
# =========================
def optical_depth(z, delta_z, z_abs, b, nH, vpec):
    return (
        (c / H(z))
        * nH
        * I_alpha
        * voigt(delta_z, z_abs, b, vpec)
        * (c / (1 + z))
        * 1.0 / (b * np.sqrt(np.pi))
    )

# =========================
# Spectrum computation
# =========================
def compute_spectrum(ngrids, zmin, zmax, zgrid, b, nH, vpec):
    tau_arr = np.zeros(ngrids)
    dz_local = zgrid[1] - zgrid[0]
    width = zmax - zmin

    for i in range(ngrids):
        sum_tau = 0.0
        for j in range(ngrids):
            delta_z = zgrid[j] - zgrid[i]

            if abs(delta_z) > width / 2.0:
                if delta_z > 0:
                    delta_z = -(width - delta_z)
                else:
                    delta_z = width + delta_z

            sum_tau += optical_depth(
                zgrid[j],
                delta_z,
                zgrid[i],
                b[j],
                nH[j],
                vpec[j]
            ) * dz_local

        tau_arr[i] = sum_tau

    return tau_arr

# =========================
# MPI LOS distribution
# =========================

z_min = z_low
z_max = z_high
v_pec = vel_grid

los_indices = np.arange(n_los)
local_indices = np.array_split(los_indices, size)[rank]

tau_local = np.zeros((n_grids, len(local_indices)))

start_time = time.time()

for k, los_idx in enumerate(local_indices):
    Nx = Gamma_3D.shape[1]
    Ny = Gamma_3D.shape[2]

    x = np.random.randint(0, Nx)
    y = np.random.randint(0, Ny)

    # Extracting Gamma along LOS and interpolating to the LOS grid
    Gamma_LOS = Gamma_3D[x, y, :]
    Gamma_interp = np.interp(x_los, x_gamma, Gamma_LOS)
   
    Gamma_interp = np.clip(Gamma_interp, 1e-6, None)


    # Hydrogen number density on this LOS
    nH = 2e-7 * delta_grid[:, los_idx] * (1 + z_grid)**3
    


    # Neutral fraction and neutral hydrogen density

    alpha_LOS = alpha[:, los_idx]
    xHI_LOS = alpha_LOS * ne_factor * nH**2 / (J * Gamma_interp)  # cm^-3
    # n_HI = ue * up * nH * alpha_LOS / (J * Gamma_interp)
    # xHI_LOS = n_HI * nH
    xHI_LOS=np.minimum(xHI_LOS,nH)
    tau_local[:, k] = compute_spectrum(
        n_grids,
        z_min,
        z_max,
        z_grid,
        b_grid[:, los_idx],
        xHI_LOS,
        v_pec[:, los_idx]
    )

end_time = time.time()
print(f"Rank {rank} finished in {end_time - start_time:.2f} seconds")

# =========================
# Gather results
# =========================
tau_gather = comm.gather(tau_local, root=0)
index_gather = comm.gather(local_indices, root=0)

# =========================
# Reconstruct final array
# =========================
if rank == 0:
    tau_arr = np.zeros((n_grids, n_los))

    for idxs, part in zip(index_gather, tau_gather):
        tau_arr[:, idxs] = part

    print("Final tau shape:", tau_arr.shape)

    np.savetxt(f"Optical_Depth_Best_Fit_correct_version_{z_min:.2f}_{z_max:.2f}.txt", tau_arr)

    plt.plot(z_grid, np.exp(-tau_arr[:, 0]), label="LOS 1")
    plt.xlabel("Redshift")
    plt.ylabel("Transmitted Flux")
    plt.legend()
    plt.show()
