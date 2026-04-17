import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, binary_closing
import os
import h5py

# --- MNRAS-style formatting ---
plt.rcParams.update({
    'figure.figsize': (12, 12),
    'axes.facecolor': 'white',
    'axes.grid': False,
    'legend.numpoints': 1,
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
    'xtick.major.pad': 7,
    'xtick.minor.pad': 7,
    'font.size': 35,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.family': 'serif',
    'font.sans-serif': ['Georgia'],
    'text.usetex': False,
    'image.cmap': 'jet',
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.dpi': 200,
    'legend.fancybox': True,
    'legend.frameon': True,
    'legend.edgecolor': "black"
})


'''
The following code sets up the automated dark gap identifier pipeline which automatically detects the dark gaps for regions
with F_{cont}< 0.05: Steps and all the technical details are given below :

'''

# =============================================================================
# COSMOLOGICAL & INSTRUMENTAL PARAMETERS
# =============================================================================

H0       = 100          # km/s/(h^-1 Mpc)
h        = 0.678
omega_m  = 0.302
omega_l  = 0.698

threshold    = 0.05     # flux threshold for dark-gap detection
snr          = 56.7     # median SNR of the observed sightlines (Zhu+2021)
sigma_noise  = 1.0 / snr

c_kms        = 2.998e5  # km/s
FWHM_kms     = 30.0     # XQR-30 instrumental FWHM  [km/s]
sigma_v      = FWHM_kms / 2.3548  # Gaussian sigma  [km/s]

alpha_rest_A = 1215.67  # Ly-alpha  [Å]

# File bins: overlapping redshift ranges used in the filenames
# Each tuple is (z_lo, z_hi) of the optical-depth file
file_bins = [
    (5.07, 5.30),
    (5.22, 5.45),
    (5.37, 5.62),
    (5.54, 5.80),
    (5.72, 5.98),
]

# Valid (non-overlapping) zone boundaries — one per file bin.
# Gaps are ONLY recorded if they fall entirely within [valid_lo, valid_hi].
# Together these make 5.110–5.946 without double-counting.
valid_edges = [5.110, 5.257, 5.413, 5.579, 5.756, 5.946]
valid_bins  = [(valid_edges[k], valid_edges[k+1]) for k in range(len(valid_edges)-1)]
# Result:
#  Bin 0 : file [5.07–5.30]  =>  valid [5.110–5.257]
#  Bin 1 : file [5.22–5.45]  =>  valid [5.257–5.413]
#  Bin 2 : file [5.37–5.62]  =>  valid [5.413–5.579]
#  Bin 3 : file [5.54–5.80]  => valid [5.579–5.756]
#  Bin 4 : file [5.72–5.98]  =>  valid [5.756–5.946]

model_names = ["Best_Fit", "Ultra_Late", "Early"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def H_z(z):
    """Hubble parameter [km/s/Mpc] at redshift z (flat LCDM)."""
    return H0 * np.sqrt(omega_m * (1.0 + z)**3 + omega_l)


def velocity_pixel_size(z_lo, z_hi, n_pix):
    """
    Mean comoving velocity width of one simulation pixel [km/s].
    dv = c * dz / (1+z_mean) / n_pix
    """
    dz     = z_hi - z_lo
    z_mean = 0.5 * (z_hi + z_lo)
    return c_kms * dz / ((1.0 + z_mean) * n_pix)


def mpc_per_pixel(z_lo, z_hi, n_pix):
    """
    Comoving size of one simulation pixel [h^-1 Mpc].
    Uses dl/dpix = c*dz / (H(z)*(1+z)) / n_pix
    """
    dz      = z_hi - z_lo
    z_mean  = 0.5 * (z_hi + z_lo)
    H_zmean = H_z(z_mean)
    # comoving dr = c dz / H(z) in Mpc, convert to h^-1 Mpc by *h
    dr_total_hinvMpc = (c_kms * dz / H_zmean) * h   # total path in h^-1 Mpc
    return dr_total_hinvMpc / n_pix                  # per pixel


def pixels_per_hinvMpc(z_lo, z_hi, n_pix):
    """Number of simulation pixels spanning 1 h^-1 Mpc."""
    return int(round(1.0 / mpc_per_pixel(z_lo, z_hi, n_pix)))


# =============================================================================
# STEP-BY-STEP DARK-GAP PIPELINE (applied per sightline)
# =============================================================================
def identify_dark_gaps(flux_raw, z_lo, z_hi, n_pix,
                       sigma_noise=sigma_noise,
                       sigma_v=sigma_v,
                       threshold=threshold,
                       min_gap_hinvMpc=1.0,
                       valid_lo=None,
                       valid_hi=None,
                       seed=None):
    """
    Full pipeline from raw simulation flux → dark-gap catalogue.

    Parameters
    ----------
    flux_raw         : 1-D array, length n_pix
        Transmission F = exp(-tau) from the simulation (no instrumental effects).
    z_lo, z_hi       : float
        Redshift edges of the FILE bin (may be wider than valid zone).
    n_pix            : int
        Number of pixels in the simulation slab.
    sigma_noise      : float
        1-sigma noise per rebinned pixel (= 1/SNR, Zhu+2021 convention).
    sigma_v          : float  [km/s]
        Gaussian sigma of the XQR-30 LSF (= FWHM/2.3548).
    threshold        : float
        Flux level below which a pixel is classified as "dark".
    min_gap_hinvMpc  : float  [h^-1 Mpc]
        Minimum contiguous dark length to count as a gap.
    valid_lo, valid_hi : float or None
        Non-overlapping valid zone boundaries. Gaps outside this range
        are discarded; gaps that straddle the boundary are clipped to it.
        If None, the full file bin range is used (no clipping).
    seed             : int or None
        RNG seed for reproducible noise realisations.

    Returns
    -------
    dict with keys:
        z_grid, wavelength  : pixel grids over the full file bin
        F_raw, F_conv       : flux before/after LSF convolution
        F_rebinned, F_obs   : flux after rebinning / after noise
        z_rebinned, wav_rebinned : centres of the 1 h^-1 Mpc blocks
        dark_mask           : boolean mask on rebinned pixels
        gaps                : list of dicts clipped to valid zone
        n_pix_per_mpc       : int, pixels per h^-1 Mpc
    """

    if seed is not None:
        np.random.seed(seed)

    z_mean   = 0.5 * (z_lo + z_hi)
    dv       = velocity_pixel_size(z_lo, z_hi, n_pix)       # km/s per pixel
    n_mpc    = pixels_per_hinvMpc(z_lo, z_hi, n_pix)        # pixels per 1 h^-1 Mpc

    # --- pixel grids ---
    z_grid     = np.linspace(z_lo, z_hi, n_pix)
    wavelength = alpha_rest_A * (1.0 + z_grid)

    # ------------------------------------------------------------------
    # STEP 1 — Convolve with XQR-30 LSF (Gaussian, sigma_v in km/s)
    # ------------------------------------------------------------------

    sigma_pix = sigma_v / dv                  # LSF sigma in pixels
    F_conv    = gaussian_filter1d(flux_raw, sigma_pix, mode='wrap')

    # ------------------------------------------------------------------
    # STEP 2 — Rebin to 1 h^-1 Mpc  (boxcar of width n_mpc pixels)
    # ------------------------------------------------------------------

    # uniform_filter1d computes a running mean; we then downsample
    # by taking every n_mpc-th pixel starting at the half-width offset
    # so each output sample is centred on an independent block.

    F_smoothed  = uniform_filter1d(F_conv, size=n_mpc, mode='nearest')
    half        = n_mpc // 2
    # Downsample: independent 1 h^-1 Mpc blocks

    rebin_idx   = np.arange(half, n_pix, n_mpc)
    F_rebinned  = F_smoothed[rebin_idx]          # shape: (n_blocks,)
    z_rebinned  = z_grid[rebin_idx]
    wav_rebinned = wavelength[rebin_idx]

    # ------------------------------------------------------------------
    # STEP 3 — Add Gaussian noise  (σ = 1/SNR per *rebinned* pixel)
    # ------------------------------------------------------------------
    # The noise per rebinned pixel improves by sqrt(n_mpc) because
    # we averaged n_mpc raw pixels. Pass sigma_noise as the *raw* pixel
    # noise and let the code handle the reduction, OR pass the
    # already-rebinned sigma if your SNR refers to the 1 h^-1 Mpc pixel.
    #
    # Zhu+2021 quote SNR per 1 h^-1 Mpc pixel → use sigma_noise directly.
    noise       = np.random.normal(0.0, sigma_noise, size=F_rebinned.shape)
    F_obs       = F_rebinned + noise
    F_obs       = np.clip(F_obs, -0.2, 1.2)

    # ------------------------------------------------------------------
    # STEP 4 — Dark-gap identification on rebinned+noised spectrum
    # ------------------------------------------------------------------
    # 4a. Threshold
    dark_mask = F_obs < threshold              # True where flux is "dark"

    # 4b. Binary closing with a kernel of 1 pixel to patch isolated
    #     bright spikes inside a gap (1-pixel closing, not n_mpc).
    #     Using n_mpc here would be far too aggressive.
    dark_mask = binary_closing(dark_mask, structure=np.ones(3))

    # 4c. Walk the mask and record contiguous dark runs
    min_gap_pix = int(np.ceil(min_gap_hinvMpc))   # in rebinned pixels

    gaps      = []
    in_gap    = False
    gap_start = None

    for j, is_dark in enumerate(dark_mask):
        if is_dark and not in_gap:
            gap_start = j
            in_gap    = True
        elif (not is_dark) and in_gap:
            gap_end = j - 1
            in_gap  = False
            _record_gap(gaps, gap_start, gap_end,
                        z_rebinned, wav_rebinned, min_gap_pix,
                        valid_lo=valid_lo, valid_hi=valid_hi)
    if in_gap:   # gap reaching the end of the array
        _record_gap(gaps, gap_start, len(dark_mask) - 1,
                    z_rebinned, wav_rebinned, min_gap_pix,
                    valid_lo=valid_lo, valid_hi=valid_hi)

    return dict(
        z_grid        = z_grid,
        wavelength    = wavelength,
        F_raw         = flux_raw,
        F_conv        = F_conv,
        F_obs         = F_obs,
        F_rebinned    = F_rebinned,
        z_rebinned    = z_rebinned,
        wav_rebinned  = wav_rebinned,
        dark_mask     = dark_mask,
        gaps          = gaps,
        n_pix_per_mpc = n_mpc,
    )


def _record_gap(gap_list, i_start, i_end, z_arr, wav_arr, min_pix,
                valid_lo=None, valid_hi=None):
    """
    Append a gap entry if it meets the minimum length criterion AND
    falls within the valid redshift zone [valid_lo, valid_hi].

    Gaps that partially overlap the valid zone are CLIPPED to it before
    the length check, so no rebinned pixel outside the valid zone is counted.
    """

    # --- valid-zone clipping (on rebinned pixel indices) ---
    if valid_lo is not None:
        # find first rebinned index inside valid zone
        lo_idx = np.searchsorted(z_arr, valid_lo, side='left')
        i_start = max(i_start, lo_idx)
    if valid_hi is not None:
        # find last rebinned index inside valid zone
        hi_idx = np.searchsorted(z_arr, valid_hi, side='right') - 1
        i_end = min(i_end, hi_idx)

    if i_start > i_end:
        return   # gap is entirely outside the valid zone after clipping

    length_hinvMpc = (i_end - i_start + 1)   # 1 unit = 1 h^-1 Mpc rebinned pixel
    if length_hinvMpc >= min_pix:
        gap_list.append(dict(
            z_start        = z_arr[i_start],
            z_end          = z_arr[i_end],
            wav_start_A    = wav_arr[i_start],
            wav_end_A      = wav_arr[i_end],
            length_hinvMpc = float(length_hinvMpc),
        ))


# =============================================================================
# DIAGNOSTIC PLOT  (one sightline)
# =============================================================================

def plot_pipeline_stages(result, model_name, z_lo, z_hi,
                         vz_lo=None, vz_hi=None, save=True):
    """Four-panel plot showing each stage of the pipeline."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=False)

    wav      = result['wavelength']
    wav_reb  = result['wav_rebinned']

    # valid-zone wavelengths for shading
    wav_vlo = alpha_rest_A * (1.0 + vz_lo) if vz_lo is not None else wav_reb[0]
    wav_vhi = alpha_rest_A * (1.0 + vz_hi) if vz_hi is not None else wav_reb[-1]

    def shade_invalid(ax):
        """Grey out the regions outside the valid zone."""
        if vz_lo is not None:
            ax.axvspan(wav_reb[0], wav_vlo, color='grey', alpha=0.15, lw=0)
        if vz_hi is not None:
            ax.axvspan(wav_vhi, wav_reb[-1], color='grey', alpha=0.15, lw=0)

    # Panel 0 — raw flux
    axes[0].plot(wav, result['F_raw'], color='steelblue', lw=1.2)
    axes[0].set_ylabel(r'$F_{\rm raw}$')
    axes[0].set_title("Step 1 input: raw simulation flux")
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 1 — after LSF convolution
    axes[1].plot(wav, result['F_raw'],  color='steelblue', lw=1, alpha=0.4, label='raw')
    axes[1].plot(wav, result['F_conv'], color='darkorange', lw=1.8, label='LSF convolved')
    axes[1].set_ylabel(r'$F$')
    axes[1].set_title("Step 1: XQR-30 convolution (FWHM = 30 km/s)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', fontsize=16)

    # Panel 2 — rebinned + noise
    axes[2].plot(wav_reb, result['F_rebinned'], color='green', lw=1.8,
                 label=r'rebinned (1 $h^{-1}$ Mpc)', drawstyle='steps-mid')
    axes[2].plot(wav_reb, result['F_obs'],      color='black', lw=1, alpha=0.7,
                 label='+ noise', drawstyle='steps-mid')
    axes[2].axhline(threshold, color='red', ls='--', lw=1.5, label=f'threshold = {threshold}')
    shade_invalid(axes[2])
    axes[2].set_ylabel(r'$F$')
    axes[2].set_title(r"Rebinning to 1 $h^{-1}$ Mpc + noise (SNR = %.1f)" % snr)
    axes[2].set_ylim(-0.15, 1.05)
    axes[2].legend(loc='upper right', fontsize=16)

    # Panel 3 — dark-gap mask
    axes[3].plot(wav_reb, result['F_obs'], color='black', lw=1,
                 drawstyle='steps-mid', label='observed flux')
    axes[3].axhline(threshold, color='red', ls='--', lw=1.5)
    for g in result['gaps']:
        axes[3].axvspan(g['wav_start_A'], g['wav_end_A'],
                        color='crimson', alpha=0.25)
    shade_invalid(axes[3])
    # mark the valid zone boundaries
    for ax in [axes[2], axes[3]]:
        if vz_lo is not None:
            ax.axvline(wav_vlo, color='navy', ls=':', lw=1.5, label='valid zone')
        if vz_hi is not None:
            ax.axvline(wav_vhi, color='navy', ls=':', lw=1.5)
    axes[3].set_ylabel(r'$F_{\rm obs}$')
    axes[3].set_xlabel(r'Wavelength ($\AA$)')
    axes[3].set_title(
        f"Step 4: dark gaps (≥1 $h^{{-1}}$ Mpc) "
    )
    axes[3].set_ylim(-0.15, 1.05)

    for ax in axes:
        ax.tick_params(which='both', direction='in', top=True, right=True)

    vstr = f"  valid=[{vz_lo:.3f}–{vz_hi:.3f}]" if vz_lo is not None else ""
    plt.suptitle(f"{model_name}   file=[{z_lo:.3f}–{z_hi:.3f}]{vstr}",
                 fontsize=20, y=1.01)
    plt.tight_layout()

    if save:
        fname = f"pipeline_stages_{model_name}_z{z_lo:.3f}_{z_hi:.3f}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname}")
    plt.show()
    plt.close()


# =============================================================================
# MAIN LOOP
# =============================================================================

make_diagnostic_plot = True   # set False to skip the example plot

model_name="Best_Fit"
for i, ((z_lo, z_hi), (vz_lo, vz_hi)) in enumerate(zip(file_bins, valid_bins)):

        filepath = (fr"\MFP_optical_depth_files\{model_name}\Optical_Depth_{model_name}_correct_version_{z_lo:.2f}_{z_hi:.2f}.txt")

        data = np.loadtxt(filepath)          # shape: (n_pix, n_sightlines)
        n_pix, n_sightlines = data.shape

        print(f"\n{model_name}  file=[{z_lo:.2f}-{z_hi:.2f}]  "
              f"valid=[{vz_lo:.3f}-{vz_hi:.3f}] : "
              f"{n_pix} pix x {n_sightlines} sightlines  "
              f"({pixels_per_hinvMpc(z_lo, z_hi, n_pix)} pix/h⁻¹Mpc)")

        z_start_list     = []
        z_end_list       = []
        length_list      = []
        sightline_id_list = []

        for i_sl in range(n_sightlines):
            flux_raw = np.exp(-data[:, i_sl])

            result = identify_dark_gaps(
                flux_raw, z_lo, z_hi, n_pix,
                sigma_noise = sigma_noise,
                sigma_v     = sigma_v,
                threshold   = threshold,
                valid_lo    = vz_lo,
                valid_hi    = vz_hi,
                seed        = i_sl,
            )

            # Diagnostic plot for first sightline of first bin of first model
            if make_diagnostic_plot and model_name == model_names[0] and i == 0 and i_sl == 0:
                plot_pipeline_stages(result, model_name, z_lo, z_hi,
                                     vz_lo=vz_lo, vz_hi=vz_hi, save=True)
                make_diagnostic_plot = False

            n_gaps = len(result['gaps'])
            for g in result['gaps']:
                z_start_list.append(g['z_start'])
                z_end_list.append(g['z_end'])
                length_list.append(g['length_hinvMpc'])
            # tag every gap found in this sightline with its sightline index
            sightline_id_list.extend([i_sl] * n_gaps)

        z_start_arr      = np.asarray(z_start_list)
        z_end_arr        = np.asarray(z_end_list)
        length_arr       = np.asarray(length_list)
        sightline_id_arr = np.asarray(sightline_id_list, dtype=int)

        print(f"  Total dark gaps in valid zone : {len(z_start_arr)}  "
              f"across {n_sightlines} sightlines")

        # --- save output keyed by the VALID zone, not the file bin ---
        outdir  = os.path.join("MFP_Dark_Gap_Output", model_name)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(
            outdir,
            f"dark_gaps_{model_name}_z{vz_lo:.3f}_{vz_hi:.3f}.npz"
        )
        np.savez(
            outfile,
            z_start_array      = z_start_arr,       # gap start redshift
            z_end_array        = z_end_arr,          # gap end redshift
            length_hinvMpc     = length_arr,         # gap length [h^-1 Mpc]
            sightline_id_array = sightline_id_arr,   # which sightline (0-indexed)
            n_sightlines       = n_sightlines,       # total sightlines in this bin
            valid_z_lo         = vz_lo,              # valid zone edges (metadata)
            valid_z_hi         = vz_hi,
        )
        print(f"  Saved: {outfile}")