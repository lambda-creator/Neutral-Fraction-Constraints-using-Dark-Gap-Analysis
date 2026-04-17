## Data Description

This dataset contains **dark gaps identified along synthetic quasar sightlines** generated using the **EX-CITE radiative transfer simulations** (Gaikwad et al. 2023).

The EX-CITE model produces **45 realizations of 44 quasar sightlines**. Dark gaps are identified following a methodology similar to that used in Zhu et al. 2021.

---

## Methodology

Dark gaps are defined using the following procedure:

1. The normalized flux ($F_{\mathrm{norm}}$) is rebinned to **$1,h^{-1},\mathrm{Mpc}$** resolution. Regions with
   $F_{\mathrm{norm}} \leq 0.05$ are identified.

2. **Contiguous regions** below this threshold are classified as dark gaps.
   The minimum gap length is set to **$1,h^{-1},\mathrm{Mpc}$**.

3. Regions within **$7,\mathrm{pMpc}$** of the quasar (proximity zone) are excluded.
   The **Lyman-$\beta$ limit** is used as the blueward boundary.

4. The mean photoionization rate is rescaled using:

   ```
   log_Gamma = [0.3, 0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
   ```

   This allows exploration of different neutral hydrogen fractions.

5. Three reionization models are considered, defined by the redshift at which reionization is 50% complete ($z_{50}$):

   * Early: $z_{50} = 8.35$
   * Fiducial (Late): $z_{50} = 7.20$
   * Ultra-Late: $z_{50} = 6.15$

6. For each model and realization, dark gaps are identified and stored in files named as:

   ```
   gamma_rebinned_0pt3
   ```

   corresponding to different photoionization scaling values.

7. The parameter mapping of $\langle \Gamma_{\mathrm{HI}} \rangle - \lambda_{0}$ to $\langle \Gamma_{\mathrm{HI}} \rangle - \lambda_{\mathrm{mfp}}$ is contained in the the files named as: 
   ```
   fitted_z_5.11 
   ```


---

## Data Format

Each file contains dark gap measurements extracted from individual sightlines.
(Describe columns here if possible — e.g., $z_{\mathrm{start}}, z_{\mathrm{end}},$ gap length, etc.)

---

## Availability

The full simulation data (e.g., synthetic spectra, density fields, and fluctuation maps) are part of an ongoing collaboration and are **not publicly available**.

However, **processed dark gap datasets are included** in this repository to demonstrate the analysis pipeline.

For access to the full simulation outputs, please refer to the original publication (Gaikwad et al. 2023) or contact the respective authors.
