## Results

This directory contains the **main outputs of the dark gap analysis**.

---

## Contents

The following results are included:

* **CDF plots** of dark gap distributions
* **$F_{10}$ statistics** (fraction of dark gaps above threshold)
* **Sightline visualization**
* **PDFs and $\chi^2$ analysis results**

---

## Directory Structure

The results are organized as follows:

```
chi2_directory/
    └── chi2red_fromPDFinterp_bin1_pt1
        (pt1 corresponds to masking with PDF_Zhu > 0.1 or 0.5)

kde_bin1/
    └── kde_pdf_gamma_0pt3_bin_1
```

* `chi2_directory/` contains reduced $\chi^2$ values computed using PDF interpolation.
* `kde_bin1/` contains kernel density estimation (KDE)–based PDFs for different $\Gamma$ scaling values.

---

## Redshift Bins

The analysis is performed over the following redshift bins:

* **Bin 1:**
  $z = [5.0100, 5.3440]$,
  $z_{\mathrm{center}} = 5.202$, $\Delta z = 0.334$, $N = 246$

* **Bin 2:**
  $z = [5.3440, 5.5652]$,
  $z_{\mathrm{center}} = 5.457$, $\Delta z = 0.221$, $N = 246$

* **Bin 3:**
  $z = [5.5652, 5.9725]$,
  $z_{\mathrm{center}} = 5.711$, $\Delta z = 0.407$, $N = 245$


---

## Neutral-Fraction-Upper-Bound Constraints

* **Ultra-Late Reionization Model**
  We present $1\sigma$ upper bound constraints on volume-averaged neutral fraction as:

  $\langle x_{\mathrm{HI}} \rangle \leq 7.9\times10^{-4}$, $1.423\times10^{-2}$, $8.398\times10^{-2}$, at $\bar{z} = 5.177$, $5.455$, $5.769$

* **Late (Fiducial) Reionization Model**
  We present $1\sigma$ upper bound constraints on volume-averaged neutral fraction as:

  $\langle x_{\mathrm{HI}} \rangle \leq 1.4\times10^{-4}$, $2.76\times10^{-3}$, $4.705\times10^{-2}$, at $\bar{z} = 5.177$, $5.455$, $5.769$ 

* **Early Reionization Model**
  We present $1\sigma$ upper bound constraints on volume-averaged neutral fraction as:

  $\langle x_{\mathrm{HI}} \rangle \leq 4.0\times10^{-5}$, $9.0\times10^{-5}$, $3.55\times10^{-3}$, at  $\bar{z} = 5.177$, $5.455$, $5.769$ 






---
## Notes
* **$x_{HI}$ upper bound constraints are model-dependent, indicating a strong degeneracy between reionization history and neutral fraction.**
* The $\chi^2$ values are computed by comparing simulated PDFs with observational constraints (e.g., Zhu et al. 2021).
* We choose threshold-cutoff (mask)=0.5 in the observational PDF to avoid the double minimum peak shown at threshold-cutoff (mask)=0.1. 
* File naming conventions encode:

  * redshift bin
  * $\log \Gamma_{\mathrm{HI}}$ scaling
  * masking thresholds applied during analysis
