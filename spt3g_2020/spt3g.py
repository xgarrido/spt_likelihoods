""".. module:: SPT-3G

:Synopsis: Definition of python-native CMB likelihood for SPT-3G TE+EE. Adapted from Fortran
likelihood code https://pole.uchicago.edu/public/data/dutcher21/SPT3G_2018_EETE_likelihood.tar.gz

:Author: Xavier Garrido

"""

import itertools
import os
import re
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

default_spectra_list = [
    "90_Ex90_E",
    "90_Tx90_E",
    "90_Ex150_E",
    "90_Tx150_E",
    "90_Ex220_E",
    "90_Tx220_E",
    "150_Ex150_E",
    "150_Tx150_E",
    "150_Ex220_E",
    "150_Tx220_E",
    "220_Ex220_E",
    "220_Tx220_E",
]


class SPT3GPrototype(InstallableLikelihood):
    install_options = {
        "download_url": "https://pole.uchicago.edu/public/data/dutcher21/SPT3G_2018_EETE_likelihood_v3.zip",
        "data_path": "spt3g_2018",
    }

    bibtex_file = "spt3g.bibtex"

    bin_min: Optional[int] = 1
    bin_max: Optional[int] = 44
    windows_lmin: Optional[int] = 1
    windows_lmax: Optional[int] = 3200

    aberration_coefficient: Optional[float] = -0.0004826
    super_sample_lensing: Optional[bool] = True

    poisson_switch: Optional[bool] = True
    dust_switch: Optional[bool] = True

    beam_cov_scaling: Optional[float] = 1.0

    spectra_to_fit: Optional[Sequence[str]] = default_spectra_list

    # SPT-3G Y1 EE/TE Effective band centres for polarised galactic dust.
    nu_eff_list: Optional[dict] = {90: 9.670270e01, 150: 1.499942e02, 220: 2.220433e02}

    data_folder: Optional[str] = "spt3g_2018/SPT3G_2018_EETE_likelihood_v3/data/SPT3G_Y1_EETE"
    bp_file: Optional[str]
    cov_file: Optional[str]
    beam_cov_file: Optional[str]
    calib_cov_file: Optional[str]
    window_dir: Optional[str]

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to SPTPol data. Set the likelihood property 'path' or "
                f"the common property '{packages_path_input}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                f"The 'data_folder' directory does not exist. Check the given path [{self.data_folder}].",
            )

        # Get likelihood name and add the associated mode
        lkl_name = self.__class__.__name__.lower()
        self.use_cl = [lkl_name[i : i + 2] for i in range(0, len(lkl_name), 2)]

        self.nbins = self.bin_max - self.bin_min + 1
        if self.nbins < 1:
            raise LoggedError(self.log, f"Selected an invalid number of bandpowers ({self.nbins})")
        # if self.nfreq != 1:
        #     raise LoggedError(self.log, "Sorry, current code wont work for multiple freqs")
        if self.windows_lmin < 1 or self.windows_lmin >= self.windows_lmax:
            raise LoggedError(self.log, "Invalid ell ranges for SPTPol")

        # Read in bandpowers (remove index column)
        self.bandpowers = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)[1:]

        # Read in covariance
        self.cov = np.loadtxt(os.path.join(self.data_folder, self.cov_file))

        # Read in beam covariance
        self.beam_cov = np.loadtxt(os.path.join(self.data_folder, self.beam_cov_file))

        # Read in windows
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_dir, f"window_{i}.txt"), unpack=True
                )[1:]
                for i in range(self.bin_min, self.bin_max + 1)
            ]
        )

        # Compute spectra/cov indices given spectra to fit
        vec_indices = np.array([default_spectra_list.index(spec) for spec in self.spectra_to_fit])
        self.bandpowers = self.bandpowers[vec_indices].flatten()
        self.windows = self.windows[:, vec_indices, :]
        cov_indices = np.array(
            [np.arange(i * self.nbins, (i + 1) * self.nbins) for i in vec_indices]
        )
        cov_indices = cov_indices.flatten()
        # Select spectra/cov elements given indices
        self.cov = self.cov[np.ix_(cov_indices, cov_indices)]
        self.beam_cov = self.beam_cov[np.ix_(cov_indices, cov_indices)] * self.beam_cov_scaling
        self.log.debug(f"Selected bp indices: {vec_indices}")
        self.log.debug(f"Selected cov indices: {cov_indices}")

        # Compute cross-spectra frequencies and mode given the spectra name to fit
        r = re.compile("(.+?)_(.)x(.+?)_(.)")
        self.cross_frequencies = [r.search(spec).group(1, 3) for spec in self.spectra_to_fit]
        self.cross_spectra = ["".join(r.search(spec).group(2, 4)) for spec in self.spectra_to_fit]
        self.frequencies = sorted(
            {float(freq) for freqs in self.cross_frequencies for freq in freqs}
        )
        self.log.debug(f"Using {self.cross_frequencies} cross-frequencies")
        self.log.debug(f"Using {self.cross_spectra} cross-spectra")
        self.log.debug(f"Using {self.frequencies} GHz frequency bands")

        # Read in calibration covariance and select mode/frequencies
        # The order of the cal covariance is T90, T150, T220, E90, E150, E220
        calib_cov = np.loadtxt(os.path.join(self.data_folder, self.calib_cov_file))
        cal_indices = np.array([[90.0, 150.0, 220.0].index(freq) for freq in self.frequencies])
        if "TE" not in self.cross_spectra:
            # Only polar calibrations shift by 3
            cal_indices += 3
        else:
            cal_indices = np.concatenate([cal_indices, cal_indices + 3])
        calib_cov = calib_cov[np.ix_(cal_indices, cal_indices)]
        self.inv_calib_cov = np.linalg.inv(calib_cov)
        self.calib_params = np.array(
            ["map{}cal{}".format(*p) for p in itertools.product(["T", "P"], [90, 150, 220])]
        )[cal_indices]
        self.log.debug(f"Calibration parameters: {self.calib_params}")

        self.lmin = self.windows_lmin
        self.lmax = self.windows_lmax + 1  # to match fortran convention
        self.ells = np.arange(self.lmin, self.lmax)

    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: self.lmax for cl in self.use_cl}}

    def loglike(self, dlte, dlee, **params_values):

        T_CMB = 2.72548  # CMB temperature
        h = 6.62606957e-34  # Planck's constant
        kB = 1.3806488e-23  # Boltzmann constant
        Ghz_Kelvin = h / kB * 1e9

        # Planck function normalised to 1 at nu0
        def Bnu(nu, nu0, T):
            return (
                (nu / nu0) ** 3
                * (np.exp(Ghz_Kelvin * nu0 / T) - 1)
                / (np.exp(Ghz_Kelvin * nu / T) - 1)
            )

        # Derivative of Planck function normalised to 1 at nu0
        def dBdT(nu, nu0, T):
            x0 = Ghz_Kelvin * nu0 / T
            x = Ghz_Kelvin * nu / T

            dBdT0 = x0**4 * np.exp(x0) / (np.exp(x0) - 1) ** 2
            dBdT = x**4 * np.exp(x) / (np.exp(x) - 1) ** 2

            return dBdT / dBdT0

        lmin, lmax = self.lmin, self.lmax
        ells = np.arange(lmin, lmax + 2)

        dbs = np.empty_like(self.bandpowers)
        for i, (cross_spectrum, cross_frequency) in enumerate(
            zip(self.cross_spectra, self.cross_frequencies)
        ):
            dl_cmb = dlee if cross_spectrum == "EE" else dlte

            # Calculate derivatives for this position in parameter space.
            cl_derivative = dl_cmb[ells] * 2 * np.pi / (ells * (ells + 1))
            cl_derivative = 0.5 * (cl_derivative[2:] - cl_derivative[:-2])

            # Add CMB
            dls = dl_cmb[self.ells]

            # Add super sample lensing
            # (In Cl space) SSL = -k/l^2 d/dln(l) (l^2Cl) = -k(l*dCl/dl + 2Cl)
            if self.super_sample_lensing:
                kappa = params_values.get("kappa")
                dls += -kappa * (
                    self.ells**2 * (self.ells + 1) / (2 * np.pi) * cl_derivative
                    + 2 * dl_cmb[self.ells]
                )

            # Aberration correction
            # AC = beta*l(l+1)dCl/dln(l)/(2pi)
            # Note that the CosmoMC internal aberration correction and the SPTpol Henning likelihood differ
            # CosmoMC uses dCl/dl, Henning et al dDl/dl
            # In fact, CosmoMC is correct:
            # https://journals-aps-org.eu1.proxy.openathens.net/prd/pdf/10.1103/PhysRevD.89.023003
            dls += (
                -self.aberration_coefficient
                * cl_derivative
                * self.ells**2
                * (self.ells + 1)
                / (2 * np.pi)
            )

            # Simple poisson foregrounds
            # This is any poisson power. Meant to describe both radio galaxies and DSFG. By giving each frequency combination an amplitude
            # to play with this gives complete freedom to the data
            if self.poisson_switch and cross_spectrum == "EE":
                Dl_poisson = params_values.get("Dl_Poisson_{}x{}".format(*cross_frequency))
                dls += self.ells * (self.ells + 1) * Dl_poisson / (3000 * 3001)

            # Polarised galactic dust
            if self.dust_switch:
                TDust = params_values.get("TDust")
                ADust = params_values.get(f"ADust_{cross_spectrum}_150")
                AlphaDust = params_values.get(f"AlphaDust_{cross_spectrum}")
                BetaDust = params_values.get(f"BetaDust_{cross_spectrum}")
                dfs = (
                    lambda beta, temp, nu0, nu: (nu / nu0) ** beta
                    * Bnu(nu, nu0, temp)
                    / dBdT(nu, nu0, T_CMB)
                )
                dust = ADust * (self.ells / 80) ** (AlphaDust + 2)
                for freq in cross_frequency:
                    dust *= dfs(BetaDust, TDust, 150, self.nu_eff_list.get(int(freq)))
                dls += dust

            # Scale by calibration
            if cross_spectrum == "EE":
                # Calibration for EE: 1/(Ecal_1*Ecal_2) since we matched the EE spectrum to Planck's
                calibration = 1.0
                for freq in cross_frequency:
                    calibration /= params_values.get(f"mapPcal{freq}")
            if cross_spectrum == "TE":
                # Calibration for TE: 0.5*(1/(Tcal_1*Ecal_2) + 1/(Tcal_2*Ecal_1))
                freq1, freq2 = cross_frequency
                calibration = 0.5 * (
                    1
                    / (params_values.get(f"mapTcal{freq1}") * params_values.get(f"mapPcal{freq2}"))
                    + 1
                    / (params_values.get(f"mapTcal{freq2}") * params_values.get(f"mapPcal{freq1}"))
                )
            dls *= calibration

            # Binning via window and concatenate
            dbs[i * self.nbins : (i + 1) * self.nbins] = self.windows[:, i, :] @ dls

        # Take the difference to the measured bandpower
        delta_cb = dbs - self.bandpowers

        # Construct the full covariance matrix
        cov_w_beam = self.cov + self.beam_cov * np.outer(dbs, dbs)

        chi2 = delta_cb @ np.linalg.inv(cov_w_beam) @ delta_cb
        sign, slogdet = np.linalg.slogdet(cov_w_beam)

        # Add calibration prior
        delta_cal = np.log(np.array([params_values.get(p) for p in self.calib_params]))
        cal_prior = delta_cal @ self.inv_calib_cov @ delta_cal

        self.log.debug(f"SPT3G X²/ndof = {chi2:.2f}/{len(delta_cb)}")
        self.log.debug(f"SPT3G detcov = {slogdet:.2f}")
        self.log.debug(f"SPT3G cal. prior = {cal_prior:.2f}")
        return -0.5 * (chi2 + slogdet + cal_prior)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("te"), Cls.get("ee"), **data_params)


class TEEE(SPT3GPrototype):
    r"""
    Likelihood for Dutcher et al. 2020
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, EE/TE
    Written by Lennart Balkenhol
    """
