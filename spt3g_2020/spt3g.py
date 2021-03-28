""".. module:: SPT-3G

:Synopsis: Definition of python-native CMB likelihood for SPT-3G TE+EE. Adapted from Fortran
likelihood code https://pole.uchicago.edu/public/data/dutcher21/SPT3G_2018_EETE_likelihood.tar.gz

:Author: Xavier Garrido

"""

# Global
import os
import re
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import _packages_path
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
        "download_url": "https://pole.uchicago.edu/public/data/dutcher21/SPT3G_2018_EETE_likelihood.tar.gz",
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
    radio_galaxies_nu0: Optional[float] = 150.0
    dsfg_nu0: Optional[float] = 150.0
    dust_nu0: Optional[float] = 150.0

    beam_cov_scaling: Optional[float] = 1.0

    spectra_to_fit: Optional[Sequence[str]] = default_spectra_list

    # SPT-3G Y1 EE/TE Effective band centres for polarised galactic dust.
    nu_eff_list: Optional[dict] = {90: 9.670270e01, 150: 1.499942e02, 220: 2.220433e02}

    data_folder: Optional[str] = "spt3g_2018/SPT3G_2018_EETE_likelihood/data/SPT3G_Y1_EETE"
    bp_file: Optional[str]
    cov_file: Optional[str]
    beam_cov_file: Optional[str]
    calib_cov_file: Optional[str]
    window_dir: Optional[str]

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                f"No path given to SPTPol data. Set the likelihood property 'path' or "
                "the common property '{_packages_path}'.",
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

        # bands_per_freq = 3  # Should be three for SPTpol (TT,TE,EE, although mostly ignore TT).
        # self.nband = bands_per_freq * self.nfreq
        # self.nall = self.nbin * self.nfreq * (bands_per_freq - 1)  # Cov doesn't know about TT.

        # Compute spectra/cov indices given spectra to fit
        self.indices = np.array([default_spectra_list.index(spec) for spec in self.spectra_to_fit])
        self.log.debug(f"Selected indices: {self.indices}")

        # Read in bandpowers (remove index column)
        self.bandpowers = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)[1:]

        # Read in covariance
        self.cov = np.loadtxt(os.path.join(self.data_folder, self.cov_file))

        # Read in beam covariance
        self.beam_cov = np.loadtxt(os.path.join(self.data_folder, self.beam_cov_file))

        # self.log.debug(f"First entry of covariance matrix: {cov[0, 0]}")
        # self.invcov = np.linalg.inv(cov)
        # self.logp_const = np.log(2 * np.pi) * (-len(self.spec) / 2)
        # self.logp_const -= 0.5 * np.linalg.slogdet(cov)[1]

        # Read in windows
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_dir, f"window_{i}.txt"), unpack=True
                )[1:]
                for i in range(self.bin_min, self.bin_max + 1)
            ]
        )

        # Select spectra/cov elements given indices
        self.bandpowers = self.bandpowers[self.indices]
        self.cov = self.cov[self.indices[:, None], self.indices]
        self.beam_cov = self.beam_cov[self.indices[:, None], self.indices] * self.beam_cov_scaling
        self.windows = self.windows[:, self.indices, :]

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

        # Read in calibration covariance and select frequencies
        calib_cov = np.loadtxt(os.path.join(self.data_folder, self.calib_cov_file))
        cal_indices = np.array([[90.0, 150.0, 220.0].index(freq) for freq in self.frequencies])
        cal_indices = np.concatenate([cal_indices, len(cal_indices) + cal_indices])
        calib_cov = calib_cov[cal_indices[:, None], cal_indices]
        self.inv_calib_cov = np.linalg.inv(calib_cov)

        self.lmin = self.windows_lmin
        self.lmax = self.windows_lmax + 1  # to match fortran convention
        self.ells = np.arange(self.lmin, self.lmax)
        # self.cl_to_dl_conversion = (self.ells * (self.ells + 1)) / (2 * np.pi)
        # ells = np.arange(self.lmin - 1, self.lmax + 1)
        # self.rawspec_factor = ells ** 2 / (ells * (ells + 1)) * 2 * np.pi

        # for var in ["nbin", "nfreq", "nall", "windows_lmin", "windows_lmax", "data_folder", "lmax"]:
        #     self.log.debug(f"{var} = {getattr(self, var)}")

    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: self.lmax for cl in self.use_cl}}

    # def get_foregrounds(self, dlte, dlee, **params_values):
    #     # First get model foreground spectrum (in Cl).
    #     # Note all the foregrounds are recorded in Dl at l=3000, so we
    #     # divide by d3000 to get to a normalized Cl spectrum.
    #     #
    #     # Start with Poisson power
    #     d3000 = 3000 * 3001 / (2 * np.pi)
    #     poisson_level_TE = params_values.get("czero_psTE_150") / d3000
    #     poisson_level_EE = params_values.get("czero_psEE_150") / d3000
    #     dlte_fg = poisson_level_TE * self.cl_to_dl_conversion
    #     dlee_fg = poisson_level_EE * self.cl_to_dl_conversion

    #     # Add dust foreground model (defined in Dl)
    #     ADust_TE = params_values.get("ADust_TE")
    #     ADust_EE = params_values.get("ADust_EE")
    #     alphaDust_TE = params_values.get("alphaDust_TE")
    #     alphaDust_EE = params_values.get("alphaDust_EE")
    #     dlte_fg += ADust_TE * (self.ells / 80) ** (alphaDust_TE + 2)
    #     dlee_fg += ADust_EE * (self.ells / 80) ** (alphaDust_EE + 2)

    #     return dlte_fg, dlee_fg

    def loglike(self, dlte, dlee, **params_values):

        for cross_spectrum, cross_frequency in zip(self.cross_spectra, self.cross_frequencies):
            freq1, freq2 = cross_frequency
            print(cross_spectrum, freq1, freq2)

        # chi2 = delta_cb @ self.invcov @ delta_cb
        chi2 = 0.0
        # self.log.debug(f"SPTPol X²/ndof = {chi2:.2f}/{self.nall}")
        return -0.5 * chi2  # + self.logp_const

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("te"), Cls.get("ee"), **data_params)


class TEEE(SPT3GPrototype):
    r"""
    Likelihood for Dutcher et al. 2020
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, EE/TE
    Written by Lennart Balkenhol
    """
    pass
