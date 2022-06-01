""".. module:: SPTPol

:Synopsis: Definition of python-native CMB likelihood for SPTPol TE+EE. Adapted from Fortran
likelihood code https://lambda.gsfc.nasa.gov/product/spt/sptpol_lh_2017_get.cfm

:Author: Xavier Garrido

"""

# Global
import os
from typing import Optional

import numpy as np
from cobaya.conventions import packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError


class SPTPolPrototype(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/SPT/sptpol_2017/sptpol_cosmomc_nov16_v1p3.tar.gz",
        "data_path": "sptpol_2017",
    }

    bibtex_file = "sptpol.bibtex"

    nbin: Optional[int] = 56
    nfreq: Optional[int] = 1
    windows_lmin: Optional[int] = 3
    windows_lmax: Optional[int] = 10600
    correct_aberration: Optional[bool] = True

    data_folder: Optional[str] = "sptpol_2017/SPTpol_Likelihood_1p3/data/sptpol_500d_TEEE"
    bp_file: Optional[str]
    cov_file: Optional[str]
    window_dir: Optional[str]
    beam_file: Optional[str]

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

        if self.nfreq != 1:
            raise LoggedError(self.log, "Sorry, current code wont work for multiple freqs")
        if self.windows_lmin < 2 or self.windows_lmin >= self.windows_lmax:
            raise LoggedError(self.log, "Invalid ell ranges for SPTPol")

        bands_per_freq = 3  # Should be three for SPTpol (TT,TE,EE, although mostly ignore TT).
        self.nband = bands_per_freq * self.nfreq
        self.nall = self.nbin * self.nfreq * (bands_per_freq - 1)  # Cov doesn't know about TT.

        # Read in bandpowers
        # Should be TE, EE, TT, in that order from SPTpol analysis.
        dummy, self.spec = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)
        self.spec = self.spec[: self.nall]  # Only keep TE, EE

        # Read in covariance
        # Should be TE, EE
        cov = np.fromfile(os.path.join(self.data_folder, self.cov_file))
        cov = cov.reshape((self.nall, self.nall))
        if self.use_cl == ["te"] or self.use_cl == ["ee"]:
            self.log.debug("Zero off-diagonal cov blocks...")
            cov[: self.nbin, self.nbin :] = 0.0
            cov[self.nbin :, : self.nbin] = 0.0

            # Explode TE auto-block if we only want EE.
            if self.use_cl == ["ee"]:
                self.log.debug("Exploding TE auto-cov block...")
                for i in range(self.nbin):
                    tmp = cov[i, i] * 10**24
                    cov[i, :] = 0.0
                    cov[:, i] = 0.0
                    cov[i, i] = tmp
            # Explode EE auto-block if we only want TE.
            if self.use_cl == ["te"]:
                self.log.debug("Exploding EE auto-cov block...")
                for i in range(self.nbin, self.nall):
                    tmp = cov[i, i] * 10**24
                    cov[i, :] = 0.0
                    cov[:, i] = 0.0
                    cov[i, i] = tmp

        self.log.debug(f"First entry of covariance matrix: {cov[0, 0]}")
        self.invcov = np.linalg.inv(cov)
        self.logp_const = np.log(2 * np.pi) * (-len(self.spec) / 2)
        self.logp_const -= 0.5 * np.linalg.slogdet(cov)[1]

        # Read in windows
        # Should be TE, EE
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_dir, f"window_{i}"), unpack=True
                )[1]
                for i in range(1, self.nall + 1)
            ]
        )

        # Get beam error term
        n_beam_terms = 2
        dummy, beam_err = np.loadtxt(os.path.join(self.data_folder, self.beam_file), unpack=True)
        self.beam_err = beam_err.reshape((n_beam_terms, self.nall))

        self.lmin = self.windows_lmin
        self.lmax = self.windows_lmax + 1  # to match fortran convention
        self.ells = np.arange(self.lmin, self.lmax)
        self.cl_to_dl_conversion = (self.ells * (self.ells + 1)) / (2 * np.pi)
        ells = np.arange(self.lmin - 1, self.lmax + 1)
        self.rawspec_factor = ells**2 / (ells * (ells + 1)) * 2 * np.pi

        for var in ["nbin", "nfreq", "nall", "windows_lmin", "windows_lmax", "data_folder", "lmax"]:
            self.log.debug(f"{var} = {getattr(self, var)}")

    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: self.lmax for cl in self.use_cl}}

    def get_foregrounds(self, **params_values):
        # First get model foreground spectrum (in Cl).
        # Note all the foregrounds are recorded in Dl at l=3000, so we
        # divide by d3000 to get to a normalized Cl spectrum.
        #
        # Start with Poisson power
        d3000 = 3000 * 3001 / (2 * np.pi)
        poisson_level_TE = params_values.get("czero_psTE_150") / d3000
        poisson_level_EE = params_values.get("czero_psEE_150") / d3000
        dlte_fg = poisson_level_TE * self.cl_to_dl_conversion
        dlee_fg = poisson_level_EE * self.cl_to_dl_conversion

        # Add dust foreground model (defined in Dl)
        ADust_TE = params_values.get("ADust_TE")
        ADust_EE = params_values.get("ADust_EE")
        alphaDust_TE = params_values.get("alphaDust_TE")
        alphaDust_EE = params_values.get("alphaDust_EE")
        dlte_fg += ADust_TE * (self.ells / 80) ** (alphaDust_TE + 2)
        dlee_fg += ADust_EE * (self.ells / 80) ** (alphaDust_EE + 2)

        return dlte_fg, dlee_fg

    def loglike(self, dlte, dlee, **params_values):
        # Getting foregrounds
        dlte_fg, dlee_fg = self.get_foregrounds(**params_values)

        # CMB from theory
        lmin, lmax = self.lmin, self.lmax
        dlte_cmb = dlte[lmin:lmax]
        dlee_cmb = dlee[lmin:lmax]

        # Calculate derivatives for this position in parameter space.
        dlte = dlte[lmin - 1 : lmax + 1]
        dlee = dlee[lmin - 1 : lmax + 1]
        rs_te = self.rawspec_factor * dlte
        rs_ee = self.rawspec_factor * dlee
        deriv_te = 0.5 * (rs_te[2:] - rs_te[:-2])
        deriv_ee = 0.5 * (rs_ee[2:] - rs_ee[:-2])

        # Subtract the kappa parameter for super sample lensing.
        # kappa parameter as described in Manzotti, et al. 2014, equation (32).
        kappa = params_values.get("kappa")
        dlte_fg -= kappa * deriv_te * self.cl_to_dl_conversion / self.ells
        dlee_fg -= kappa * deriv_ee * self.cl_to_dl_conversion / self.ells

        # Correct for aberration
        if self.correct_aberration:
            beta = 0.0012309
            dipole_cosine = -0.4033
            dlte_fg -= beta * dipole_cosine * self.ells * 0.5 * (dlte[2:] - dlte[:-2])
            dlee_fg -= beta * dipole_cosine * self.ells * 0.5 * (dlee[2:] - dlee[:-2])

        # Now bin into bandpowers with the window functions.
        win_te, win_ee = self.windows[: self.nbin], self.windows[self.nbin :]
        dbte = win_te @ (dlte_cmb + dlte_fg)
        dbee = win_ee @ (dlee_cmb + dlee_fg)

        # Scale theory spectrum by calibration:
        mapTcal = params_values.get("mapTcal")
        mapPcal = params_values.get("mapPcal")
        cal_TE = mapTcal**2 * mapPcal
        cal_EE = mapTcal**2 * mapPcal**2
        dbte /= cal_TE
        dbee /= cal_EE

        # Beam errors
        beam1 = params_values.get("beam1")
        beam2 = params_values.get("beam2")
        beam_factor = (1 + self.beam_err[0] * beam1) * (1 + self.beam_err[1] * beam2)
        delta_cb = np.concatenate([dbte, dbee]) * beam_factor - self.spec

        chi2 = delta_cb @ self.invcov @ delta_cb

        self.log.debug(f"SPTPol X²/ndof = {chi2:.2f}/{self.nall}")
        return -0.5 * chi2  # + self.logp_const

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("te"), Cls.get("ee"), **data_params)


class TEEE(SPTPolPrototype):
    r"""
    SPTpol 2017 500d EETE power spectrum 50 < ell < 8000 (Henning et al 2017)
    """


class TE(SPTPolPrototype):
    r"""
    SPTpol 2017 500d TE power spectrum 50 < ell < 8000 (Henning et al 2017)
    """


class EE(SPTPolPrototype):
    r"""
    SPTpol 2017 500d EE power spectrum 50 < ell < 8000 (Henning et al 2017)
    """
