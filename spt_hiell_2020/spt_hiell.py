""".. module:: SPTPol

:Synopsis: Definition of python-native CMB likelihood for SPT likelihood (SPTpol+SZ l=2000-11000
power spectrum). Adapted from Fortran likelihood code
https://lambda.gsfc.nasa.gov/product/spt/spt_ps_2020_get.cfm

:Author: Mathieu Tristram

"""
import os
from typing import Optional, Sequence

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import _packages_path
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

from . import spt_foregrounds as sptfg


class SPTHiellLikelihood(InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/SPT/reichardt_2020/likelihood.tar.gz",
        "data_path": "spt_hiell_2020",
    }

    cal_cov = [
        [1.1105131e-05, 3.5551351e-06, 1.1602891e-06],
        [3.5551351e-06, 3.4153547e-06, 2.1348018e-06],
        [1.1602891e-06, 2.1348018e-06, 1.7536000e-05],
    ]

    freq: Sequence[int] = [95, 150, 220]

    data_folder: Optional[str] = "spt_hiell_2020/likelihood"
    desc_file: Optional[str]
    bp_file: Optional[str]
    cov_file: Optional[str]
    beamerr_file: Optional[str]
    window_file: Optional[str]

    normalizeSZ_143GHz = True
    callFGprior = True
    applyFTSprior = True
    use_dZ = False

    foregrounds = dict(
        spt_dataset_tSZ="ptsrc/dl_shaw_tsz_s10_153ghz_norm1_fake25000.txt",
        spt_dataset_kSZ="ptsrc/dl_ksz_CSFplusPATCHY_13sep2011_norm1_fake25000.txt",
        spt_dataset_kSZ2="ptsrc/dl_ksz_oz_patchy_nolowell_20110708_norm1_fake25000.txt",
        spt_dataset_clustered="ptsrc/dl_cib_1halo_norm1_25000.txt",
        spt_dataset_clustered2="ptsrc/dl_cib_2halo_norm1_25000.txt",
    )

    def initialize(self, verbose=True):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to CIB_Likelihood data. Set the likelihood property 'path' or the common property '%s'.",
                _packages_path,
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        # check data_folder
        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

        # Init foreground model
        self.fg = sptfg.SPTforegounds(data_folder=self.data_folder, **self.foregrounds)

        # Update data_folder location
        self.data_folder = os.path.join(self.data_folder, "data/spt_hiell_2020")

        # get info from the desc_file
        self._update_with_desc_file()

        if verbose:
            print("nall: {}".format(self.nall))
            print("nfreq: {}".format(self.nfreq))
            print("spt_windows_lmin: {}".format(self.spt_windows_lmin))
            print("spt_windows_lmax: {}".format(self.spt_windows_lmax))
        self.lmax = self.spt_windows_lmax

        if self.spt_windows_lmax > self.fg.ReportFGLmax():
            raise ValueError("Hard-wired lmax in foregrounds is too low for SPT_hiell")

        if self.spt_windows_lmin < 2 or self.spt_windows_lmin >= self.spt_windows_lmax:
            raise ValueError("Invalid lranges for sptpol")

        # ells vector is 2 ell longer in order to do cl derivatives.
        self.ells = np.arange(self.spt_windows_lmin, self.spt_windows_lmax)

        # Define an array with the l*(l+1)/2pi factor to convert to Dl from Cl.
        cl_to_dl_conversion = self.ells * (self.ells + 1) / 2 / np.pi

        # read bandpowers (90x90, 90x150, 90x220, 150x150, 150x220, 220x220)
        # check file before
        bla, self.spec = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)

        # read covariance
        # check file before
        self.cov = np.fromfile(os.path.join(self.data_folder, self.cov_file), np.float64).reshape(
            self.nall, self.nall
        )
        #        if verbose:
        #            print( self.cov)

        # read beam_err
        # check file before
        self.beam_err = np.fromfile(
            os.path.join(self.data_folder, self.beamerr_file), np.float64
        ).reshape(self.nall, self.nall)
        #        if verbose:
        #            print( self.beam_err)

        # Read windows
        # check file before
        self.windows = self._read_windows(os.path.join(self.data_folder, self.window_file))

        # define indicies
        i = 0
        self.indices = []
        for j in range(self.nfreq):
            for k in range(j, self.nfreq):
                self.indices.append((j, k))

        # define offsets for xfreq in Cl vector
        self.offsets = [0]
        for i in range(1, self.nband):
            self.offsets.append(self.offsets[i - 1] + self.nbins[i - 1])

        self.Successful_SPT_Init = True

        if verbose:
            print("Init SPTlik done")

    def _read_windows(self, filename):
        import struct

        with open(filename, "rb") as f:
            efflmin, efflmax = struct.unpack("@II", f.read(2 * np.dtype(np.int32).itemsize))

        if efflmax < self.spt_windows_lmin or efflmin > self.spt_windows_lmax:
            raise ValueError("unallowed l-ranges for binary window functions")

        #        j0 = self.spt_windows_lmin if self.spt_windows_lmin > efflmin else efflmin
        #        j1 = self.spt_windows_lmax if self.spt_windows_lmax < efflmax else efflmax
        #        if j1 < j0:
        #            raise ValueError( "unallowed l-ranges for binary window functions - no allowed ells")

        offset = 2 * (np.dtype(np.int32).itemsize)
        print("********* efflmin = ", efflmin)
        print("********* efflmax = ", efflmax)
        print("********* offset = ", offset)
        #        windows = np.zeros( (self.nall, self.spt_windows_lmax+1) )
        #        windows[:,j0:j1+1] = np.fromfile( filename, dtype=np.float64, offset=offset).reshape(self.nall,-1)
        # windows = np.zeros((self.nall, efflmax + 1))
        windows = np.fromfile(filename, dtype=np.float64, offset=offset).reshape(self.nall, -1)

        print(windows[:5, 2000:2005])
        print(windows.shape)
        return windows
        # windows[:, : self.spt_windows_lmin] = 0
        # return windows[:, : self.spt_windows_lmax + 1]

    def _update_with_desc_file(self):
        filename = os.path.join(self.data_folder, self.desc_file)
        with open(filename) as f:
            self.nall, self.nfreq = [int(float(x)) for x in next(f).split()]
            self.nband = int(self.nfreq * (self.nfreq + 1) / 2)

            self.nbins = [int(next(f)) for i in range(self.nband)]
            if self.nall != sum(self.nbins):
                raise ValueError("mismatched number of bandpowers")

            self.spt_norm_fr = [float(next(f)) for i in range(5)]

            if self.normalizeSZ_143GHz:
                self.spt_norm_fr[4] = 143.0
                print("Using 143 as tSZ center freq")

            self.spt_windows_lmin, self.spt_windows_lmax = [int(float(x)) for x in next(f).split()]

            eff_fr = []
            for j in range(self.nfreq):
                eff_fr.append([float(next(f)) for i in range(5)])
            self.spt_eff_fr = np.array(eff_fr)

            self.spt_prefactor = np.array([float(next(f)) for i in range(self.nfreq)])

    #        maxnbin = np.max( nbins)
    #        indices = np.array( (2,nband) )
    #        offsets = np.array( nband)

    def _GaussianLogLike(self, dlcov, res):
        """
        Returns -Log Likelihood for Gaussian: (d^T Cov^{-1} d + log|Cov|)/2
        """
        from scipy.linalg import cho_factor, cho_solve

        L, low = cho_factor(dlcov)

        # compute det
        detcov = 2.0 * np.sum(np.log(np.diag(L)))

        # Compute C-1.d
        invCd = cho_solve((L, low), res)

        # Compute chi2
        chi2 = res @ invCd

        return chi2 / 2.0, detcov / 2.0

    def loglike(self, dl_cmb, **params):
        """
        dl_cmb: Dl TT
        """
        CalFactors = [params["mapCal{}".format(nu)] for nu in self.freq]
        FTSfactor = params["FTS_calibration_error"]

        # scaling theory
        #        tszfac = sptfg.cosmo_scale_tsz(theory.H0,theory.sigma_8,theory.omb)
        #        params["czero_tsz"] = params["czero_tsz"] * tszfac
        #        kszfac = sptfg.cosmo_scale_ksz(theory.H0,theory.sigma_8,theory.omb,theory.omc+theory.omb+theory.omnu,theory.InitPower(ns_index),theory.tau)
        #        params["czero_ksz"] = params["czero_ksz"] * kszfac

        # loop on nband
        cbs = np.zeros(self.nall)
        for i in range(self.nband):
            j, k = self.indices[i]
            thisoffset = self.offsets[i]
            thisbin = self.nbins[i]

            # get theory spectra
            dl_fg = self.fg.dl_foregrounds(
                params,
                j,
                k,
                self.nfreq,
                self.spt_eff_fr + FTSfactor,
                self.spt_norm_fr,
                self.spt_windows_lmin,
                self.spt_windows_lmax,
            )
            dl_th = (
                dl_cmb[self.spt_windows_lmin : self.lmax + 1]
                + dl_fg[self.spt_windows_lmin : self.lmax + 1]
            )

            print("************* i", i)
            print("************* j", j)
            print("************* k", k)
            print("************* fgs", dl_fg[10:20])
            print("************* cmb", dl_cmb[10:20])
            print("************ offset", thisoffset)
            print("************ thisbin", thisbin)
            print("************ window", self.windows.shape)
            print("************ dl_th", dl_th[:10])

            # bin theory with window functions
            tmpcb = self.windows @ dl_th

            print("tmpcb", tmpcb)
            # apply prefactors
            tmpcb = (
                tmpcb
                * self.spt_prefactor[k]
                * self.spt_prefactor[j]
                * CalFactors[j]
                * CalFactors[k]
            )

            cbs[thisoffset : thisoffset + thisbin] = tmpcb[thisoffset : thisoffset + thisbin]

        print("*************** cbs", cbs)
        print("*************** spec", self.spec)
        # residual
        delta_cb = cbs - self.spec

        # Dl covariance (with beams)
        cov_w_beam = self.cov + (self.beam_err.T * cbs).T * cbs
        print("************** beam_err=", self.beam_err[:10, :10])
        print("************** cov_w_beam=", cov_w_beam[-11:, -11:])

        # compute LogLike
        LnL, detcov = self._GaussianLogLike(cov_w_beam, delta_cb)
        SPTHiEllLnLike = LnL + detcov
        NoCalLnLike = SPTHiEllLnLike
        print("################ NoCalLnLike", NoCalLnLike)
        self.log.debug(f"chi2/ndof = {LnL:.2f}/{self.nall}")

        # add FG priors
        if self.callFGprior:
            FGPriorLnLike = self.fg.getForegroundPriorLnL(params)
            SPTHiEllLnLike = SPTHiEllLnLike + FGPriorLnLike

        # add calib LogLike
        delta_calib = np.log(CalFactors)
        CalibLnLike, detcov2 = self._GaussianLogLike(self.cal_cov, delta_calib)
        SPTHiEllLnLike = SPTHiEllLnLike + CalibLnLike

        # add FTS prior
        # Prior is 0.3 GHz for 1 sigma around 0.
        if self.applyFTSprior:
            FTSLnLike = 0.5 * (FTSfactor / 0.3) ** 2
            SPTHiEllLnLike = SPTHiEllLnLike + FTSLnLike

        self.log.debug("SPTHiEllLnLike lnlike = {} (with priors)".format(SPTHiEllLnLike))
        self.log.debug("Calibration chisq = {}".format(2 * CalibLnLike))
        self.log.debug("lnLcov term = {}".format(detcov))
        self.log.debug("chisq for cov only: {}".format(2 * LnL))
        self.log.debug("chisq for FG prior: {}".format(2 * FGPriorLnLike))
        self.log.debug(
            "SPTHiEllLnLike chisq (after prior) = {}".format(2 * (SPTHiEllLnLike - detcov))
        )
        #        self.log.debug("LnL = {:.2f} (with priors)".format(SPTHiEllLnLike))

        print("******************* detcov2", detcov2)

        print("-------------", delta_cb @ np.linalg.inv(cov_w_beam) @ delta_cb)
        print("-------------", np.linalg.slogdet(cov_w_beam))
        # return lnL = -0.5*chi2
        return -SPTHiEllLnLike

    def get_requirements(self):
        requirements = dict(Cl={mode: self.lmax for mode in ["tt"]})
        return requirements

    def logp(self, **params_values):
        dl = self.theory.get_Cl(units="muK2", ell_factor=True)["tt"]
        return self.loglike(dl, **params_values)


class TT(SPTHiellLikelihood):
    """
    CMB likelihood with SPT-SZ and SPTpol surveys (Reichard et al. 2020)
    """
