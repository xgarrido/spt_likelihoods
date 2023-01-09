""".. module:: SPT-3G

:Synopsis: Definition of python-native CMB likelihood for SPT-3G TT+TE+EE. Adapted from Fortran
likelihood code https://pole.uchicago.edu/public/data/balkenhol22/SPT3G_2018_TTTEEE_public_likelihood.zip
[Balkenhold et al. 2022]

:Author: Matthieu Tristram

"""

import itertools
import os
import re
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

from . import spt3g_TTTEEE_foregrounds as spt3g_fg

default_spectra_list = [
    "90_Tx90_T",
    "90_Tx90_E",
    "90_Ex90_E",
    "90_Tx150_T",
    "90_Tx150_E",
    "90_Ex150_E",
    "90_Tx220_T",
    "90_Tx220_E",
    "90_Ex220_E",
    "150_Tx150_T",
    "150_Tx150_E",
    "150_Ex150_E",
    "150_Tx220_T",
    "150_Tx220_E",
    "150_Ex220_E",
    "220_Tx220_T",
    "220_Tx220_E",
    "220_Ex220_E"
]


class SPT3GPrototype(InstallableLikelihood):
    install_options = {
        "download_url": "https://pole.uchicago.edu/public/data/balkenhol22/SPT3G_2018_TTTEEE_public_likelihood.zip",
        "data_path": "spt3g_2018",
    }

    bibtex_file = "spt3g_TTTEEE.bibtex"

    bin_min: Optional[int] = 1
    bin_max: Optional[int] = 44
    windows_lmin: Optional[int] = 1
    windows_lmax: Optional[int] = 3200

    spectra_to_fit: Optional[Sequence[str]] = default_spectra_list

    spec_bin_min: Optional[Sequence[int]] = [10,  1,  1, 10,  1,  1, 10,  1,  1, 10,  1,  1, 15,  1,  1, 15,  1,  1]
    spec_bin_max: Optional[Sequence[int]] = [44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44]

    data_folder: Optional[str] = "spt3g_2018/SPT3G_2018_TTTEEE_public_likelihood/data/SPT3G_2018_TTTEEE"
    bandpower_filename: Optional[str]
    covariance_filename: Optional[str]
    fiducial_covariance_filename: Optional[str]
    beam_covariance_filename: Optional[str]
    cal_covariance_filename: Optional[str]
    window_folder: Optional[str]
    nu_eff_filename: Optional[str]

    cov_eval_cut_threshold: Optional[float] = 0.2
    cov_eval_large_number_replacement: Optional[float] = 1e3
    beam_cov_scale: Optional[float] = 1.0
    aberration_coefficient: Optional[float] = 0.0

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

        self.N_s = len(self.spectra_to_fit)
        for b in range(self.N_s):
            if self.spec_bin_min[b] <= 0 or self.spec_bin_max[b] >= 45:
                raise LoggedError(self.log, f"SPT-3G 2018 TTTEEE: bad ell range selection for spectrum: {self.spectra_to_fit[b]}")
        self.N_b = np.array(self.spec_bin_max) - np.array(self.spec_bin_min) + 1

        # Check if a late crop is requested and read in the mask if necessary
        # MT: Not Implemented 

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

        # Determine how many spectra are TT vs TE vs EE and the total number of bins we are fitting
        self.N_s_TT = sum([c == 'TT' for c in self.cross_spectra])
        self.N_s_TE = sum([c == 'TE' for c in self.cross_spectra])
        self.N_s_EE = sum([c == 'EE' for c in self.cross_spectra])

        # Determine how many different frequencies get used
        self.N_freq = len( self.frequencies)

        # Band Powers
        self.bandpowers = np.loadtxt(os.path.join(self.data_folder, self.bandpower_filename), unpack=True)
        self.bandpowers = self.bandpowers.reshape( -1, self.bin_max)

        # Covariance Matrix
        bp_cov  = np.loadtxt(os.path.join(self.data_folder, self.covariance_filename))
        fid_cov = np.loadtxt(os.path.join(self.data_folder, self.fiducial_covariance_filename))

        # Beam Covariance Matrix
        self.beam_cov = np.loadtxt(os.path.join(self.data_folder, self.beam_covariance_filename))
        self.beam_cov = self.beam_cov * self.beam_cov_scale

        # Windows Functions
        # These are a bit trickier to handle due to the independent cuts possible for TT/TE/EE
        # The windows for low ell TT spectra exist in the files so that we can read these in in a nice array
        # Re-order/crop later when the binning is performed
        #MT: WARNING starts at l=1
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_folder, f"window_{i}.txt"), unpack=True
                )[1:]
                for i in range(self.bin_min, self.bin_max + 1)
            ]
        )

        # Compute spectra/cov indices given spectra to fit
        vec_indices = np.array([default_spectra_list.index(spec) for spec in self.spectra_to_fit])
        self.bandpowers = self.bandpowers[vec_indices].flatten()
        self.windows = self.windows[:, vec_indices, :]
        cov_indices = np.concatenate(
            [np.arange(sum(self.N_b[:i]), sum(self.N_b[:i+1]), dtype=int) for i in vec_indices]
        )
        # Select spectra/cov elements given indices
        bp_cov = bp_cov[np.ix_(cov_indices, cov_indices)]
        fid_cov = fid_cov[np.ix_(cov_indices, cov_indices)]
        self.beam_cov = self.beam_cov[np.ix_(cov_indices, cov_indices)]
        self.log.debug(f"Selected bp indices: {vec_indices}")
        self.log.debug(f"Selected cov indices: {cov_indices}")

        # Ensure covariance is positive definite
        self.bp_cov_posdef = self._MakeCovariancePositiveDefinite(bp_cov, fid_cov)

        # Calibration Covariance
        # The order of the cal covariance is T90, T150, T220, E90, E150, E220
        calib_cov = np.loadtxt(os.path.join(self.data_folder, self.cal_covariance_filename))
        cal_indices = np.array([[90.0, 150.0, 220.0].index(freq) for freq in self.frequencies])
        if "TE" not in self.cross_spectra:
            # Only polar calibrations shift by 3
            cal_indices += 3
        else:
            cal_indices = np.concatenate([cal_indices, cal_indices + 3])
        calib_cov = calib_cov[np.ix_(cal_indices, cal_indices)]
        self.inv_calib_cov = np.linalg.inv(calib_cov)
        self.calib_params = np.array(
            ["{}cal{}".format(*p) for p in itertools.product(["T", "E"], [90, 150, 220])]
        )[cal_indices]
        self.log.debug(f"Calibration parameters: {self.calib_params}")

        # Effective band centres
        nu_eff = np.loadtxt(os.path.join(self.data_folder, self.nu_eff_filename))
        nu_eff_gal_cirrus, nu_eff_pol_gal_dust, nu_eff_DSFG, bu_eff_radio, nu_eff_tSZ = nu_eff

        self.lmin = self.windows_lmin
        self.lmax = self.windows_lmax + 1  # to match fortran convention
        self.ells = np.arange(self.lmin, self.lmax)

        # Initialise foreground model
        self.fg = spt3g_fg.SPT3G_2018_TTTEEE_Ini_Foregrounds(data_folder=self.data_folder, **self.foregrounds)

        self.log.debug(f"SPT-3G 2018 TTTEEE: Likelihood successfully initialised!")


    def _MakeCovariancePositiveDefinite( self, input_cov, fiducial_cov):
        # Checks if a matrix is positive definite
        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)

        # Changes the basis of the matrix using the eigenvectors of another
        # Goes either fowards or backwards
        def ChangeBasisToFiducial( cov_input, cov_fiducial, forward):
            # Calculate the eigenvectors
            cov_eigenvalues,cov_eigenvectors = np.linalg.eig( cov_input)
            
            # Get the inverse of the eigenvectors matrix
            cov_eigenvectors_inv = np.linalg.inv( cov_eigenvectors)

            # Change basis (either to fiducial or back)
            if forward:
                cov_output = cov_eigenvectors_inv @ cov_input @ cov_eigenvectors
            else:
                cov_output = cov_eigenvectors @ cov_input @ cov_eigenvectors_inv
            
            return cov_output

        # Fix negative eigenvalues of a matrix by setting anything below a threshold to some fixed value
        def FixEvalsOfMatrix( cov_input):
            # Calculate the eigenvectors
            cov_eigenvalues,cov_eigenvectors = np.linalg.eig( cov_input)

            # Set negative eigenvalues to a large positive number
            cov_eigenvalues[cov_eigenvalues < self.cov_eval_cut_threshold] = cov_eval_large_number_replacement

            # Get the inverse of the eigenvectors matrix
            cov_eigenvectors_inv = np.linalg.inv( cov_eigenvectors)

            # Cast back into a covariance matrix
            cov_output = cov_eigenvectors @ np.diag(cov_eigenvalues) @ cov_eigenvectors_inv
            
            return cov_output

        if is_pos_def( input_cov):
            output_cov = input_cov
        else:
            cov_in_new_basis = ChangeBasisToFiducial(input_cov, fiducial_cov, True)

            # Rescale
            v = np.sqrt(np.diag(cov_in_new_basis))
            cov_in_new_basis_scaled = cov_in_new_basis / np.outer(v, v)

            # Fix the negative eigenvalues
            cov_in_new_basis_scaled_fixed = FixEvalsOfMatrix( cov_in_new_basis_scaled)

            # Undo rescaling
            cov_in_new_basis_scaled_fixed *= np.outer(v, v)

            # Undo change of basis
            output_cov = ChangeBasisToFiducial(cov_in_new_basis_scaled_fixed, fiducial_cov, False)

        return output_cov


    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: self.lmax for cl in self.use_cl}}

    def loglike(self, dl_cmb, **params):

        lmin, lmax = self.lmin, self.lmax
        ells = np.arange(lmin, lmax + 2)
        fg = self.fg

        dbs = np.empty_like(self.bandpowers)
        dlfg = np.zeros( fg.N_fg_max, SPT3G_windows_lmax - SPT3G_windows_lmin + 1)
        for i, (cross_spectrum, cross_frequency) in enumerate(
            zip(self.cross_spectra, self.cross_frequencies)
        ):

            # Add CMB
            dls = dl_cmb[cross_spectrum][self.ells]

            fg.ApplySuperSampleLensing( params.get("kappa"),
                                        dls, dlfg)

            fg.ApplyAberrationCorrection( params.get("aberration_coefficient"),
                                          dls, dlfg)

            if cross_spectrum == "TT":
                fg.AddPoissonPower( params.get(f"{cross_spectrum}_Poisson_{cross_frequency[0]}x{cross_frequency[1]}"),
                                    dls, dlfg)

                fg.AddGalacticDust( params.get("TT_GalCirrus_Amp"),
                                    params.get("TT_GalCirrus_Alpha"),
                                    params.get("TT_GalCirrus_Beta"),
                                    nu_eff_gal_cirrus[cross_frequency[0]], nu_eff_gal_cirrus[cross_frequency[1]],
                                    dls, dlfg)
                
                fg.AddCIBClustering( params.get("TT_CIBClustering_Amp"),
                                     params.get("TT_CIBClustering_Alpha"),
                                     params.get("TT_CIBClustering_Beta"),
                                     nu_eff_DSFG[cross_spectrum[0]], nu_eff_DSFG[cross_spectrum[1]],
                                     params.get(f"TT_CIBClustering_decorr_{cross_spectrum[0]}"),
                                     params.get(f"TT_CIBClustering_decorr_{cross_spectrum[1]}"),
                                     dls, dlfg)

                fg.AddtSZ( params.get( "TT_tSZ_Amp"),
                           nu_eff_tSZ[cross_spectrum[0]], nu_eff_tSZ[cross_spectrum[1]],
                           _, _, _,
                           dls, dlfg)

                fg.AddtSZCIBCorrelation( params.get("TT_tSZ_CIB_corr"),
                                         params.get("TT_tSZ_Amp"),
                                         params.get("TT_CIB_Clustering_Amp"),
                                         params.get("TT_CIB_Clustering_Alpha"),
                                         params.get("TT_CIB_Clustering_Beta"),
                                         params.get(f"TT_CIBClustering_decorr_{cross_spectrum[0]}"),
                                         params.get(f"TT_CIBClustering_decorr_{cross_spectrum[1]}"),
                                         nu_eff_DSFG[cross_spectrum[0]], nu_eff_DSFG[cross_spectrum[1]],
                                         nu_eff_tSZ[cross_spectrum[0]], nu_eff_tSZ[cross_spectrum[1]],
                                         _, _, _,
                                         dls, dlfg)
                
                fg.addkSZ( params.get("TT_kSZ_Amp"),
                           _, _, _, _, _, _,
                           dls, dlfg)

            elif cross_spectrum == "TE":
                fg.AddGalacticDust( params.get("TE_PolGalDust_Amp"),
                                    params.get("TE_PolGalDust_Alpha"),
                                    params.get("TE_PolGalDust_Beta"),
                                    nu_eff_gal_cirrus[cross_frequency[0]], nu_eff_gal_cirrus[cross_frequency[1]],
                                    dls, dlfg)
                
            elif cross_spectrum == "EE":
                fg.AddPoissonPower( params.get(f"{cross_spectrum}_Poisson_{cross_frequency[0]}x{cross_frequency[1]}"),
                                    dls, dlfg)
                
                fg.AddGalacticDust( params.get("EE_PolGalDust_Amp"),
                                    params.get("EE_PolGalDust_Alpha"),
                                    params.get("EE_PolGalDust_Beta"),
                                    nu_eff_gal_cirrus[cross_frequency[0]], nu_eff_gal_cirrus[cross_frequency[1]],
                                    dls, dlfg)

            fg.ApplyCalibration( params.get(f"{cross_spectrum[0]}cal{cross_frequency[0]}"),
                                 params.get(f"{cross_spectrum[1]}cal{cross_frequency[1]}"),
                                 params.get(f"{cross_spectrum[0]}cal{cross_frequency[1]}"),
                                 params.get(f"{cross_spectrum[1]}cal{cross_frequency[0]}"),
                                 dls, dlfg)

            # Binning via window and concatenate
            dbs[sum(self.N_b[:i]):sum(self.N_b[:i+1])] = self.windows[self.spec_bin_min[i]:self.spec_bin_max[i], i, :] @ dls

        # Calculate difference of theory and data
        delta_data_model = self.bandpowers - dbs
            
        # Add the beam coariance to the band power covariance
        cov_for_logl = self.bp_cov_posdef + self.beam_cov * np.outer(dbs, dbs)

        # Final crop to ignore select band powers
        # MT: not implemented

        # Compute chisq
        chi2, slogdet = self._gaussian_loglike(cov_for_logl, delta_data_model, cholesky=True)

        # Apply calibration prior
        delta_cal = np.array([params_values.get(p)-1. for p in self.calib_params])
        cal_prior = delta_cal @ self.inv_calib_cov @ delta_cal

        self.log.debug(f"SPT3G X²/ndof = {chi2:.2f}/{len(delta_data_model)}")
        self.log.debug(f"SPT3G detcov = {slogdet:.2f}")
        self.log.debug(f"SPT3G cal. prior = {cal_prior:.2f}")
        return -0.5 * (chi2 + slogdet + cal_prior)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike( {'TT':Cls.get("tt"),'TE':Cls.get("te"),'EE':Cls.get("ee")}, **data_params)


    def _gaussian_loglike(self, dlcov, res, cholesky=True):
        """
        Returns -Log Likelihood for Gaussian: (d^T Cov^{-1} d + log|Cov|)/2
        """

        if cholesky:
            from scipy.linalg import cho_factor, cho_solve

            L, low = cho_factor(dlcov)

            # compute ln det
            slogdet = 2.0 * np.sum(np.log(np.diag(L)))

            # Compute C-1.d
            invCd = cho_solve((L, low), res)

            # Compute chi2
            chi2 = res @ invCd

        else:
            chi2 = res @ np.linalg.inv(dlcov) @ res
            sign, slogdet = np.linalg.slogdet(dlcov)

        return chi2 / 2.0, slogdet / 2.0

class TTTEEE(SPT3GPrototype):
    r"""
    Likelihood for Balkenhold et al. 2022
    SPT-3G Y1 95, 150, 220GHz bandpowers, l=300-3000, TT/TE/EE
    Written by Lennart Balkenhol
    """
