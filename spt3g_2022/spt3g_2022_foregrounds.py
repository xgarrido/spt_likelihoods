# pylint: disable=C0103
# Foregrounds for SPT3G 2018 TT/TE/EE likelihood
import os

import numpy as np
from cobaya.log import HasLogger

# Physical constants
T_CMB = 2.72548  # CMB temperature
h = 6.62606957e-34  # Planck's constant
kB = 1.3806488e-23  # Boltzmann constant
Ghz_Kelvin = h / kB * 1e9


# ---------------------------------------------------------#
#                  INITIALISATION METHOD                   #
# ---------------------------------------------------------#

# Initialise the foreground data
# NEED TO READ INPUT TEMPLATES
class SPT3G_2018_TTTEEE_Ini_Foregrounds(HasLogger):
    # Default ell range matching window files, but can be adjusted
    SPT3G_windows_lmin = 1
    SPT3G_windows_lmax = 3200

    # Indices specifying order of fg components in output array
    N_fg_max = 9

    def __init__(
        self,
        data_folder="",
        galdust_nu0=150,
        galdust_T=19.6,
        CIB_nu0=150,
        CIB_T=25.0,
        tSZ_nu0=143,
        tSZ_template_filename="",
        kSZ_template_filename="",
        # etSZCosmologyScalingEnabled=False,
        # ekSZCosmologyScalingEnabled=False,
    ):

        self.set_logger()
        self.nu_0_CIB = CIB_nu0
        self.T_CIB = CIB_T
        self.nu_0_tSZ = tSZ_nu0
        self.nu_0_galdust = galdust_nu0
        self.T_galdust = galdust_T

        # Cosmology scaling not supported
        # tSZCosmologyScalingEnabled = False
        # kSZCosmologyScalingEnabled = False

        # Read in tSZ template and normalise
        ltSZ, dl_tSZ = np.loadtxt(os.path.join(data_folder, tSZ_template_filename), unpack=True)
        self.tSZ_template = dl_tSZ / dl_tSZ[ltSZ == 3000]  # Ensure normalisation

        # Read in kSZ template and normalise
        lkSZ, dl_kSZ = np.loadtxt(os.path.join(data_folder, kSZ_template_filename), unpack=True)
        self.kSZ_template = dl_kSZ / dl_kSZ[lkSZ == 3000]  # Ensure normalisation

    # ---------------------------------------------------------#
    #                  FOREGROUND FUNCTIONS                    #
    # ---------------------------------------------------------#
    # Dl_theory(SPT3G_windows_lmin:SPT3G_windows_lmax)
    # Dl_foregrounds(N_fg_max,SPT3G_windows_lmin:SPT3G_windows_lmax)

    # Calibration
    # Data is scaled as: TT: T1*T2, TE: 0.5*(T1*E2+T2*E1), EE: E1*E2
    # Theory is scaled by the inverse
    # In function this is calculated as  0.5*(cal1*cal2+cal3*cal4)
    def ApplyCalibration(self, cal1, cal2, cal3, cal4):

        # This is how the data spectra are calibrated
        calibration = 0.5 * (cal1 * cal2 + cal3 * cal4)
        return calibration

    # Add Poisson power, referenced at ell=3000
    def PoissonPower(self, pow_at_3000):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Calculate and add Poisson power
        Dl_Poisson = ells * ells * pow_at_3000 / (3000 * 3000)

        return Dl_Poisson

    # Add galactic dust (intensity and polarisation)
    # Referenced at ell=80, with power law dependence (alpha+2)
    # At effective frequencies nu1 and nu2, with spectral index beta
    def GalacticDust(self, pow_at_80, alpha, beta, nu1, nu2):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Calculate and add galactic dust power
        Dl_galdust = pow_at_80 * (ells / 80) ** (alpha + 2.0)
        Dl_galdust = (
            Dl_galdust
            * DustFreqScaling(beta, self.T_galdust, self.nu_0_galdust, nu1)
            * DustFreqScaling(beta, self.T_galdust, self.nu_0_galdust, nu2)
        )

        return Dl_galdust

    # Add CIB clustering
    # Referenced at ell=300, with power law dependence (alpha)
    # At effective frequencies nu1 and nu2, with spectral index beta
    # Decorrelation parameters zeta1 and zeta2
    def CIBClustering(self, pow_at_3000, alpha, beta, nu1, nu2, zeta1, zeta2):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Calculate and add polarised galactic dust power
        Dl_cib_clustering = pow_at_3000 * (ells / 3000) ** (alpha)
        Dl_cib_clustering = (
            Dl_cib_clustering
            * DustFreqScaling(beta, self.T_CIB, self.nu_0_CIB, nu1)
            * DustFreqScaling(beta, self.T_CIB, self.nu_0_CIB, nu2)
        )
        Dl_cib_clustering = Dl_cib_clustering * np.sqrt(zeta1 * zeta2)

        return Dl_cib_clustering

    # Add tSZ contribution
    # Template normalised at ell=3000
    # Cosmology scaling not supported
    def tSZ(self, pow_at_3000, nu1, nu2, cosmo=None):

        # Calculate tSZ power
        Dl_tSZ = pow_at_3000 * self.tSZ_template  # Template
        Dl_tSZ = (
            Dl_tSZ
            * tSZFrequencyScaling(nu1, self.nu_0_tSZ, T_CMB)
            * tSZFrequencyScaling(nu2, self.nu_0_tSZ, T_CMB)
        )  # Frequency scaling

        # Cosmology scaling
        # if tSZCosmologyScalingEnabled:
        #  Dl_tSZ = Dl_tSZ * tSZCosmologyScaling(cosmo['H0'], cosmo['sigma_8'], cosmo['omb'])

        return Dl_tSZ

    # Correlation between tSZ and CIB
    # Only use clustered CIB component here and simplified model
    # Sorry for the horrible call signature
    def tSZCIBCorrelation(
        self,
        xi_tsz_CIB,
        tsz_pow_at_3000,
        CIB_pow_at_3000,
        alpha,
        beta,
        zeta1,
        zeta2,
        CIB_nu1,
        CIB_nu2,
        tSZ_nu1,
        tSZ_nu2,
        cosmo=None,
    ):

        # Calculate CIB components
        Dl_cib_clustering_11 = self.CIBClustering(
            CIB_pow_at_3000, alpha, beta, CIB_nu1, CIB_nu1, zeta1, zeta1
        )
        Dl_cib_clustering_22 = self.CIBClustering(
            CIB_pow_at_3000, alpha, beta, CIB_nu2, CIB_nu2, zeta2, zeta2
        )

        # Calculate the tSZ components
        Dl_tSZ_11 = self.tSZ(tsz_pow_at_3000, tSZ_nu1, tSZ_nu1, cosmo=cosmo)
        Dl_tSZ_22 = self.tSZ(tsz_pow_at_3000, tSZ_nu2, tSZ_nu2, cosmo=cosmo)

        # Calculate tSZ-CIB correlation
        # Sign defined such that a positive xi corresponds to a reduction at 150GHz
        with np.errstate(invalid="ignore"):
            Dl_tSZ_CIB_corr = (
                -1
                * xi_tsz_CIB
                * (
                    np.sqrt(Dl_tSZ_11 * Dl_cib_clustering_22)
                    + np.sqrt(Dl_tSZ_22 * Dl_cib_clustering_11)
                )
            )

        return Dl_tSZ_CIB_corr

    # Add kSZ contribution
    # Template normalised at ell=3000
    # Cosmology scaling not supported
    def kSZ(self, pow_at_3000, cosmo=None):

        # Calculate kSZ power
        Dl_kSZ = pow_at_3000 * self.kSZ_template  # Template

        # Cosmology scaling
        # if kSZCosmologyScalingEnabled:
        #  Dl_kSZ = Dl_kSZ * kSZCosmologyScaling(cosmo['H0'], cosmo['sigma_8'], cosmo['omb'], cosmo['omegam'], cosmo['ns'], cosmo['tau'])

        return Dl_kSZ

    # Super sample lensing
    # Based on Manzotti et al. 2014 (https://arxiv.org/pdf/1401.7992.pdf) Eq. 32
    # Applies correction to the spectrum and returns the correction slotted into the fg array
    def ApplySuperSampleLensing(self, kappa, Dl_theory):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Grab Cl derivative
        Cl_derivative = self._GetClDerivative(Dl_theory)

        # Calculate super sample lensing correction
        # (In Cl space) SSL = -k/l^2 d/dln(l) (l^2Cl) = -k(l*dCl/dl + 2Cl)
        ssl_correction = ells * Cl_derivative  # l*dCl/dl
        ssl_correction = (
            ssl_correction * ells * (ells + 1) / (2 * np.pi)
        )  # Convert this part to Dl space already
        ssl_correction = ssl_correction + 2 * Dl_theory  # 2Cl - but already converted to Dl
        ssl_correction = -kappa * ssl_correction  # -kappa

        return ssl_correction

    # Aberration Correction
    # Based on Jeong et al. 2013 (https://arxiv.org/pdf/1309.2285.pdf) Eq. 23
    # Applies correction to the spectrum and returns the correction by itself
    def ApplyAberrationCorrection(self, ab_coeff, Dl_theory):
        # AC = beta*l(l+1)dCl/dln(l)/(2pi)

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Grab Cl derivative
        Cl_derivative = self._GetClDerivative(Dl_theory)

        # Calculate aberration correction
        # (In Cl space) AC = -coeff*dCl/dln(l) = -coeff*l*dCl/dl
        # where coeff contains the boost amplitude and direction (beta*<cos(theta)> in Jeong+ 13)
        aberration_correction = -ab_coeff * Cl_derivative * ells
        aberration_correction = (
            aberration_correction * ells * (ells + 1) / (2 * np.pi)
        )  # Convert to Dl

        return aberration_correction

    # Helper to get the derivative of the spectrum
    # Takes Dl in, but returns Cl derivative#
    # Handles end points approximately
    # Smoothes any spike at the ell_max
    def _GetClDerivative(self, Dl_theory):

        # Grab ells helper (1-3200)
        ells = np.arange(1, self.SPT3G_windows_lmax + 1)

        # Calculate derivative
        Cl_derivative = Dl_theory * 2 * np.pi / (ells * (ells + 1))  # Convert to Cl
        Cl_derivative[1:-1] = 0.5 * (Cl_derivative[2:] - Cl_derivative[:-2])  # Find gradient
        Cl_derivative[0] = Cl_derivative[1]  # Handle start approximately
        Cl_derivative[-1] = Cl_derivative[-2]  # Handle end approximately

        # Smooth over spike at lmax
        # Transition point between Boltzmann solver Cl and where the spectrum comes from a lookup table/interpolation can cause a spike in derivative
        #  if (CosmoSettings%lmax_computed_cl .LT. SPT3G_windows_lmax-1) then
        #    Cl_derivative(CosmoSettings%lmax_computed_cl) = 0.75*Cl_derivative(CosmoSettings%lmax_computed_cl-1) + 0.25*Cl_derivative(CosmoSettings%lmax_computed_cl+2)
        #    Cl_derivative(CosmoSettings%lmax_computed_cl+1) = 0.75*Cl_derivative(CosmoSettings%lmax_computed_cl+2) + 0.25*Cl_derivative(CosmoSettings%lmax_computed_cl-1)

        return Cl_derivative


# ---------------------------------------------------------#
#                     SCALING HELPERS                      #
# ---------------------------------------------------------#

# Galactic Dust Frequency Scaling
def DustFreqScaling(beta, Tdust, nu0, nu_eff):
    fdust = (nu_eff / nu0) ** beta
    fdust = fdust * Bnu(nu_eff, nu0, Tdust) / dBdT(nu_eff, nu0, T_CMB)
    return fdust


# Planck function normalised to 1 at nu0
def Bnu(nu, nu0, T):
    Bnu = (nu / nu0) ** 3
    Bnu = Bnu * (np.exp(Ghz_Kelvin * nu0 / T) - 1.0) / (np.exp(Ghz_Kelvin * nu / T) - 1.0)
    return Bnu


# Derivative of Planck function normalised to 1 at nu0
def dBdT(nu, nu0, T):
    x0 = Ghz_Kelvin * nu0 / T
    x = Ghz_Kelvin * nu / T

    dBdT0 = x0**4 * np.exp(x0) / (np.exp(x0) - 1) ** 2
    dBdT = x**4 * np.exp(x) / (np.exp(x) - 1) ** 2

    return dBdT / dBdT0


# tSZ Frequency Scaling
# Gives conversion factor for frequency nu from reference nu0
def tSZFrequencyScaling(nu, nu0, T):
    x0 = Ghz_Kelvin * nu0 / T
    x = Ghz_Kelvin * nu / T

    tSZfac0 = x0 * (np.exp(x0) + 1) / (np.exp(x0) - 1) - 4
    tSZfac = x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4

    return tSZfac / tSZfac0


# Taken from Reichardt et al. 2020 likelihood
# NOT SUPPORTED - DO NOT USE
def tSZCosmologyScaling(H0, sigma8, omegab):
    return ((H0 / 71.0) ** 1.73) * ((sigma8 / 0.8) ** 8.34) * ((omegab / 0.044) ** 2.81)


# Taken from Reichardt et al. 2020 likelihood
# NOT SUPPORTED - DO NOT USE
def kSZCosmologyScaling(H0, sigma8, omegab, omegam, ns, tau):
    return (
        ((H0 / 71.0) ** 1.7)
        * ((sigma8 / 0.8) ** 4.7)
        * ((omegab / 0.044) ** 2.1)
        * ((omegam / 0.264) ** (-0.44))
        * ((ns / 0.96) ** (-0.19))
    )
