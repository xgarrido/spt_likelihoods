import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)

cosmo_params = dict(
    cosmomc_theta=0.0104025,
    As=1e-10 * np.exp(3.037),
    ombh2=0.02224,
    omch2=0.1166,
    ns=0.970,
    tau=0.054,
)

fg_params = dict(
    kappa=0.0,
    Tcal90=1.0,
    Tcal150=0.9975,
    Tcal220=0.9930,
    Ecal90=1.0009,
    Ecal150=1.0020,
    Ecal220=1.019,
    EE_Poisson_90x90=0.041,
    EE_Poisson_90x150=0.0177,
    EE_Poisson_90x220=0.0157,
    EE_Poisson_150x150=0.0115,
    EE_Poisson_150x220=0.0188,
    EE_Poisson_220x220=0.048,
    EE_PolGalDust_Amp=0.052,
    EE_PolGalDust_Alpha=-2.42,
    EE_PolGalDust_Beta=1.51,
    TE_PolGalDust_Amp=0.138,
    TE_PolGalDust_Alpha=-2.42,
    TE_PolGalDust_Beta=1.51,
    TT_Poisson_90x90=62.61,
    TT_Poisson_90x150=27.9,
    TT_Poisson_90x220=24.3,
    TT_Poisson_150x150=16.7,
    TT_Poisson_150x220=28.6,
    TT_Poisson_220x220=78.5,
    TT_GalCirrus_Amp=1.93,
    TT_GalCirrus_Alpha=-2.53,
    TT_GalCirrus_Beta=1.48,
    TT_CIBClustering_Amp=5.2,
    TT_CIBClustering_Alpha=0.8,
    TT_CIBClustering_Beta=1.85,
    TT_CIBClustering_decorr_90=1.0,
    TT_CIBClustering_decorr_150=1.0,
    TT_CIBClustering_decorr_220=1.0,
    TT_tSZ_Amp=4.7,
    TT_tSZ_CIB_corr=0.09,
    TT_kSZ_Amp=3.7,
)


class SPT3GTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"spt3g_2022.TTTEEE": None}}, path=packages_path)

    def test_with_cosmomc_data(self):
        from spt3g_2022 import TTTEEE

        my_spt = TTTEEE(dict(packages_path=packages_path))

        # Read Dl generated by CosmoMC
        test_dir = os.path.dirname(__file__)
        dl_tt, dl_te, dl_ee = np.load(os.path.join(test_dir, "data", "dl_from_cosmomc.npy"))
        loglike = my_spt.loglike({"TT": dl_tt, "TE": dl_te, "EE": dl_ee}, **fg_params)
        self.assertAlmostEqual(-2 * loglike, 1876.4526255716014, 5)

    def test_cobaya(self):
        """Test the Cobaya interface to the SPT3G likelihood."""
        from cobaya.model import get_model

        info = {
            "debug": True,
            "likelihood": {"spt3g_2022.TTTEEE": None},
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": {**cosmo_params, **fg_params},
            "packages_path": packages_path,
        }

        model = get_model(info)
        chi2 = -2 * model.loglike({})[0]
        self.assertAlmostEqual(chi2, 1874.4356, 2)


if __name__ == "__main__":
    unittest.main()
