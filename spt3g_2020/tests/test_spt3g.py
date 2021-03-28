import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)

cosmo_params = dict(
    cosmomc_theta=0.010411,
    As=1e-10 * np.exp(3.1),
    ombh2=0.0221,
    omch2=0.1200,
    ns=0.96,
    Alens=1.0,
    tau=0.09,
)

fg_params = dict(
    kappa=0.0,
    Dl_Poisson_90x90=0.1,
    Dl_Poisson_90x150=0.1,
    Dl_Poisson_90x220=0.1,
    Dl_Poisson_150x150=0.1,
    Dl_Poisson_150x220=0.1,
    Dl_Poisson_220x220=0.1,
    ADust_TE_150=0.1647,
    ADust_EE_150=0.0236,
    AlphaDust_TE=-2.42,
    AlphaDust_EE=-2.42,
    mapTcal90=1.0,
    mapTcal150=1.0,
    mapTcal220=1.0,
    mapPcal90=1.0,
    mapPcal150=1.0,
    mapPcal220=1.0,
)


class SPT3GTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"spt3g_2020.TEEE": None}}, path=packages_path)

    def test_cobaya(self):
        """Test the Cobaya interface to the SPT3G likelihood."""
        from cobaya.model import get_model

        info = {
            "debug": True,
            "likelihood": {"spt3g_2020.TEEE": None},
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": {**cosmo_params, **fg_params},
            "modules": packages_path,
        }

        model = get_model(info)
        chi2 = -2 * model.loglike({})[0]
        # self.assertAlmostEqual(chi2, expected_chi2, 3)


if __name__ == "__main__":
    unittest.main()
