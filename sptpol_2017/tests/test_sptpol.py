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
    czero_psTE_150=0.0,
    czero_psEE_150=0.0837416,
    ADust_TE=0.1647,
    ADust_EE=0.0236,
    alphaDust_TE=-2.42,
    alphaDust_EE=-2.42,
    mapTcal=1.0,
    mapPcal=1.0,
    beam1=0.0,
    beam2=0.0,
)


class SPTPolTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"sptpol_2017.TEEE": None}}, path=packages_path)

    def test_cobaya(self):
        """Test the Cobaya interface to the SPTPol likelihood."""
        from cobaya.model import get_model

        info = {
            "debug": True,
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": {**cosmo_params, **fg_params},
            "modules": packages_path,
        }

        for use_cl, expected_chi2 in {
            "teee": 162.9840,
            "te": 74.7255,
            "ee": 76.80106189735758,
        }.items():
            print("use_cl", use_cl)
            info["likelihood"] = {f"sptpol_2017.{use_cl.upper()}": None}
            model = get_model(info)
            chi2 = -2 * model.loglike({})[0]
            self.assertAlmostEqual(chi2, expected_chi2, 3)


if __name__ == "__main__":
    unittest.main()
