import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)

cosmo_params = dict(
    cosmomc_theta=0.010411,
    As=1e-10 * np.exp(3.05),
    ombh2=0.0221,
    omch2=0.1200,
    ns=0.96,
    Alens=1.0,
    tau=0.06,
)

nuisance_params = dict(
    mapCal95=1.0,
    mapCal150=1.0,
    mapCal220=1.0,
    FTS_calibration_error=0.0,
    czero_tsz=3.5,
    czero_ksz=3.9,
    czero_ksz2=0,
    czero_dg_po=8.6,
    T_dg_po=25,
    beta_dg_po=1.5,
    sigmasq_dg_po=0.1,
    czero_dg_cl=3.4,
    T_dg_cl=25,
    beta_dg_cl=0.5,
    sigmasq_dg_cl=0.0,
    czero_dg_cl2=1.0,
    T_dg_cl2=25,
    beta_dg_cl2=0.5,
    sigmasq_dg_cl2=0.0,
    czero_rg_po=1.0,
    czero_rg_cl=0.0,
    alpha_rg=-0.7,
    sigmasq_rg=0.0,
    tsz_dg_cor=0.07,
    tsz_cib_slope=0.0,
    tsz_rg_cor=0.0,
    czero_cirrus=2.19,
    T_cirrus=25,
    beta_cirrus=1.5,
)


class SPTLikeTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"spt_hiell_2020.TT": None}}, path=packages_path)

    def test_spt_hiell(self):
        import camb
        # camb_cosmo = cosmo_params.copy()
        # camb_cosmo.update({"lmax": 9000, "lens_potential_accuracy": 1})
        # pars = camb.set_params(**camb_cosmo)
        # results = camb.get_results(pars)
        # powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        # cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}
        from spt_hiell_2020 import TT

        my_spt = TT(dict(packages_path=packages_path))
        # loglike = my_mflike.loglike(cl_dict, **nuisance_params)
        # self.assertAlmostEqual(-2 * (loglike - my_mflike.logp_const), chi2, 2)

    def test_cobaya(self):
        info = dict(
            debug=True,
            likelihood={"spt_hiell_2020.TT": None},
            theory=dict(camb={"extra_args": {"lens_potential_accuracy": 1}}),
            params=cosmo_params,
            modules=packages_path,
        )
        from cobaya.model import get_model

        model = get_model(info)
        my_spt = model.likelihood["spt_hiell_2020.TT"]
        chi2 = -2 * (model.loglikes(nuisance_params)[0])
        self.assertAlmostEqual(chi2[0], 1289.6487995333528, 3)


if __name__ == "__main__":
    unittest.main()
