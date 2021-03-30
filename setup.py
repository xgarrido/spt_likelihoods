from setuptools import find_packages, setup

setup(
    name="spt",
    version="1.0",
    description="SPT likelihoods for cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "astropy",
        "cobaya>=3.0.4",
    ],
    package_data={f"{lkl}": ["*.yaml", "*.bibtex"] for lkl in ["sptpol_2017", "spt_hiell_2020"]},
)
