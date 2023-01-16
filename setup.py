from setuptools import find_packages, setup

setup(
    name="spt",
    version="1.0",
    description="SPT likelihoods for cobaya",
    author="Xavier Garrido",
    author_email="xavier.garrido@ijclab.in2p3.fr",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "astropy",
        "cobaya>=3.1",
    ],
    package_data={
        f"{lkl}": ["*.yaml", "*.bibtex", "tests/*.py", "tests/data/*.npy"]
        for lkl in ["sptpol_2017", "spt_hiell_2020", "spt3g_2020", "spt3g_2022"]
    },
    scripts=["scripts/test-spt"],
)
