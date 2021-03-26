from setuptools import find_packages, setup

setup(
    name="spt",
    version="1.0",
    description="SPT likelihoods for cobaya",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "cobaya>=3.0.4",
    ],
    package_data={"spt": ["SPTPol.yaml", "SPTPol.bibtex"]},
)
