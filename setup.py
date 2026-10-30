from setuptools import setup, find_packages

setup(
    name="GGAE",
    version="0.1.0",
    description="GG-AE: Genetically-Guided Autoencoder for Disentangling "
                "Structural Networks of Brain Atrophy",
    author="Marilena De Pian",
    author_email="depianm@pennmedicine.upenn.edu",
    url="https://github.com/marilenadp/GG-AE",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12",
        "numpy",
        "scipy",
        "pandas",
        "nibabel",
    ],
)
