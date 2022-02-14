from setuptools import setup, find_packages
import version

setup(
    name='uiuc-mas',
    version=version.__version__,
    packages=find_packages(),
    author="Evan Widloski, Ulas Kamaci",
    author_email="evan@evanw.org",
    description="Milliarc-second Photon Sieve Simulations",
    long_description=open('README.md').read(),
    license="GPLv3",
    keywords="photonsieve csbs sieve uiuc",
    url="https://github.com/uiuc-sine/mas",
    install_requires=[
        "matplotlib",
        "h5py",
        "numpy",
        "scipy",
        "imageio",
        "opencv-python",
        "tqdm",
        "pandas",
        "scikit-image==0.17.2",
        "cachalot",
        "pyabel",
        # "pybm3d"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
