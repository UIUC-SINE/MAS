from setuptools import setup

setup(
    name='uiuc-mas',
    version='2',
    packages=['mas'],
    author="Evan Widloski",
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
        "opencv-python"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
