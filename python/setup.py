from setuptools import setup

setup(
    name='mas',
    packages=['./'],
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
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
