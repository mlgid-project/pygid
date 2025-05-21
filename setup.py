from setuptools import setup, find_packages


setup(
    name='pygid',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'h5py',
        'fabio',
        'tifffile',
        'matplotlib',
        'opencv-python',
        'numexpr',
        'typing',
        'streamz',
        'pytest',
        'tqdm',
        'joblib',
        'PyYAML'
    ],
)
