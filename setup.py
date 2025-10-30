from setuptools import setup, find_packages
from pathlib import Path
# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='pygid',
    version='0.2.0',
    author='Ainur Abukaev',
    description='Fast Preprocessing of Grazing Incidence Diffraction Data',
    python_requires='>=3.8',
    author_email='ainurabukaev@gmail.com',
    url='https://github.com/mlgid-project/pygid',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0,<3.0',
        'h5py>=3.12.1,<4.0',
        'fabio>=2024.9.0,<2025.0',
        'tifffile>=2023.7.18,<2024.0',
        'matplotlib>=3.10.0,<4.0',
        'opencv-python>=4.10.0.82,<5.0',
        'numexpr>=2.10.0,<3.0',
        'streamz>=0.6.3,<1.0',
        'pytest>=8.2.0,<9.0',
        'tqdm>=4.66.4,<5.0',
        'joblib>=1.4.2,<2.0',
        'PyYAML>=6.0.1,<7.0',
        'adjustText>=1.3.0,<2.0',
        'typing',
        'ipywidgets>=8.1.7,<9.0',
    ],
    extras_require={
        'test': [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'pytest-xdist>=3.0.0',
                'pytest-mock>=3.10.0',
            ]
    }
)
