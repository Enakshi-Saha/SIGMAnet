from setuptools import setup, find_packages

setup(
    name='siren',
    version='0.1.0',
    author='Enakshi Saha',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'netZooPy',
    ],
)
