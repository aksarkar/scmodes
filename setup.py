import setuptools

setuptools.setup(
    name='scmodes',
    description='Investigation of single cell modes',
    version='0.2',
    url='https://www.github.com/aksarkar/scmodes',
    author='Abhishek Sarkar',
    author_email='aksarkar@uchicago.edu',
    license='MIT',
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pandas',
        'rpy2',
        'scipy',
        'sklearn',
        'torch',
    ],
    packages=setuptools.find_packages(),
)
