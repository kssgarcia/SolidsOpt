from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'This package is made to perfome topology optimization of 2D solids'
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION =  f.read()

requirements = ['numpy',
                'scipy',
                'matplotlib',
                'easygui',
                'meshio==3.0',
                'tensorflow==2.15.0',
                'solidspy']

# Configurando
setup(
    name="SolidsOpt", 
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    url='https://github.com/kssgarcia/SolidsOpt',
    author='Kevin Sepúlveda-García <kssepulveg@eafit.edu.co>, Nicolas Guarin-Zapata <nguarinz@eafit.edu.co>',
    author_email='kssepulveg@eafit.edu.co',
    license='MIT',
    keywords=['finite-elements', 'scientific-computing', 'deep learning', 'topology', 'optimization'],
    classifiers= [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)