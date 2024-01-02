from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'This package is made to perfome topology optimization of 2D solids'
LONG_DESCRIPTION = 'This package is made to perfome topology optimization of 2D solids'

requirements = ['numpy',
                'scipy',
                'matplotlib',
                'easygui',
                'meshio==3.0',
                'tensorflow==2.15.0',]

# Configurando
setup(
        name="SolidsOpt", 
        version=VERSION,
        author="kevin Sepúlveda-García",
        author_email="<kssgarcia@outlook.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=requirements,
        
        keywords=['python', 'primer paquete'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
)