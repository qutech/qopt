from setuptools import setup

setup(
    name='qopt',
    version='0.1',
    packages=['qopt'],
    url='https://git-ce.rwth-aachen.de/qutech/qopt',
    license='GLP3',
    author='Julian Teske',
    author_email='j.teske@fz-juelich.de',
    description='Optimal Control for Quantum Systems',
    package_dir={'qopt': 'qopt'},
    install_requires=['numpy', 'scipy', 'matplotlib', 'cython', 'nose',
                      'simanneal', 'pandas', 'spyder',
                      'filter_functions', 'qutip', 'opt_einsum', 'sparse'],
    extras_require={
        'fancy_progressbar': ['tqdm', 'requests'],
        'doc': ['ipython', 'ipykernel', 'nbsphinx', 'numpydoc', 'sphinx',
                'jupyter_client', 'sphinx_rtd_theme'],
        'tests': ['pytest', 'coverage', 'coveralls'],
    },
    test_suite='tests',
    classifiers=[
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: GNU General Public License v3 or later '
      '(GPLv3+)',
      'Operating System :: OS Independent',
      'Topic :: Scientific/Engineering :: Physics',
    ]
)
