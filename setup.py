from setuptools import setup

setup(
    name='qopt',
    version='1.0',
    packages=['qopt'],
    url='https://git-ce.rwth-aachen.de/qutech/qopt',
    license='GLP3',
    author='Julian Teske',
    author_email='j.teske@fz-juelich.de',
    description='Optimal Control for Quantum Systems',
    package_dir={'qopt': 'qopt'},
    install_requires=['numpy', 'scipy', 'matplotlib', 'cython',
                      'simanneal', 'pandas', 'spyder', 'jupyter', 'notebook',
                      'filter_functions', 'qutip'],
    extras_require={
        'doc': ['ipython', 'ipykernel', 'nbsphinx', 'numpydoc', 'sphinx',
                'jupyter_client', 'sphinx_rtd_theme'],
        'tests': ['pytest'],
    },
    test_suite='qopt_tests',
    classifiers=[
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: GNU General Public License v3 or later '
      '(GPLv3+)',
      'Operating System :: OS Independent',
      'Topic :: Scientific/Engineering :: Physics',
    ]
)
