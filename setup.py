from setuptools import setup

setup(
    name='qopt',
    version='1.3.2',
    packages=['qopt'],
    url='https://github.com/qutech/qopt',
    license='GLP3',
    author='Julian Teske',
    author_email='j.teske@fz-juelich.de',
    description='Qubit Simulation and Optimal Control for Quantum Systems',
    package_dir={'qopt': 'qopt'},
    install_requires=['numpy', 'scipy', 'matplotlib', 'filter_functions>=1.1.2'],
    extras_require={
        'doc': ['ipython', 'ipykernel', 'nbsphinx', 'numpydoc', 'sphinx',
                'jupyter_client', 'sphinx_rtd_theme'],
        'qopt_tests': ['pytest', 'pytest_cov'],
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
