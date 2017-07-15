Python bindings to the `Flatiron Institute Non-Uniform FFT (FINUFFT) library
<https://github.com/ahbarnett/finufft>`_.

.. image:: https://travis-ci.org/dfm/python-finufft.svg?branch=master&style=flat
    :target: https://travis-ci.org/dfm/python-finufft
.. image:: http://readthedocs.org/projects/finufft/badge/?version=latest&style=flat
    :target: http://finufft.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Installation
------------

If you're using `conda <https://conda.io>`_, you can install the prerequisites as follows:

.. code-block:: bash

    conda install -c conda-forge numpy pybind11

Then to build:

.. code-block:: bash

    git clone --recursive https://github.com/dfm/python-finufft.git
    cd python-finufft
    python setup.py install
    
**Note for macOS users:** The standard compilers on macOS are not compatible with
OpenMP so you'll need to install GCC to get OpenMP support. Using `Homebrew
<https://brew.sh/>`_:

.. code-block:: bash

    brew reinstall gcc --without-multilib
    brew reinstall fftw --with-openmp
    CC=/usr/local/bin/gcc-6 CXX=/usr/local/bin/g++-6 python setup.py install
