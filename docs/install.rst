.. _install:

Installation
============

From Source
-----------

The source code for python-finufft can be downloaded `from GitHub
<https://github.com/dfm/python-finufft>`_ by running

.. code-block:: bash

    git clone --recursive https://github.com/dfm/python-finufft.git


.. _python-deps:

Python Dependencies
+++++++++++++++++++

For this Python interface, you'll (obviously) need a Python installation and I
recommend `conda <http://continuum.io/downloads>`_ if you don't already have
your own opinions.

After installing Python, the following dependencies are required to build:

1. `NumPy <http://www.numpy.org/>`_ for math and linear algebra in Python, and
2. `pybind11 <https://pybind11.readthedocs.io>`_ for the Python–C++ interface.

If you're using conda, you can install these with the following command:

.. code-block:: bash

    conda install -c conda-forge numpy pybind11

Other Dependencies & Building
+++++++++++++++++++++++++++++

You'll also need `FFTW3 <http://www.fftw.org/>`_ and you'll get the most out
of this if you also include OpenMP support.

On macOS with `Homebrew <https://brew.sh/>`_ you'll get everything as follows:

.. code-block:: bash

    brew install gcc --without-multilib
    brew install fftw --with-openmp
    CC=/usr/local/bin/gcc-7 CXX=/usr/local/bin/g++-7 python setup.py install

.. note:: More platforms TBD...


Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then, in
the root directory of the repository, execute:

.. code-block:: bash

    py.test -v tests

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/python-finufft/issues>`_.
