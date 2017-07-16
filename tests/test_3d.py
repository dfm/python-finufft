# -*- coding: utf-8 -*-

from __future__ import division, print_function

import finufft
from finufft import interface

import numpy as np

__all__ = [
    "test_nufft3d1", "test_nufft3d2", "test_nufft3d3",
]

def test_nufft3d1(seed=42, iflag=1):
    np.random.seed(seed)

    m1 = 20
    m2 = 40
    m3 = 50
    n = int(1e3)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    z = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)

    f = finufft.nufft3d1(x, y, z, c, m1, m2, m3, eps=tol, iflag=iflag)
    f0 = interface.dirft3d1(x, y, z, c, m1, m2, m3, iflag=iflag)

    assert np.all(np.abs((f - f0) / f0) < 2e-6)


def test_nufft3d2(seed=42, iflag=1):
    np.random.seed(seed)

    m1 = 20
    m2 = 40
    m3 = 50
    n = int(1e3)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    z = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    f = finufft.nufft3d1(x, y, z, c, m1, m2, m3, eps=tol, iflag=iflag)

    c = finufft.nufft3d2(x, y, z, f, eps=tol, iflag=iflag)
    c0 = interface.dirft3d2(x, y, z, f, iflag=iflag)
    assert np.all(np.abs((c - c0) / c0) < 1e-6)


def test_nufft3d3(seed=42, iflag=1):
    np.random.seed(seed)

    ms = 50
    n = 100
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    z = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)

    s = 0.5 * ms * (1.7 + np.random.uniform(-1.0, 1.0, ms))
    t = 0.5 * ms * (-0.5 + np.random.uniform(-1.0, 1.0, ms))
    u = 0.5 * ms * (0.9 + np.random.uniform(-1.0, 1.0, ms))

    f = finufft.nufft3d3(x, y, z, c, s, t, u, eps=tol, iflag=iflag)
    f0 = interface.dirft3d3(x, y, z, c, s, t, u, iflag=iflag)
    assert np.all(np.abs((f - f0) / f0) < 1e-6)
