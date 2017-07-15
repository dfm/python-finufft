# -*- coding: utf-8 -*-

from __future__ import division, print_function

import finufft
from finufft import interface

import numpy as np

import pytest

__all__ = [
    "test_nufft2d1", "test_nufft2d2", "test_nufft2d3",
]

def test_nufft2d1(seed=42, iflag=1):
    np.random.seed(seed)

    ms = 100
    mt = 89
    n = int(1e3)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    f = finufft.nufft2d1(x, y, c, ms, mt, eps=tol, iflag=iflag)
    f0 = interface.dirft2d1(x, y, c, ms, mt, iflag=iflag)
    assert np.all(np.abs((f - f0) / f0) < 1e-6)


def test_nufft2d2(seed=42, iflag=1):
    np.random.seed(seed)

    ms = 100
    mt = 89
    n = int(1e3)
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    f = finufft.nufft2d1(x, y, c, ms, mt, eps=tol, iflag=iflag)

    c = finufft.nufft2d2(x, y, f, eps=tol, iflag=iflag)
    c0 = interface.dirft2d2(x, y, f, iflag=iflag)
    assert np.all(np.abs((c - c0) / c0) < 1e-6)


def test_nufft2d3(seed=42, iflag=1):
    np.random.seed(seed)

    ms = 100
    n = 500
    tol = 1.0e-9

    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.random.uniform(-np.pi, np.pi, n)
    c = np.random.uniform(-1.0, 1.0, n) + 1.0j*np.random.uniform(-1.0, 1.0, n)
    s = 0.5 * n * (1.7 + np.random.uniform(-1.0, 1.0, ms))
    t = 0.5 * n * (-0.5 + np.random.uniform(-1.0, 1.0, ms))

    f = finufft.nufft2d3(x, y, c, s, t, eps=tol, iflag=iflag)
    f0 = interface.dirft2d3(x, y, c, s, t, iflag=iflag)
    assert np.all(np.abs((f - f0) / f0) < 1e-6)
