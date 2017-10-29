# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
    

def build_poly(x, deg):
    """polynomial basis functions."""
    mat = np.ones((len(x), deg+1))
    for d in range(1, deg+1):
        mat[:,d] = x**d;
    return mat