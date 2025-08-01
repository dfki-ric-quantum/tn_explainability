import os
import unittest
from typing import Callable

import numpy as np
from scipy.integrate import quad

from tn.encoding.fourier import FourierBasisFunction
from tn.encoding.hermite_poly import IsoHermitePolynomial
from tn.encoding.laguerre_poly import IsoLaguerrePolynomial
from tn.encoding.legendre_poly import IsoLegendrePolynomial

ATOL = 1e-4

N_FOUR = 10
N_HERM = 10
N_LAGU = 9
N_LEGE = 10


@unittest.skipIf(int(os.getenv("TN_SLOW_TEST", 0)) < 1, "Slow test due to integration")
class TestIsometries(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()

    def test_fourier_baasis(self):
        """Test orthonormality/kernel condition for Fourier basis functions."""
        bounds = (0.0, 1.0)

        f = [FourierBasisFunction(n) for n in range(1, N_FOUR + 1)]
        data = self.rng.uniform(low=0.0, high=1.0, size=(10, 2))

        self.assertTrue(self._is_orthonormal(f, bounds))
        self.assertTrue(self._is_kernel(f, bounds, data))

    def test_hermite_poly(self):
        """Test orthonormality/kernel condition for Hermite Polynomials."""
        bounds = (-np.inf, np.inf)

        f = [IsoHermitePolynomial(n) for n in range(N_HERM)]
        data = self.rng.uniform(low=-10.0, high=10.0, size=(10, 2))

        self.assertTrue(self._is_orthonormal(f, bounds))
        self.assertTrue(self._is_kernel(f, bounds, data))

    def test_laguerre_poly(self):
        """Test orthonormality/kernel condition for Laguerre Polynomials."""
        bounds = (0.0, np.inf)

        f = [IsoLaguerrePolynomial(n) for n in range(N_LAGU)]
        data = self.rng.uniform(low=0.0, high=10.0, size=(10, 2))

        self.assertTrue(self._is_orthonormal(f, bounds))
        self.assertTrue(self._is_kernel(f, bounds, data))

    def test_legendre_poly(self):
        """Test orthonormality/kernel condition for Legendre Polynomials."""
        bounds = (-1.0, 1.0)

        f = [IsoLegendrePolynomial(n) for n in range(N_LEGE)]
        data = self.rng.uniform(low=-1.0, high=1.0, size=(10, 2))

        self.assertTrue(self._is_orthonormal(f, bounds))
        self.assertTrue(self._is_kernel(f, bounds, data))

    def test_legendre_shifted_poly(self):
        """Test orthonormality/kernel condition for shifted Legendre Polynomials."""
        bounds = (0.0, 1.0)

        f = [IsoLegendrePolynomial(n, shifted=True) for n in range(N_LEGE)]
        data = self.rng.uniform(low=-0.0, high=1.0, size=(10, 2))

        self.assertTrue(self._is_orthonormal(f, bounds))
        self.assertTrue(self._is_kernel(f, bounds, data))

    def _is_orthonormal(self, f: list[Callable], bounds: tuple[float, float]) -> bool:
        """Checks if a given set of functions is orthonormal within the specified bounds,
        that is:

            ∫ f_m(x)*f_n(x)dx = δ(m,n)

        with δ(m,n) being the Kronecker delta.

        Parameters
        ----------
        f: list[Callable]
            List of functions.
        bounds: tuple[float float]
            The bounds (lower, upper) for which to check the condition.

        Returns
        -------
        True if the functions are orthonormal, False otherwise.
        """
        n_funcs = len(f)
        low, high = bounds

        m = np.empty((n_funcs, n_funcs))

        for i, fi in enumerate(f):
            for j in range(i, n_funcs):
                m[i, j] = quad(lambda x: fi(x) * f[j](x), low, high, full_output=1)[0]
                m[j, i] = m[i, j]

        return np.allclose(m, np.eye(n_funcs), atol=ATOL)

    def _is_kernel(
        self, f: list[Callable], bounds: tuple[float, float], data: np.ndarray
    ) -> bool:
        """Checks if a set of functions obeys the kernel condition

            Σ f_i(x)*f_i(x') = Π(x, x')

        with ∫ Π(x, x')Π(x', x'')dx' = Π(x, x'').

        Parameters
        ----------
        f: list[Callable]
            List of functions.
        bounds: tuple[float float]
            The bounds (lower, upper) for which to check the condition.
        data: np.ndarray
            Data to check the condition with.

        Returns
        -------
        True if the functions obey the kernel condition, False otherwise.
        """
        low, high = bounds

        def pi(x, y):
            return sum([fi(x) * fi(y) for fi in f])

        def pi_prod(x, y, z):
            return pi(y, x) * pi(x, z)

        res = np.array(
            [
                quad(pi_prod, low, high, args=(x, y), full_output=1)[0] - pi(x, y)
                for x, y in data
            ]
        )

        return np.allclose(res, 0.0, atol=ATOL)
