import unittest

import jax.numpy as jnp
from qiskit.quantum_info import random_density_matrix, entropy

from tn.explain import von_neumann_entropy

ATOL=1e-3

class TestVNE(unittest.TestCase):
    def get_entropies(self, sites):
        """Get expected and actual von Neumann entropy of a random state with n sites."""
        dims = 2**sites

        rho = random_density_matrix(dims=dims)
        expected = entropy(rho, base=jnp.e)
        actual = von_neumann_entropy(rho.reverse_qargs().data)

        return expected, actual

    def test_single_site(self):
        """Test von Neumann entropy for a single site."""
        expected, actual = self.get_entropies(sites=1)

        self.assertTrue(jnp.isclose(expected - actual, 0., atol=ATOL), msg=f"{expected} - {actual}")

    def test_two_sites(self):
        """Test von Neumann entropy for two sites."""
        expected, actual = self.get_entropies(sites=2)

        self.assertTrue(jnp.isclose(expected - actual, 0., atol=ATOL), msg=f"{expected} - {actual}")

    def test_multiple_sites(self):
        """Test von Neumann entropy for miltiple sites."""
        expected, actual = self.get_entropies(sites=4)

        self.assertTrue(jnp.isclose(expected - actual, 0., atol=ATOL), msg=f"{expected} - {actual}")
