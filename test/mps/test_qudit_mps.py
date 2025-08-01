import random
import sys
import unittest

import jax
import jax.numpy as jnp

from tn.mps import random_mps

N_SITES = 12
BOND_DIM = 20
PHYS_DIM = 8
ATOL = 1e-3


class TestMPS(unittest.TestCase):
    def setUp(self):
        """Prepare random MPS for tests."""

        self.key = jax.random.key(random.randint(0, sys.maxsize))
        self.key, subkey = jax.random.split(self.key)

        self.mps = random_mps(
            n_sites=N_SITES, bond_dim=BOND_DIM, random_key=subkey, d=PHYS_DIM
        )

    def test_matching_shapes(self):
        """Test if the tensors in the MPS are compatible with respect to their shared bonds."""
        bond = 1

        for idx, tensor in enumerate(self.mps):
            self.assertEqual(bond, tensor.shape[0], msg=f"Bond mismatch on {idx}")
            bond = tensor.shape[2]

        self.assertEqual(1, self.mps[-1].shape[2], msg="Bond mismatch on end")

    def test_left_canonicalization(self):
        """Test left canonicalization:

        o->o->o->o->...->o

        For all but the last tensor it holds:

        +-o-
        | |  = |
        +-o-

        The last tensor has unit norm.
        """
        self.mps.left_canonicalize()

        for tensor in self.mps[:-1]:
            res = jnp.einsum("abc,abd->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        self.assertTrue(jnp.isclose(jnp.linalg.norm(self.mps[-1]), 1.0), msg="Norm")

    def test_right_canonicalization(self):
        """Test right canonicalization:

        o<-...<-o<-o<-o<-o

        For all but the first tensor it holds:

        -o-+
         | |  = |
        -o-+

        The first tensor has unit norm.
        """
        self.mps.right_canonicalize()

        for tensor in reversed(self.mps[1:]):
            res = jnp.einsum("cba,dba->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        self.assertTrue(jnp.isclose(jnp.linalg.norm(self.mps[0]), 1.0), msg="Norm")

    def test_mixed_canonicalization_inside(self):
        """Test mixed canonicalization:

                      m
        o->o->...->o->o<-o<-o<-...<-o<-o

        For all tensors left of site m, it hold

        +-o-
        | |  = |
        +-o-

        For all tensors right of site m,  it holds:

        -o-+
         | |  = |
        -o-+

        The tensor at site m has unit norm.
        """
        site = random.randint(1, N_SITES - 1)
        self.mps.mixed_canonicalize(site)

        for tensor in self.mps[:site]:
            res = jnp.einsum("abc,abd->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        for tensor in reversed(self.mps[site + 1 :]):
            res = jnp.einsum("cba,dba->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        self.assertTrue(jnp.isclose(jnp.linalg.norm(self.mps[site]), 1.0), msg="Norm")

    def test_mixed_canonicalization_left_boundary(self):
        """Test mixed canonicalization on the left boundary. Identical to right canonicalization."""
        site = 0
        self.mps.mixed_canonicalize(site)

        for tensor in reversed(self.mps[site + 1 :]):
            res = jnp.einsum("cba,dba->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        self.assertTrue(jnp.isclose(jnp.linalg.norm(self.mps[site]), 1.0), msg="Norm")

    def test_mixed_canonicalization_right_boundary(self):
        """Test mixed canonicalization on the right boundary. Identical to left canonicalization."""
        site = N_SITES - 1
        self.mps.mixed_canonicalize(site)

        for tensor in self.mps[:site]:
            res = jnp.einsum("abc,abd->cd", tensor, jnp.conjugate(tensor))
            self.assertTrue(jnp.allclose(res, jnp.eye(res.shape[0]), atol=ATOL))

        self.assertTrue(jnp.isclose(jnp.linalg.norm(self.mps[site]), 1.0), msg="Norm")
