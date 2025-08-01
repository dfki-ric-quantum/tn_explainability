import random
import sys
import unittest

import jax
import jax.numpy as jnp

from tn.mps.ops import (
    svd_normalize_left,
    svd_normalize_right,
    trunc_svd_normalize_left,
    trunc_svd_normalize_right,
)

ATOL = 1e-4
MAX_BOND = 10
SIGMA_THRESH = 1e-3


def get_random_tensors() -> tuple[jax.Array, jax.Array]:
    key = jax.random.key(random.randint(0, sys.maxsize))
    key, skl, skr = jax.random.split(key, 3)

    left = jax.random.uniform(skl, (50, 2, 25))
    right = jax.random.uniform(skr, (25, 2, 50))

    return left, right


class TestSVD(unittest.TestCase):
    def test_decomp(self):
        """Test contraction+decomposition.

        * Generate two random order-3 tensors
        * Contract them into an order-4 tensors
        * Perform left and right normalized decompositions
        * Re-contract decomposition
        * Results should be equal

        """
        lhs, rhs = get_random_tensors()

        ln_left, ln_right = svd_normalize_left(lhs, rhs)
        rn_left, rn_right = svd_normalize_right(lhs, rhs)

        ln_contracted = jnp.einsum("abc,cde->abde", ln_left, ln_right)
        rn_contracted = jnp.einsum("abc,cde->abde", rn_left, rn_right)

        self.assertTrue(jnp.allclose(ln_contracted, rn_contracted, atol=ATOL))

    def test_left_normalization(self):
        """Test left normalization condition of SVD decomposition."""

        left, right = svd_normalize_left(*get_random_tensors())

        lcont = jnp.einsum("abc,abd->cd", left, jnp.conjugate(left))
        identity = jnp.eye(left.shape[2])

        self.assertTrue(jnp.allclose(lcont, identity, atol=ATOL), msg="left id")
        self.assertTrue(jnp.isclose(jnp.linalg.norm(right), 1.0), msg="right norm")

    def test_right_normalization(self):
        """Test right normalization condition of SVD decomposition."""

        left, right = svd_normalize_right(*get_random_tensors())

        rcont = jnp.einsum("cba,dba->cd", right, jnp.conjugate(right))
        identity = jnp.eye(left.shape[2])

        self.assertTrue(jnp.allclose(rcont, identity, atol=ATOL), msg="right id")
        self.assertTrue(jnp.isclose(jnp.linalg.norm(left), 1.0), msg="left norm")

    def test_trunc_decomp(self):
        """Test contraction + truncated decomposition.

        * Generate two random order-3 tensors
        * Contract them into an order-4 tensors
        * Perform left and right truncated normalized decompositions
        * Re-contract decomposition
        * Results should be equal
        * Shared truncated bonds of decomposed tensors should match

        """
        lhs, rhs = get_random_tensors()

        ln_left, ln_right = trunc_svd_normalize_left(
            lhs, rhs, max_bond=MAX_BOND, sigma_thresh=SIGMA_THRESH
        )
        rn_left, rn_right = trunc_svd_normalize_right(
            lhs, rhs, max_bond=MAX_BOND, sigma_thresh=SIGMA_THRESH
        )

        ln_contracted = jnp.einsum("abc,cde->abde", ln_left, ln_right)
        rn_contracted = jnp.einsum("abc,cde->abde", rn_left, rn_right)

        self.assertTrue(ln_left.shape[2] <= MAX_BOND, msg="left trunc bond")
        self.assertTrue(rn_left.shape[2] <= MAX_BOND, msg="right trunc bond")
        self.assertTrue(jnp.allclose(ln_contracted, rn_contracted, atol=ATOL))

    def test_trunc_left_normalization(self):
        """Test left normalization condition of truncated SVD decomposition."""

        left, right = trunc_svd_normalize_left(
            *get_random_tensors(), max_bond=MAX_BOND, sigma_thresh=SIGMA_THRESH
        )

        lcont = jnp.einsum("abc,abd->cd", left, jnp.conjugate(left))
        identity = jnp.eye(left.shape[2])

        self.assertTrue(jnp.allclose(lcont, identity, atol=ATOL), msg="left id")
        self.assertTrue(jnp.isclose(jnp.linalg.norm(right), 1.0), msg="right norm")

    def test_trunc_right_normalization(self):
        """Test right normalization condition of truncated SVD decomposition."""

        left, right = trunc_svd_normalize_right(
            *get_random_tensors(), max_bond=MAX_BOND, sigma_thresh=SIGMA_THRESH
        )

        rcont = jnp.einsum("cba,dba->cd", right, jnp.conjugate(right))
        identity = jnp.eye(left.shape[2])

        self.assertTrue(jnp.allclose(rcont, identity, atol=ATOL), msg="right id")
        self.assertTrue(jnp.isclose(jnp.linalg.norm(left), 1.0), msg="left norm")
