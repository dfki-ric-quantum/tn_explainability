import random
import sys
import unittest

import jax
import jax.numpy as jnp

from tn.ttn.ops import (
    contract_left_inner_node,
    contract_left_leaf,
    contract_left_with_root,
    contract_right_inner_node,
    contract_right_leaf,
    contract_right_with_root,
    decomp_left_from_root_down,
    decomp_left_from_root_up,
    decomp_left_inner_node_down,
    decomp_left_inner_node_up,
    decomp_left_leaf_down,
    decomp_left_leaf_up,
    decomp_right_from_root_down,
    decomp_right_from_root_up,
    decomp_right_inner_node_down,
    decomp_right_inner_node_up,
    decomp_right_leaf_down,
    decomp_right_leaf_up,
)

ATOL = 1e-3


def ri():
    return random.randint(a=2, b=10)


def get_left_leaf(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    lkey, pkey = jax.random.split(key, 2)

    lshape = (ri(), ri())
    pshape = (lshape[1], ri(), ri())

    leaf = jax.random.uniform(lkey, shape=lshape, minval=-1.0, maxval=1.0)
    parent = jax.random.uniform(pkey, shape=pshape, minval=-1.0, maxval=1.0)

    return leaf, parent


def get_right_leaf(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    lkey, pkey = jax.random.split(key)

    lshape = (ri(), ri())
    pshape = (ri(), ri(), lshape[1])

    leaf = jax.random.uniform(lkey, shape=lshape, minval=-1.0, maxval=1.0)
    parent = jax.random.uniform(pkey, shape=pshape, minval=-1.0, maxval=1.0)

    return leaf, parent


def get_left_inner(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    ckey, pkey = jax.random.split(key)

    cshape = (ri(), ri(), ri())
    pshape = (cshape[1], ri(), ri())

    child = jax.random.uniform(ckey, shape=cshape, minval=-1.0, maxval=1.0)
    parent = jax.random.uniform(pkey, shape=pshape, minval=-1.0, maxval=1.0)

    return child, parent


def get_right_inner(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    ckey, pkey = jax.random.split(key)

    cshape = (ri(), ri(), ri())
    pshape = (ri(), ri(), cshape[1])

    child = jax.random.uniform(ckey, shape=cshape, minval=-1.0, maxval=1.0)
    parent = jax.random.uniform(pkey, shape=pshape, minval=-1.0, maxval=1.0)

    return child, parent


def get_left_with_root(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    ckey, rkey = jax.random.split(key)

    cshape = (ri(), ri(), ri())
    rshape = (cshape[1], ri())

    child = jax.random.uniform(ckey, shape=cshape, minval=-1.0, maxval=1.0)
    root = jax.random.uniform(rkey, shape=rshape, minval=-1.0, maxval=1.0)

    return child, root


def get_right_with_root(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    ckey, rkey = jax.random.split(key)

    cshape = (ri(), ri(), ri())
    rshape = (ri(), cshape[1])

    child = jax.random.uniform(ckey, shape=cshape, minval=-1.0, maxval=1.0)
    root = jax.random.uniform(rkey, shape=rshape, minval=-1.0, maxval=1.0)

    return child, root


class TestTTNOps(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.key(random.randint(0, sys.maxsize))

    def test_left_leaf(self):
        """Test left leaf contraction/decomposition

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        leaf, parent = get_left_leaf(subkey)

        contracted = contract_left_leaf(leaf, parent)

        uleaf, uparent = decomp_left_leaf_up(contracted)
        dleaf, dparent = decomp_left_leaf_down(contracted)

        cu = contract_left_leaf(uleaf, uparent)
        cd = contract_left_leaf(dleaf, dparent)

        lnorm = jnp.einsum("abc,dbc->ad", dparent, dparent)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(
            jnp.allclose(uleaf.T @ uleaf, jnp.eye(uleaf.shape[1]), atol=ATOL)
        )
        self.assertTrue(jnp.allclose(lnorm, jnp.eye(lnorm.shape[0]), atol=ATOL))

    def test_righ_leaf(self):
        """Test right leaf contraction/decomposition

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        leaf, parent = get_right_leaf(subkey)

        contracted = contract_right_leaf(leaf, parent)

        uleaf, uparent = decomp_right_leaf_up(contracted)
        dleaf, dparent = decomp_right_leaf_down(contracted)

        cu = contract_right_leaf(uleaf, uparent)
        cd = contract_right_leaf(dleaf, dparent)

        rnorm = jnp.einsum("abc,abd->cd", dparent, dparent)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(
            jnp.allclose(uleaf @ uleaf.T, jnp.eye(uleaf.shape[0]), atol=ATOL)
        )
        self.assertTrue(jnp.allclose(rnorm, jnp.eye(rnorm.shape[0]), atol=ATOL))

    def test_left_inner(self):
        """Test left inner contraction/decomposition

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        child, parent = get_left_inner(subkey)

        contracted = contract_left_inner_node(child, parent)

        uchild, uparent = decomp_left_inner_node_up(contracted)
        dchild, dparent = decomp_left_inner_node_down(contracted)

        cu = contract_left_inner_node(uchild, uparent)
        cd = contract_left_inner_node(dchild, dparent)

        lnorm = jnp.einsum("abc,dbc->ad", dparent, dparent)
        unorm = jnp.einsum("abc,adc->bd", uchild, uchild)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(jnp.allclose(lnorm, jnp.eye(lnorm.shape[0]), atol=ATOL))
        self.assertTrue(jnp.allclose(unorm, jnp.eye(unorm.shape[0]), atol=ATOL))

    def test_right_inner(self):
        """Test right inner contraction/decomposition

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        child, parent = get_right_inner(subkey)

        contracted = contract_right_inner_node(child, parent)

        uchild, uparent = decomp_right_inner_node_up(contracted)
        dchild, dparent = decomp_right_inner_node_down(contracted)

        cu = contract_right_inner_node(uchild, uparent)
        cd = contract_right_inner_node(dchild, dparent)

        rnorm = jnp.einsum("abc,abd->cd", dparent, dparent)
        unorm = jnp.einsum("abc,adc->bd", uchild, uchild)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(jnp.allclose(rnorm, jnp.eye(rnorm.shape[0]), atol=ATOL))
        self.assertTrue(jnp.allclose(unorm, jnp.eye(unorm.shape[0]), atol=ATOL))


    def test_left_with_root(self):
        """Test contraction/decomposition of left inner node with root node

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        child, root = get_left_with_root(subkey)

        contracted = contract_left_with_root(child, root)
        uchild, uparent = decomp_left_from_root_up(contracted)
        dchild, dparent = decomp_left_from_root_down(contracted)

        cu = contract_left_with_root(uchild, uparent)
        cd = contract_left_with_root(dchild, dparent)

        unorm = jnp.einsum("abc,adc->bd", uchild, uchild)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(jnp.allclose(dparent @ dparent.T, jnp.eye(dparent.shape[0]), atol=ATOL))
        self.assertTrue(jnp.allclose(unorm, jnp.eye(unorm.shape[0]), atol=ATOL))

    def test_right_with_root(self):
        """Test contraction/decomposition of right inner node with root node

        * Pushing singular values up and down lead to the same contracted tensor
        * Normalization conditions
        """
        self.key, subkey = jax.random.split(self.key)

        child, root = get_right_with_root(subkey)

        contracted = contract_right_with_root(child, root)
        uchild, uparent = decomp_right_from_root_up(contracted)
        dchild, dparent = decomp_right_from_root_down(contracted)

        cu = contract_right_with_root(uchild, uparent)
        cd = contract_right_with_root(dchild, dparent)

        unorm = jnp.einsum("abc,adc->bd", uchild, uchild)

        self.assertTrue(jnp.allclose(cu, cd, atol=ATOL))
        self.assertTrue(jnp.allclose(dparent.T @ dparent, jnp.eye(dparent.shape[1]), atol=ATOL))
        self.assertTrue(jnp.allclose(unorm, jnp.eye(unorm.shape[0]), atol=ATOL))
