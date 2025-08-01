import random
import sys
import unittest
from itertools import product

import jax
import jax.numpy as jnp
from qiskit.quantum_info import random_unitary

from tn.encoding.simple import AngleEncoder, BasisEncoder
from tn.mps import random_mps
from tn.mps.metrics import NLLFunctor, ProbFunctor

N_SITES = 10
BOND_DIM = 20
P_ATOL = 1e-2


class TestMetrics(unittest.TestCase):
    def setUp(self):
        """Create some random MPS to work with and the functors."""
        self.key = jax.random.key(random.randint(0, sys.maxsize))
        self.key, subkey = jax.random.split(self.key)

        mixed_site = random.randint(0, N_SITES - 1)
        self.mps = random_mps(n_sites=N_SITES, bond_dim=BOND_DIM, random_key=subkey)
        self.mps.mixed_canonicalize(mixed_site)

        self.prob = ProbFunctor(self.mps.shapes)
        self.nll = NLLFunctor(self.mps.shapes)

        self.basis_encoder = BasisEncoder()
        self.angle_encoder = AngleEncoder()

    def test_valid_probs(self):
        """Test probability functor onyl returns valid probabilities."""
        self.key, subkey = jax.random.split(self.key)
        data = jax.random.uniform(subkey, shape=(200, N_SITES))
        states = self.angle_encoder(data)

        probs = self.prob(states, self.mps)

        self.assertTrue(jnp.all(probs >= 0.0), msg="Got probabilities < 0")
        self.assertTrue(jnp.all(probs <= 1.0), msg="Got probabilities > 0")

    def test_total_p_basis(self):
        """Total probability over the computational basis states."""
        binary_data = jnp.array(list(product([0, 1], repeat=N_SITES)), dtype=int)
        states = self.basis_encoder(binary_data)

        total_prob = jnp.sum(self.prob(states, self.mps))

        self.assertTrue(
            jnp.isclose(total_prob, 1.0, atol=P_ATOL),
            msg=f"Got total prob. {total_prob}",
        )

    def test_total_p_basis_change(self):
        """Total probability over all basis states in a random basis."""
        binary_data = jnp.array(list(product([0, 1], repeat=N_SITES)), dtype=int)
        states = self.basis_encoder(binary_data)

        unitary = random_unitary(dims=2).data

        evolved_states = jax.vmap(jax.vmap(lambda s: jnp.matmul(unitary, s)))(states)

        total_prob = jnp.sum(self.prob(evolved_states, self.mps))

        self.assertTrue(
            jnp.isclose(total_prob, 1.0, atol=P_ATOL),
            msg=f"Got total prob. {total_prob}",
        )

    def test_nll_functor(self):
        """Note: Not much to test here outside of training. Basically just a sanity
        check, that it works and the returned values make sense."""

        self.key, subkey = jax.random.split(self.key)
        data = jax.random.uniform(subkey, shape=(200, N_SITES))
        states = self.angle_encoder(data)

        loss = self.nll(states, self.mps).mean()

        self.assertTrue(loss > 0.0, msg=f"Got NLL <= 0: {loss}")
