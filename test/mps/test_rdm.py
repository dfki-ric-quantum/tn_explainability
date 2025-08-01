import random
import sys
import unittest

import jax
import jax.numpy as jnp
from qiskit.quantum_info import partial_trace, random_statevector

from tn.mps import mps_from_statevector, random_mps
from tn.mps.rdm import reduced_density_matrix

N_QUBITS = 10
ATOL = 1e-3


class TestRDM(unittest.TestCase):
    def test_one_site_rdm(self):
        """Test rdm computation vs partial trace on a quantum state for a single site."""
        n_dim = 2**N_QUBITS
        statevector = random_statevector(n_dim)

        sub_sys = random.randint(0, N_QUBITS - 1)

        rho = partial_trace(
            statevector, [i for i in range(N_QUBITS) if i != sub_sys]
        ).data
        state = statevector.reverse_qargs().data

        mps = mps_from_statevector(state)
        rdm = reduced_density_matrix(mps, [sub_sys])

        self.assertTrue(jnp.allclose(rdm, rho, atol=ATOL))

    def test_multi_site_rdm(self):
        """Test rdm computation vs partial trace on a quantum state for a multiple sites."""
        n_dim = 2**N_QUBITS
        statevector = random_statevector(n_dim)

        sub_sys = random.sample(range(N_QUBITS), k=3)
        sub_sys.sort()

        rho = partial_trace(
            statevector, [i for i in range(N_QUBITS) if i not in sub_sys]
        ).data
        state = statevector.reverse_qargs().data

        mps = mps_from_statevector(state)
        rdm = reduced_density_matrix(mps, sub_sys)

        self.assertTrue(jnp.allclose(rdm, rho, atol=ATOL))

    def test_rdm_from_random_mps(self):
        """Test rdm computation from random MPS returns valid density matrix."""

        key = jax.random.key(random.randint(0, sys.maxsize))
        mps = random_mps(N_QUBITS, bond_dim=10, random_key=key)

        sub_sys = random.sample(range(N_QUBITS), k=3)
        sub_sys.sort()

        rdm = reduced_density_matrix(mps, sub_sys)

        trace = jnp.trace(rdm)
        herm_norm = jnp.linalg.norm(rdm - jnp.conjugate(rdm.T))
        eigen_vals, _ = jnp.linalg.eigh(rdm)

        # We squash very small eigenvalues around 0 to 0, to account for numerical errors
        eigen_vals = jnp.where(jnp.abs(eigen_vals) < 1e-3, 0.0, eigen_vals)

        # Note: the trace is notoriously inaccurate here, hence the larger atol
        self.assertTrue(
            jnp.isclose(trace, 1.0, atol=1e-2), msg=f"No unit trace: {trace}"
        )
        self.assertTrue(
            jnp.isclose(herm_norm, 0.0, atol=ATOL), msg=f"Not hermitian: {herm_norm}"
        )
        self.assertTrue(jnp.all(eigen_vals >= 0), msg="Not PSD")
