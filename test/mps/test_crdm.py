import random
import sys
import unittest
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from qiskit.quantum_info import Operator, Statevector, partial_trace, random_statevector

from tn.encoding.simple import AngleEncoder
from tn.mps import mps_from_statevector, random_mps
from tn.mps.rdm import conditional_reduced_density_matrix

N_QUBITS = 10
ATOL = 1e-3

projectors = [jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [0, 1]])]


class TestCRDM(unittest.TestCase):
    def setUp(self):
        self.encoder = AngleEncoder()

    def test_one_site_crdm(self):
        """Test crdm computation vs partial trace on a quantum state for a single site
        and single conditioned site."""
        n_dim = 2**N_QUBITS
        statevector = random_statevector(n_dim)

        rand_sites = random.sample(range(N_QUBITS - 1), k=2)
        sub_sys = rand_sites[0]
        cond_vals = rand_sites[1]

        single_qubit_operators = [
            jnp.eye(2) if i != cond_vals else projectors[0]
            for i in reversed(range(N_QUBITS))
        ]
        op_matrix = reduce(jnp.kron, single_qubit_operators)
        operator = Operator(np.array(op_matrix))

        projected_statevector = statevector.evolve(operator)

        cond_statevector = Statevector(
            np.array(
                projected_statevector.data / jnp.linalg.norm(projected_statevector.data)
            )
        )

        rho = partial_trace(
            cond_statevector, [i for i in range(N_QUBITS) if i != sub_sys]
        ).data

        state = statevector.reverse_qargs().data

        input_data = jnp.array(
            [random.uniform(0, 1) if i != cond_vals else 0 for i in range(N_QUBITS)]
        ).reshape(1, -1)
        input_array = self.encoder(input_data)

        mps = mps_from_statevector(state)
        crdm = conditional_reduced_density_matrix(
            mps, [sub_sys], [cond_vals], input_array
        )

        self.assertTrue(jnp.allclose(crdm, rho, atol=ATOL))

    def test_multi_site_crdm(self):
        """Test crdm computation vs partial trace on a quantum state for a multiple sites and
        multiple conditioned sites."""
        n_dim = 2**N_QUBITS
        statevector = random_statevector(n_dim)

        rand_sites = np.array(random.sample(range(N_QUBITS), k=5))
        split = random.randint(1, 4)

        sub_sys = list(rand_sites[:split])
        cond_vals = list(rand_sites[split:])
        sub_sys.sort()
        cond_vals.sort()

        states = np.random.choice([0, 1], size=N_QUBITS)

        single_qubit_operators = [
            jnp.eye(2) if i not in cond_vals else projectors[states[i]]
            for i in reversed(range(N_QUBITS))
        ]
        op_matrix = reduce(jnp.kron, single_qubit_operators)
        operator = Operator(np.array(op_matrix))

        projected_statevector = statevector.evolve(operator)

        cond_statevector = Statevector(
            np.array(
                projected_statevector.data / jnp.linalg.norm(projected_statevector.data)
            )
        )

        rho = partial_trace(
            cond_statevector, [i for i in range(N_QUBITS) if i not in sub_sys]
        ).data

        state = statevector.reverse_qargs().data

        input_data = jnp.array(
            [
                random.uniform(0, 1) if i not in cond_vals else states[i]
                for i in range(N_QUBITS)
            ]
        ).reshape(1, -1)
        input_array = self.encoder(input_data)

        mps = mps_from_statevector(state)
        crdm = conditional_reduced_density_matrix(mps, sub_sys, cond_vals, input_array)

        self.assertTrue(jnp.allclose(crdm, rho, atol=ATOL))

    def test_crdm_from_random_mps(self):
        """Test rdm computation from random MPS returns valid density matrix."""

        key = jax.random.key(random.randint(0, sys.maxsize))
        mps = random_mps(N_QUBITS, bond_dim=10, random_key=key)

        rand_sites = np.array(random.sample(range(N_QUBITS), k=5))
        split = random.randint(1, 4)

        sub_sys = list(rand_sites[:split])
        cond_vals = list(rand_sites[split:])
        sub_sys.sort()
        cond_vals.sort()

        states = np.random.choice([0, 1], size=N_QUBITS)
        input_data = jnp.array(
            [
                random.uniform(0, 1) if i not in cond_vals else states[i]
                for i in range(N_QUBITS)
            ]
        ).reshape(1, -1)
        input_array = self.encoder(input_data)

        crdm = conditional_reduced_density_matrix(mps, sub_sys, cond_vals, input_array)

        trace = jnp.trace(crdm)
        herm_norm = jnp.linalg.norm(crdm - jnp.conjugate(crdm.T))
        eigen_vals, _ = jnp.linalg.eigh(crdm)

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
