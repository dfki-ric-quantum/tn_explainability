import pickle
from enum import Enum
from functools import reduce
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from tn.mps.ops import trunc_svd_normalize_left, trunc_svd_normalize_right

MPSShapes = list[tuple[int, int, int]]


class MPSNormState(Enum):
    """Normalization state of a tensor within the MPS."""

    ANY = 1
    LN = 2
    RN = 3
    MX = 4


def load_mps(path: str) -> "MPS":
    """Loads an MPS from a file.

    Parameters
    ----------
    path: str
        Full path to the MPS file

    Returns
    -------
    The MPS
    """
    with open(path, "rb") as infile:
        data = pickle.load(infile)
        norm_states = data.get("norm_states", None)

        return MPS(len(data["tensors"]), data["tensors"], norm_states)

    raise RuntimeError(f"Could not load {path}")


def save_mps(mps: "MPS", path: str) -> None:
    """Save MPS to a file.

    Parameters
    ----------
    mps: MPS
        The MPS to save
    path: str
        Path to save to. Note: All sub-folders need to exits.
    """
    with open(path, "wb") as outfile:
        pickle.dump({"tensors": mps.tensors, "norm_states": mps.norm_states}, outfile)
        return

    raise RuntimeError(f"Could not save to {path}")


def random_mps(
    n_sites: int, bond_dim: int | list[int], random_key: jax.Array, d: int = 2
) -> "MPS":
    """Generate a random Matrix Product State.

    Parameters
    ----------
    n_sites: int
        Number of sites
    bond_dim: int | list[int]
        Either the common bond dimension for all sites, or a list of length n_sites
        with all the bond dimensions.
    random_key: jax.Array
        Random key to sample with
    d: int = 2
        Physical bond dimension of the MPS.

    Returns
    -------
    The generated MPS
    """
    if isinstance(bond_dim, list):
        assert (
            len(bond_dim) == n_sites - 1
        ), f"Got {len(bond_dim)} bond dims, for {n_sites} sites"

        bdim = bond_dim
    else:
        bdim = [bond_dim] * (n_sites - 1) + [1]

    _, *subkeys = jax.random.split(random_key, n_sites + 1)

    tensors = [
        jax.random.uniform(subkey, (bdim[k - 1], d, bdim[k]), minval=-1.0, maxval=1.0)
        for k, subkey in enumerate(subkeys)
    ]

    norm_states = np.full((n_sites,), MPSNormState.ANY)

    return MPS(n_sites=n_sites, tensors=tensors, norm_states=norm_states)


def mps_from_statevector(state: jax.typing.ArrayLike) -> "MPS":
    """Decompose state vector into an untruncated matrix product state.

    The algorithm implements the SVD based decomposition procedure described in [1]. If the
    statevector describes a system on L qubits, the largest matrices in the MPS will have dimension
    2^(L/2) x 2^(L/2).

    [1] Schollwöck, "The density-matrix renormalization group in the age of matrix product
    states" (2011)

    Parameters
    ----------
    state: jax.typing.ArrayLike
        The statevector to decompose

    Returns
    -------
    The full MPS representation of the state, normalized and left canonicalized.
    """

    n_coeffs = state.shape[0]
    assert (
        n_coeffs > 0 and n_coeffs & (n_coeffs - 1) == 0
    ), "Number of coefficients must be a power of 2"

    n_sites = int(jnp.log2(n_coeffs))
    tensors = []
    lbond = 1
    rbond = None

    psi = state.reshape((2, -1))

    for _ in range(n_sites - 1):
        u, s, v = jnp.linalg.svd(psi, full_matrices=False)
        rank = s.shape[0]
        rbond = rank

        tensors.append(u[:, :rank].reshape((lbond, 2, rbond)))
        psi = jnp.diag(s) @ v[:rank, :]
        psi /= jnp.linalg.norm(psi)
        psi = psi.reshape((rank * 2, -1))

        lbond = rbond

    tensors.append(psi.reshape(lbond, 2, -1))
    norm_states = np.full((n_sites,), MPSNormState.LN)
    norm_states[-1] = MPSNormState.MX

    return MPS(n_sites, tensors, norm_states)


class TwoSiteForwardIterator:
    """Forward iterate MPS, two neighbouring sites at a time."""

    def __init__(
        self, mps: "MPS", *, start: int = 0, stop: Optional[int] = None
    ) -> None:
        """The constructor.

        Parameters
        ----------
        mps: "MPS"
            The MPS to iterate.
        start: int = 0
            Start site.
        stop: Optional[int] = None
            Stop site.
        """
        self.mps = mps
        self.left_index = start

        if stop is not None:
            self.stop = stop
        else:
            self.stop = mps.n_sites - 1

    def __iter__(self) -> "TwoSiteForwardIterator":
        """Return the iterator."""
        return self

    def __next__(self) -> tuple[int, int, jax.Array, jax.Array]:
        """Iterate to next site.

        Returns
        -------
        Indices of the left and right site, left and right site tensor
        """
        if self.left_index == self.stop:
            raise StopIteration

        lidx = self.left_index
        ridx = lidx + 1

        self.left_index += 1

        return lidx, ridx, *self.mps[lidx : ridx + 1]


class TwoSiteReverseIterator:
    """Reverse iterate MPS, two neighbouring sites at a time."""

    def __init__(
        self, mps: "MPS", *, start: Optional[int] = None, stop: Optional[int] = None
    ) -> None:
        """The constructor.

        Parameters
        ----------
        mps: "MPS"
            The MPS to iterate.
        start: int = 0
            Start site.
        stop: Optional[int] = None
            Stop site.
        """
        self.mps = mps

        if start is None:
            self.left_index = self.mps.n_sites - 2
        else:
            self.left_index = start

        if stop is None:
            self.stop = 0
        else:
            self.stop = stop

    def __iter__(self) -> "TwoSiteReverseIterator":
        """Return the iterator."""
        return self

    def __next__(self) -> tuple[int, int, jax.Array, jax.Array]:
        """Iterate to next site.

        Returns
        -------
        Indices of the left and right site, left and right site tensor
        """
        if self.left_index < self.stop:
            raise StopIteration

        lidx = self.left_index
        ridx = lidx + 1

        self.left_index -= 1

        return lidx, ridx, *self.mps[lidx : ridx + 1]


class MPS:
    """Simple container for a Matrix Product State."""

    def __init__(
        self,
        n_sites: int,
        tensors: list[jax.Array],
        norm_states: Optional[list[MPSNormState]] = None,
    ) -> None:
        """The constructor

        Parameters
        ----------
        n_sites: int,
            number of sites/cores/... in the MPS
        tensors: list[jax.Array],
            List of Tensors for the MPS
        norm_states: Optional[list[MPSNormState]] = None,
            Normalization state of the tensors. If None is provided, no particular state is assumed.
        """
        self.n_sites = n_sites
        self.tensors = tensors
        self.d = tensors[0].shape[1]

        if norm_states is not None:
            assert (
                len(norm_states) == n_sites
            ), f"Got {len(norm_states)} norm. states for {n_sites} sites"
            self.norm_states = norm_states
        else:
            self.norm_states = np.full((n_sites,), MPSNormState.ANY)

    def __getitem__(self, key: int | slice) -> jax.Array | list[jax.Array]:
        """Get a site/several sites with an index/slice"""
        return self.tensors[key]

    def __iter__(self):
        """Return an iterator to the MPS' sites."""
        return iter(self.tensors)

    def __len__(self):
        """Length of the MPS (number of sites)."""
        return self.n_sites

    @property
    def average_bond_dim(self) -> float:
        """Average bond dimension of the MPS."""
        return reduce(lambda v, t: v + t.shape[2], self[:-1], 0) / (self.n_sites - 1)

    @property
    def shapes(self) -> MPSShapes:
        """Shapes of all the site tensors."""
        return [t.shape for t in self]

    def update_ln(self, lidx: int, ridx: int, lhs: jax.Array, rhs: jax.Array) -> None:
        """Update pair of tensors left-normalized.

        This method is called to update the MPS after a contracted order-4 tensors has been
        decomposed into

        U, SVh

        The left and right tensor given by the index are updated accordingly. Note: The method does
        not perform any checks on the validity of the passed arguments for performance reasons. It's
        within the callers responsibility to ensure passed data is valid.

        Parameters
        ----------
        lidx: int
            Index of the left tensor.
        ridx: int
            Index of the right tensor (lidx + 1)
        lhs:
            The left-normalized tensor to be placed at lidx.
        rhs:
            The tensor to be placed on ridx.
        """
        self.tensors[lidx], self.tensors[ridx] = lhs, rhs
        self.norm_states[lidx] = MPSNormState.LN
        self.norm_states[ridx] = MPSNormState.MX

    def update_rn(self, lidx: int, ridx: int, lhs: jax.Array, rhs: jax.Array) -> None:
        """Update pair of tensors right-normalized.

        This method is called to update the MPS after a contracted order-4 tensors has been
        decomposed into

        US, Vh

        The left and right tensor given by the index are updated accordingly. Note: The method does
        not perform any checks on the validity of the passed arguments for performance reasons. It's
        within the callers responsibility to ensure passed data is valid.

        Parameters
        ----------
        lidx: int
            Index of the left tensor.
        ridx: int
            Index of the right tensor (lidx + 1)
        lhs:
            The tensor to be placed at lidx.
        rhs:
            The right-normalized tensor to be placed on ridx.
        """
        self.tensors[lidx], self.tensors[ridx] = lhs, rhs
        self.norm_states[lidx] = MPSNormState.MX
        self.norm_states[ridx] = MPSNormState.RN

    def left_canonicalize(self) -> None:
        """Left canonicalize the MPS into the form:

        o->o->o->o->...->o
        """
        if np.all(self.norm_states[:-1] == MPSNormState.LN):
            return

        for lidx, ridx, lhs, rhs in self.two_site_forward():
            if self.norm_states[lidx] == MPSNormState.LN:
                continue

            bond_dim = lhs.shape[2]

            self.update_ln(
                lidx,
                ridx,
                *trunc_svd_normalize_left(
                    lhs, rhs, max_bond=bond_dim, sigma_thresh=0.0
                ),
            )

    def right_canonicalize(self) -> None:
        """Right canonicalize the MPS into the form:

        o<-o<-o<-o<-...<-o
        """
        if np.all(self.norm_states[1:] == MPSNormState.RN):
            return

        for lidx, ridx, lhs, rhs in self.two_site_reverse():
            if self.norm_states[ridx] == MPSNormState.RN:
                continue

            bond_dim = lhs.shape[2]

            self.update_rn(
                lidx,
                ridx,
                *trunc_svd_normalize_right(
                    lhs, rhs, max_bond=bond_dim, sigma_thresh=0.0
                ),
            )

    def mixed_canonicalize(self, site: int) -> None:
        """Transform MPS into mixed canonical form:

                   s
        o->...->o->o<-o<-...<-o

        Parameters
        ----------
        site: int
            The site to canonicalize to.
        """

        left_cond = np.all(self.norm_states[:site] == MPSNormState.LN)
        right_cond = np.all(self.norm_states[site + 1 :] == MPSNormState.RN)

        if left_cond and right_cond:
            return

        for lidx, ridx, lhs, rhs in self.two_site_forward(stop=site):
            if self.norm_states[lidx] == MPSNormState.LN:
                continue

            bond_dim = lhs.shape[2]

            self.update_ln(
                lidx,
                ridx,
                *trunc_svd_normalize_left(
                    lhs, rhs, max_bond=bond_dim, sigma_thresh=0.0
                ),
            )

        for lidx, ridx, lhs, rhs in self.two_site_reverse(stop=site):
            if self.norm_states[ridx] == MPSNormState.RN:
                continue

            bond_dim = lhs.shape[2]

            self.update_rn(
                lidx,
                ridx,
                *trunc_svd_normalize_right(
                    lhs, rhs, max_bond=bond_dim, sigma_thresh=0.0
                ),
            )

    def two_site_forward(
        self, *, start: int = 0, stop: Optional[int] = None
    ) -> TwoSiteForwardIterator:
        """Crate an two site forward iterator over the MPS.

        Parameters
        ----------
        start: int = 0
            Site to start at
        stop: Optional[int] = None
            Site to end at

        Returns
        -------
        The iterator
        """
        return TwoSiteForwardIterator(self, start=start, stop=stop)

    def two_site_reverse(
        self, *, start: Optional[int] = None, stop: Optional[int] = None
    ) -> TwoSiteReverseIterator:
        """Crate an two site revers iterator over the MPS.

        Parameters
        ----------
        start: Optinal[int] = None
            Site to start at
        stop: Optional[int] = None
            Site to end at

        Returns
        -------
        The iterator
        """
        return TwoSiteReverseIterator(self, start=start, stop=stop)
