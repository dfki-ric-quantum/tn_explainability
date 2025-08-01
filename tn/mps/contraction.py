from itertools import count
from typing import Callable

import opt_einsum as oe  # type: ignore

from tn.mps import MPSShapes


def get_psi_einsum_str(n_sites: int) -> str:
    """Get Einstein sum string for psi(v).

    Compiles the einsum string for the following contraction:

    o-o-o-o-...-o-o
    | | | |     | |
    o o o o ... o o

    Parameters
    ----------
    n_sites: int
        Number of MPS sites.

    Returns
    -------
    The einsum string
    """
    pidxs = [oe.get_symbol(k) for k in range(n_sites)]
    tidxs = [
        f"{oe.get_symbol(k)}{p}{oe.get_symbol(k+1)}"
        for k, p in zip(range(n_sites, n_sites * 2), pidxs)
    ]

    return ",".join(tidxs + pidxs)


def get_psic_einsum_str(n_sites: int, contracted_site: int) -> str:
    """Get Einstein sum string for psi(v) with two sites contracted.

    Compiles the einsum string for the following contraction:

           cs
    o-o-o-####-o-...-o-o
    | | | |  | |     | |
    o o o o  o o ... o o

    Parameters
    ----------
    n_sites: int
        Number of MPS sites.
    contracted_site: int
        Index of the contracted site.

    Returns
    -------
    The einsum string
    """
    pidxs = [oe.get_symbol(k) for k in range(n_sites)]
    piter = iter(pidxs)
    tidxs = []

    for site_idx in range(n_sites - 1):
        offset = n_sites + site_idx
        lsym = oe.get_symbol(offset)
        rsym = oe.get_symbol(offset + 1)
        if site_idx == contracted_site:
            tidxs.append(f"{lsym}{next(piter)}{next(piter)}{rsym}")
        else:
            tidxs.append(f"{lsym}{next(piter)}{rsym}")

    return ",".join(tidxs + pidxs)


def get_rdm_einsum_str(sub_system: list[int]) -> str:
    """Get the Einstein sum string for the reduced density matrix of a subsystem of n sites.

    Assume the sub-system is [i, j, k] with i<j<k and the MPS mixed canonical at some site in [i,k].
    Then this function compiles the Einstein sum string for the following contraction:

      i         j         k
    +-o-o-...-o-o-o-...-o-o-+
    | | |     | | |     | | |
    |   |     |   |     |   |
    | | |     | | |     | | |
    +-o-o-...-o-o-o-...-o-o-+

    The open physical indices are ordered as such, that the resulting order-L tensor can simply be
    reshaped into a d^(L/2) x d^(L/2) reduced density matrix.

    Parameters
    ----------
    sub_system: list[int]
        Sorted list containing the site indices of the sub-system to consider for the reduced
        density matrix.

    Returns
    -------
    The Einstein sum string for the necessary contraction

    """

    lidx = sub_system[0]
    ridx = sub_system[-1]
    n_contracted_sites = ridx - lidx + 1
    subsys_size = len(sub_system)

    sym_iter = count(start=0, step=1)

    uphys = list(reversed([oe.get_symbol(next(sym_iter)) for _ in sub_system]))
    lphys = list(reversed([oe.get_symbol(next(sym_iter)) for _ in sub_system]))

    ophys = uphys + lphys
    res = "->" + "".join(sorted(ophys))

    cphys = [oe.get_symbol(next(sym_iter)) for _ in range(n_contracted_sites)]

    upper = []
    lower = []

    left_close = oe.get_symbol(next(sym_iter))
    right_close = oe.get_symbol(next(sym_iter))

    for idx in range(n_contracted_sites):
        n = next(sym_iter)
        offset = idx + lidx

        lbond = left_close if offset == lidx else oe.get_symbol(n)
        pbond = ophys[sub_system.index(offset)] if offset in sub_system else cphys[idx]
        rbond = right_close if offset == ridx else oe.get_symbol(n + 1)

        upper.append(f"{lbond}{pbond}{rbond}")

    for idx in range(n_contracted_sites):
        n = next(sym_iter)
        offset = idx + lidx

        lbond = left_close if offset == lidx else oe.get_symbol(n)
        pbond = (
            ophys[sub_system.index(offset) + subsys_size]
            if offset in sub_system
            else cphys[idx]
        )
        rbond = right_close if offset == ridx else oe.get_symbol(n + 1)

        lower.append(f"{lbond}{pbond}{rbond}")

    einsum_str = ",".join(upper + lower)
    einsum_str += res

    return einsum_str


def get_crdm_einsum_str(sub_system: list[int], cond_vals: list[int]) -> str:
    """Get the Einstein sum string for the conditional reduced density matrix of a subsystem
    of n sites conditioned on m sites.

    Assume the sub-system is [i, j, k] with i<j<k, the conditional sites are [l, o, p]
    with l<o<p and the MPS mixed canonical at some site in [min(l,i), max(k,p)]. Then this
    function compiles the Einstein sum string for the following contraction:

      i         l         j         o         k         p
    +-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-+
    | | |     | | |     | | |     | | |     | | |     | | |
    |   |     | x |     |   |     | x |     |   |     | p |
    | | |     | | |     | | |     | | |     | | |     | | |
    +-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-+

    The open physical indices are ordered as such, that the resulting order-L tensor can simply be
    reshaped into a d^(L/2) x d^(L/2) reduced density matrix.

    Parameters
    ----------
    sub_system: list[int]
        Sorted list containing the site indices of the sub-system to consider for the reduced
        density matrix.
    cond_vals: list[int]
        Sorted list containing the site indices of the conditional values to consider for the reduced
        density matrix.

    Returns
    -------
    The Einstein sum string for the necessary contraction
    """

    if len(cond_vals) == 0:
        print("Conditioned sites are zero, returning string of reduced density matrix")
        return get_rdm_einsum_str(sub_system)

    lidx = min((sub_system[0], cond_vals[0]))
    ridx = max((sub_system[-1], cond_vals[-1]))

    n_contracted_sites = ridx - lidx + 1

    subsys_size = len(sub_system)

    sym_iter = count(start=0, step=1)

    usphys = list(reversed([oe.get_symbol(next(sym_iter)) for _ in sub_system]))
    lsphys = list(reversed([oe.get_symbol(next(sym_iter)) for _ in sub_system]))

    ucphys = list([oe.get_symbol(next(sym_iter)) for _ in cond_vals])
    lcphys = list([oe.get_symbol(next(sym_iter)) for _ in cond_vals])

    condphys = []

    for i in range(len(cond_vals)):
        condphys.append(f"{ucphys[i]}{lcphys[i]}")

    ophys = usphys + lsphys
    res = "->" + "".join(sorted(ophys))

    cphys = [oe.get_symbol(next(sym_iter)) for _ in range(n_contracted_sites)]

    upper = []
    lower = []

    left_close = oe.get_symbol(next(sym_iter))
    right_close = oe.get_symbol(next(sym_iter))

    for idx in range(n_contracted_sites):
        n = next(sym_iter)
        offset = idx + lidx

        lbond = left_close if offset == lidx else oe.get_symbol(n)
        pbond = (
            ophys[sub_system.index(offset)]
            if offset in sub_system
            else (
                ucphys[cond_vals.index(offset)] if offset in cond_vals else cphys[idx]
            )
        )
        rbond = right_close if offset == ridx else oe.get_symbol(n + 1)

        upper.append(f"{lbond}{pbond}{rbond}")

    for idx in range(n_contracted_sites):
        n = next(sym_iter)
        offset = idx + lidx

        lbond = left_close if offset == lidx else oe.get_symbol(n)
        pbond = (
            ophys[sub_system.index(offset) + subsys_size]
            if offset in sub_system
            else (
                lcphys[cond_vals.index(offset)] if offset in cond_vals else cphys[idx]
            )
        )
        rbond = right_close if offset == ridx else oe.get_symbol(n + 1)

        lower.append(f"{lbond}{pbond}{rbond}")

    einsum_str = ",".join(upper + lower + condphys)
    einsum_str += res

    return einsum_str


def build_psi_contraction_expr(mps_shapes: MPSShapes, d: int = 2) -> Callable:
    """Pre-compute, optimize and compile the contraction expression for psi(v).

    Pre-computes the expression for:

    o-o-o-o-...-o-o
    | | | |     | |
    o o o o ... o o

    Parameters
    ----------
    mps_shapes: MPSShapes
        The shapes of all tensors in the MPS.
    d: int = 2
        Physical bond dimension

    Returns
    -------
    The pre-computed contraction expression.
    """
    n_sites = len(mps_shapes)
    shapes = mps_shapes + [(d,)] * n_sites

    es_str = get_psi_einsum_str(n_sites)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)


def build_psic_contraction_expr(
    mps_shapes: MPSShapes,
    contracted_shape: tuple[int, ...],
    contracted_site: int,
    d: int = 2,
) -> Callable:
    """Pre-compute, optimize and compile the contraction expression for psi(v) with
    a contracted site.

    Pre-computes the expression for:

           cs
    o-o-o-####-o-...-o-o
    | | | |  | |     | |
    o o o o  o o ... o o

    Parameters
    ----------
    mps_shapes: MPSShapes
        The shapes of all tensors in the MPS.
    contracted_shape: tuple[int, int, int, int]
        Shape of the contracted order-4 tensor.
    contracted_site: int
        Index of the contracted site.
    d: int = 2
        Physical bond dimension

    Returns
    -------
    The pre-computed contraction expression.
    """
    n_sites = len(mps_shapes)

    shapes = (
        mps_shapes[:contracted_site]
        + [contracted_shape]
        + mps_shapes[contracted_site + 2 :]
        + [(d,)] * n_sites
    )

    es_str = get_psic_einsum_str(n_sites, contracted_site)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)


def build_rdm_contraction_expr(
    mps_shapes: MPSShapes, sub_system: list[int]
) -> Callable:
    """Pre-comppute, optimize and compile the contraction expression for the reduced density matrix
    of an MPS sub-system.

    Assume the sub-system is [i, j, k] with i<j<k and the MPS mixed canonical at some site in [i,k].
    Then this function compiles the Einstein sum string for the following contraction:

      i         j         k
    +-o-o-...-o-o-o-...-o-o-+
    | | |     | | |     | | |
    |   |     |   |     |   |
    | | |     | | |     | | |
    +-o-o-...-o-o-o-...-o-o-+

    The open physical indices are ordered as such, that the resulting order-L tensor can simply be
    reshaped into a d^(L/2) x d^(L/2) reduced density matrix.

    Parameters
    ----------
    mps_shapes: MPSShapes
        The shapes of all tensors in the MPS.
    sub_system: list[int]
        Sorted list containing the site indices of the sub-system to consider for the reduced
        density matrix.

    Returns
    -------
    The pre-computed contraction expression
    """
    lidx = sub_system[0]
    ridx = sub_system[-1]

    shapes = mps_shapes[lidx : ridx + 1] + mps_shapes[lidx : ridx + 1]
    es_str = get_rdm_einsum_str(sub_system)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)

def build_crdm_contraction_expr(
    mps_shapes: MPSShapes, sub_system: list[int], cond_vals: list[int], d: int = 2
) -> Callable:
    """Pre-comppute, optimize and compile the contraction expression for the conditional reduced
    density matrix of an MPS sub-system conditioned on some features.

    Assume the sub-system is [i, j, k] with i<j<k, the conditional sites are [l, o, p] with
    l<o<p and the MPS mixed canonical
    at some site in [min(l,i), max(k,p)]. Then this function compiles the Einstein sum string
    for the following contraction:

      i         l         j         o         k         p
    +-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-+
    | | |     | | |     | | |     | | |     | | |     | | |
    |   |     | x |     |   |     | x |     |   |     | x |
    | | |     | | |     | | |     | | |     | | |     | | |
    +-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-o-...-o-o-+

    The open physical indices are ordered as such, that the resulting order-L tensor can simply be
    reshaped into a d^(L/2) x d^(L/2) reduced density matrix.

    Parameters
    ----------
    mps_shapes: MPSShapes
        The shapes of all tensors in the MPS.
    sub_system: list[int]
        Sorted list containing the site indices of the sub-system to consider for the reduced
        density matrix.
    cond_vals: list[int]
        cond_vals: list[int]
        Sorted list containing the site indices of the conditional values to consider for the reduced
        density matrix.


    Returns
    -------
    The pre-computed contraction expression
    """
    lidx = min((sub_system[0], cond_vals[0]))
    ridx = max((sub_system[-1], cond_vals[-1]))

    shapes = mps_shapes[lidx:ridx+1] + mps_shapes[lidx:ridx+1] + [(d,d)]*len(cond_vals)
    es_str = get_crdm_einsum_str(sub_system, cond_vals)

    return oe.contract_expression(es_str, *shapes, memory_limit=-1)
