import os
import pickle
import sys

from absl import logging
from ucimlrepo import dotdict, fetch_ucirepo


def _undotdict(data: dotdict) -> dict:
    """Remove dotdict wrapper to make object pickable.

    Parameters
    ----------
    data: dotdict
        The dotdict as provided by ucimlrepo.fetch_ucirepo

    Returns
    -------
    Unpacked plain dict with the same fields
    """
    metadata = dict(data.metadata)
    metadata["additional_info"] = (
        dict(metadata["additional_info"]) if metadata["additional_info"] else None
    )
    metadata["intro_paper"] = (
        dict(metadata["intro_paper"]) if metadata["intro_paper"] else None
    )

    return {"data": dict(data.data), "metadata": metadata, "variables": data.variables}


def _dotdict(data: dict) -> dotdict:
    """Wrap plain dict loaded from a cached file in a ucimlrepo.dotdict.

    Parameters
    ----------
    data: dict
        The plain dict loaded from the file

    Returns
    -------
    The dotdict object, as it would have been provided by ucimlrepo.fetch_ucirepo

    """
    metadata = dotdict(data["metadata"])

    metadata.additional_info = (
        dotdict(metadata.additional_info) if metadata.additional_info else None
    )
    metadata.intro_paper = (
        dotdict(metadata.intro_paper) if metadata.intro_paper else None
    )

    return dotdict({
        "data": dotdict(data["data"]),
        "metadata": metadata,
        "variables": data["variables"],
    })


def _linux_path(repo_id: int, create: bool = True) -> str:
    """Get the cache file path for a repo on linux

    Parameters
    ----------
    repo_id: int
        Id of the repo
    create: bool = True
        Create all subdirectories, if they don't exits.

    Returns
    -------
    The entire path to the file.
    """
    cache_dir = os.environ.get("XDG_CACHE_HOME", "")

    if not cache_dir.strip():
        cache_dir = os.path.expanduser("~/.cache")

    cache_dir += "/dfki_exptn/"

    if create:
        os.makedirs(cache_dir, exist_ok=True)

    return cache_dir + f"uciml_{repo_id}.pickle"


def _load_cached(repo_id: int) -> tuple[dict | dotdict, str]:
    """Load cached repo file, if it exists.

    Note: Only implemented for Linux yet.

    Parameters
    ----------
    repo_id: int
        Id of the repo

    Returns
    -------
    The dotdict object, as it would have been provided by ucimlrepo.fetch_ucirepo, if
    the data was loaded from cache, else and empty dict. Additionally the path to the
    file is returned.
    """
    data = {}
    cache_fname = ""

    if sys.platform == "linux":
        cache_fname = _linux_path(repo_id=repo_id)

        if os.path.isfile(cache_fname):
            logging.info("Loading cached uciml repo file: %s", cache_fname)

            with open(cache_fname, "rb") as infile:
                data = _dotdict(pickle.load(infile))

    return data, cache_fname


def _save_cache(data: dotdict, fname: str) -> None:
    """Save dataset to cache.

    Note: Only implemented for Linux yet.

    Parameters
    ----------
    data: dotdict
        The data as fetched from the UCIML Repo to cache.
    fname: str
        Full path to the cache file to be writte.
    """
    if sys.platform == "linux":
        logging.info("Caching repo to %s", fname)

        with open(fname, "wb") as outfile:
            pickle.dump(_undotdict(data), outfile)


def fetch(repo_id: int) -> dotdict:
    """Fetch UCIML Repo or load it from cache. After loading the repo is cached for later use.

    Note: Caching is only implemented for Linux yet.

    Parameters
    ----------
    repo_id: int
        The repository id.

    Returns
    -------
    The dataset from the repository.
    """
    data, cache_fname = _load_cached(repo_id=repo_id)

    if not data:
        logging.info("Fetching uciml repo %d...", repo_id)
        data = fetch_ucirepo(id=repo_id)

        _save_cache(data, cache_fname)

    return data
