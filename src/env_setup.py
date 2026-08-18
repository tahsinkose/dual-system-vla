"""Project-local paths for LIBERO and LeRobot.

Both libraries default to storing data in the user's home directory, which this
module redirects into the repo so the project is self-contained and removable.

Call :func:`setup_env` at the top of every entry point, before importing
``lerobot`` or ``libero``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBERO_CONFIG_DIR = REPO_ROOT / "configs" / "libero"
LEROBOT_HOME = REPO_ROOT / "data" / "lerobot"
HF_HOME = REPO_ROOT / "data" / "huggingface"


def _installed_libero_root() -> Path:
    """Locate the installed ``libero.libero`` package directory."""
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "The 'libero' package is not installed. Install it with:\n"
            '    uv pip install "lerobot[libero]"'
        )
    return Path(list(spec.submodule_search_locations)[0]) / "libero"


def ensure_lerobot_home() -> Path:
    """Point LeRobot's dataset cache at ``data/lerobot`` inside the repo.

    Must run before ``lerobot`` is imported — it reads the variable once, at import.
    """
    os.environ["HF_LEROBOT_HOME"] = str(LEROBOT_HOME)
    LEROBOT_HOME.mkdir(parents=True, exist_ok=True)
    return LEROBOT_HOME


def ensure_hf_home() -> Path:
    """Point the HuggingFace cache at ``data/huggingface`` inside the repo.

    ``HF_LEROBOT_HOME`` only covers LeRobot *datasets*. Pretrained model weights are
    fetched by ``transformers``/``huggingface_hub``, which read ``HF_HOME`` instead —
    so without this the ~7.5GB System 2 backbone lands in the user's home directory
    while the dataset sits in the repo, splitting the project across two filesystems.

    Must run before ``transformers`` or ``huggingface_hub`` is imported: both read the
    variable once, at import.
    """
    os.environ["HF_HOME"] = str(HF_HOME)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    return HF_HOME


def link_libero_assets() -> Path | None:
    """Point the installed package's ``assets/`` at LIBERO's asset cache.

    The pip-installed ``libero`` package ships without its mesh assets, so every new
    process prints "Local assets not found. Downloading from HuggingFace Hub..." and
    then immediately "Assets already downloaded at ...". Both lines are noise: the
    first is printed before anything checks the cache, and the resolver only ever
    stats the local filesystem — nothing is fetched.

    Symlinking the cache into the package directory makes the first check succeed, so
    the resolver returns straight away and stays silent. Returns the asset path, or
    None if the cache has not been populated yet (first run downloads it for real).
    """
    cache = Path.home() / ".cache" / "libero" / "assets"
    if not cache.is_dir():
        return None

    link = _installed_libero_root() / "assets"
    if link.is_symlink():
        if link.resolve() == cache.resolve():
            return link
        link.unlink()  # stale link, e.g. from a different cache location
    elif link.exists():
        return link  # a real assets directory ships with this install; leave it alone

    try:
        link.symlink_to(cache, target_is_directory=True)
    except OSError:
        return None  # read-only site-packages; the messages stay, nothing breaks
    return link


def ensure_libero_config(force: bool = False) -> Path:
    """Write the project-local LIBERO config and export ``LIBERO_CONFIG_PATH``.

    Returns the config directory. Safe to call repeatedly.
    """
    os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
    LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    root = _installed_libero_root()
    # The 'datasets' key points at LIBERO's own HDF5 demonstrations, which this
    # project does not use — demonstrations come from the LeRobot dataset instead.
    # LIBERO still requires the key to be present, so it is set to the LeRobot root.
    #
    # Written by hand rather than via yaml.dump to keep this import-light: it must
    # run before libero (and therefore before libero's dependencies) is imported.
    wanted = (
        f"assets: {root / 'assets'}\n"
        f"bddl_files: {root / 'bddl_files'}\n"
        f"benchmark_root: {root}\n"
        f"datasets: {LEROBOT_HOME}\n"
        f"init_states: {root / 'init_files'}\n"
    )

    config_file = LIBERO_CONFIG_DIR / "config.yaml"
    # Rewrite whenever the content differs, not just when the file is absent: these
    # are absolute paths, so a config written by a previous venv (or an older version
    # of this file) is stale and would resolve to directories that no longer exist.
    if force or not config_file.exists() or config_file.read_text() != wanted:
        config_file.write_text(wanted)
    return LIBERO_CONFIG_DIR


def setup_env() -> None:
    """Configure every project-local path. Call before importing lerobot or libero."""
    ensure_lerobot_home()
    ensure_libero_config()
    link_libero_assets()


def report() -> None:
    """Print the resolved paths and whether each exists."""
    import yaml

    setup_env()
    print(f"HF_LEROBOT_HOME={os.environ['HF_LEROBOT_HOME']}")
    print(f"LIBERO_CONFIG_PATH={os.environ['LIBERO_CONFIG_PATH']}\n")

    config = yaml.safe_load((LIBERO_CONFIG_DIR / "config.yaml").read_text())
    for key, value in sorted(config.items()):
        path = Path(value)
        mark = "ok     " if path.exists() else "MISSING"
        suffix = f" -> {path.resolve()}" if path.is_symlink() else ""
        print(f"  [{mark}] {key}: {value}{suffix}")
    if not (_installed_libero_root() / "assets").exists():
        print(
            "\nNote: 'assets' is missing because LIBERO has not downloaded them yet.\n"
            "The first env build fetches them into ~/.cache/libero/assets."
        )


if __name__ == "__main__":
    report()
