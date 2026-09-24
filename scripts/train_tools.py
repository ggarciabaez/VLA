import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm


def _copy_one(DST, path: Path) -> tuple[Path, str | None]:
    """Copy one file src -> dst. Returns (dest_path, error); error is None on success."""
    dest = DST / path.name
    try:
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            return dest, None  # already copied — resume-safe after a disconnect
        shutil.copy2(path, dest)
        return dest, None
    except Exception as e:
        return dest, str(e)


def _verify_one(path: Path) -> str | None:
    """
    Cheap integrity check run right after a copy lands — catches a truncated
    or corrupt file immediately instead of failing deep inside dataset
    loading later. Only opens the .npz central directory, doesn't read arrays.
    """
    if path.suffix != ".npz":
        return None
    try:
        with np.load(path) as f:
            _ = f.files
        return None
    except Exception as e:
        return str(e)


def download_shards(src: Path, dst: Path, max_workers: int = 16) -> list[Path]:
    """Parallel copy + verify of every file in `src` into `dst`."""
    files = sorted(p for p in src.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {src}")

    results: list[Path] = []
    errors: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_copy_one, dst, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading shards"):
            dest, err = fut.result()
            if err:
                errors.append((dest, err))
                continue
            # "as they come in, we read them" — verify as soon as each file
            # lands rather than waiting for the whole batch to finish first.
            verify_err = _verify_one(dest)
            if verify_err:
                errors.append((dest, verify_err))
                continue
            results.append(dest)

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for path, err in errors:
            print(f"  {path.name}: {err}")
        raise RuntimeError(f"{len(errors)} shard(s) failed to download/verify — see above")

    print(f"Downloaded and verified {len(results)} files -> {dst}")
    return results