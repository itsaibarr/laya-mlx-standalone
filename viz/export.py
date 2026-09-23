"""Copy capture JSONL into the static web bundle."""
import shutil


def copy_runs(src: str, dst: str) -> str:
    """Copy a JSONL capture file into the web bundle runs/ directory.

    Args:
        src: Source JSONL file path.
        dst: Destination path in viz-web/runs/.

    Returns:
        str: Destination path.
    """
    shutil.copyfile(src, dst)
    return dst
