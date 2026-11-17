"""Download the original public CSV once, with a pinned revision and checksum."""
import hashlib
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCE_URL = (
    "https://raw.githubusercontent.com/Absolute-Convergence/NLP-proj5/"
    "94b97c68550b11aa6eeb1b0d948cc6cda1ac1a3a/TVC_GOLD%2C%20360.csv"
)
SOURCE_SHA256 = "aa89b7a05abad8f3ee718c65607169a547ec215370cde15cfe98f8c3a895633c"


def download_source(destination, url=SOURCE_URL, sha256=SOURCE_SHA256):
    """Reuse a verified cache or atomically save a verified HTTPS download."""
    destination = Path(destination)
    if not url.startswith("https://"):
        raise ValueError("The source URL must use HTTPS")
    if destination.exists():
        content = destination.read_bytes()
    else:
        try:
            with urlopen(Request(url, headers={"User-Agent": "candleprep"}), timeout=60) as response:
                content = response.read(16 * 1024**2 + 1)
        except (URLError, TimeoutError) as error:
            raise RuntimeError("Could not download the source CSV; check the internet connection and retry") from error
    if len(content) > 16 * 1024**2:
        raise ValueError("The source CSV exceeds the 16 MiB download limit")
    if sha256 and hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Source CSV checksum mismatch; the existing file was not overwritten")
    # Check CSV/schema/domain constraints before caching any response (including HTML).
    from .datasets.generator import audit_source
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".csv", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            audit_source(temporary)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination
