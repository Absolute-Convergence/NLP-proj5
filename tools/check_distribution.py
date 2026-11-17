"""Check the actual wheel/sdist contents before a candleprep upload."""
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULES = {path.relative_to(ROOT).as_posix() for path in (ROOT / "candleprep").rglob("*.py")}


def main():
    wheels = list((ROOT / "dist").glob("candleprep-*.whl"))
    sources = list((ROOT / "dist").glob("candleprep-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected one candleprep wheel and one source archive in dist/")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        assert EXPECTED_MODULES <= names, "Wheel is missing library modules"
        for name in names:
            assert name in EXPECTED_MODULES or ".dist-info/" in name, name
        assert any(name.endswith("/licenses/LICENSE") for name in names)
    with tarfile.open(sources[0], "r:gz") as archive:
        names = {member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()}
        assert EXPECTED_MODULES <= names, "Source archive is missing library modules"
        for name in names:
            assert (name in EXPECTED_MODULES or name in {
                "LICENSE", "MANIFEST.in", "pyproject.toml", "setup.cfg", "PKG-INFO", "candleprep/README.md"
            } or name.startswith("candleprep.egg-info/")), name
    for path in wheels + sources:
        assert path.stat().st_size < 5 * 1024**2, "Unexpectedly large package"
        print(path.name, path.stat().st_size, "bytes; contents verified")


if __name__ == "__main__":
    main()
