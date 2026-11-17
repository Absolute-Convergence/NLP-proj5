"""Check the actual wheel/sdist contents before a candleprep upload."""
import ast
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULES = {path.relative_to(ROOT).as_posix() for path in (ROOT / "candleprep").rglob("*.py")}


def release_files():
    tree = ast.parse((ROOT / "candleprep/__init__.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets))
    paths = [ROOT / "dist" / f"candleprep-{version}-py3-none-any.whl",
             ROOT / "dist" / f"candleprep-{version}.tar.gz"]
    if not all(path.is_file() for path in paths):
        raise SystemExit(f"Build both distributions for candleprep {version} first")
    return paths


def main():
    wheel, source = release_files()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert EXPECTED_MODULES <= names, "Wheel is missing library modules"
        for name in names:
            assert name in EXPECTED_MODULES or ".dist-info/" in name, name
        assert any(name.endswith("/licenses/LICENSE") for name in names)
    with tarfile.open(source, "r:gz") as archive:
        names = {member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()}
        assert EXPECTED_MODULES <= names, "Source archive is missing library modules"
        for name in names:
            assert (name in EXPECTED_MODULES or name in {
                "LICENSE", "MANIFEST.in", "pyproject.toml", "setup.cfg", "PKG-INFO", "candleprep/README.md"
            } or name.startswith("candleprep.egg-info/")), name
    for path in (wheel, source):
        assert path.stat().st_size < 5 * 1024**2, "Unexpectedly large package"
        print(path.name, path.stat().st_size, "bytes; contents verified")
    return [wheel, source]


if __name__ == "__main__":
    main()
