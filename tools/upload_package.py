"""Upload the checked release using an explicitly entered, in-memory token."""
import getpass

from twine.commands.check import check
from twine.commands.upload import upload
from twine.settings import Settings

from check_distribution import main as check_contents

def main():
    artifacts = [str(path) for path in check_contents()]
    if check(artifacts, strict=True):
        raise SystemExit("Distribution validation failed; nothing was uploaded.")
    print("Uploading to production PyPI using the token entered below.")
    print("Use a production token authorized for the candleprep project.")
    token = getpass.getpass("PyPI API token (hidden): ").strip()
    if not token:
        raise SystemExit("No token entered; nothing was uploaded.")
    settings = Settings(
        username="__token__", password=token, non_interactive=True,
        repository_url="https://upload.pypi.org/legacy/",
    )
    # Explicit credentials take precedence over .pypirc and the keyring.
    upload(settings, artifacts)


if __name__ == "__main__":
    main()
