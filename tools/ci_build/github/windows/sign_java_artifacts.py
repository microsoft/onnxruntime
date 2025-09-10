import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_gpg_path() -> Path:
    """Finds the path to the GPG executable."""
    if platform.system() == "Windows":
        program_files_x86 = os.environ.get("ProgramFiles(x86)")  # noqa: SIM112
        if not program_files_x86:
            raise OSError("ProgramFiles(x86) environment variable not found.")
        return Path(program_files_x86) / "gnupg/bin/gpg.exe"

    gpg_path_str = shutil.which("gpg")
    if gpg_path_str is None:
        raise FileNotFoundError("gpg executable not found in system PATH.")
    return Path(gpg_path_str)


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Executes a command and raises an exception if it fails."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result


def create_hash_file(file_path: Path, algorithm: str) -> None:
    """Creates a checksum file for the given file using the specified algorithm."""
    print(f"  - Generating {algorithm.upper()} checksum...")
    try:
        hasher = hashlib.new(algorithm)
        with file_path.open("rb") as f:
            # Read in chunks to handle large files efficiently
            while chunk := f.read(8192):
                hasher.update(chunk)

        hash_value = hasher.hexdigest()
        # Create checksum file in 'sha1sum'/'md5sum' format.
        # The '*' indicates to read the file in binary mode for verification tools.
        Path(f"{file_path}.{algorithm}").write_text(f"{hash_value.lower()} *{file_path.name}\n", encoding="utf-8")
    except Exception as e:
        print(f"Error generating {algorithm} hash for {file_path}: {e}")
        raise


def main() -> None:
    """
    Signs files with GPG and generates checksums.
    """
    parser = argparse.ArgumentParser(description="Signs files with GPG and generates checksums.")
    parser.add_argument("jar_file_directory", help="The directory containing files to sign.")
    args = parser.parse_args()

    jar_file_directory = Path(args.jar_file_directory)
    if not jar_file_directory.is_dir():
        print(f"Error: Directory not found at '{jar_file_directory}'", file=sys.stderr)
        sys.exit(1)

    print(f"\nListing files to be processed in '{jar_file_directory}':")
    files_to_process = [p for p in jar_file_directory.rglob("*") if p.is_file()]
    for file_path in files_to_process:
        print(f"  - {file_path}")
    print(f"Found {len(files_to_process)} files.")

    print("\nGetting GnuPG signing keys from environment variables.")
    gpg_passphrase = os.environ.get("JAVA_PGP_PWD")  # noqa: SIM112
    gpg_private_key = os.environ.get("JAVA_PGP_KEY")  # noqa: SIM112

    if not gpg_passphrase or not gpg_private_key:
        print(
            "Error: GPG passphrase or private key not found in environment variables ('java-pgp-pwd', 'java-pgp-key').",
            file=sys.stderr,
        )
        sys.exit(1)

    gpg_exe_path = get_gpg_path()
    if not gpg_exe_path.is_file():
        print(f"Error: GPG executable not found at '{gpg_exe_path}'.", file=sys.stderr)
        sys.exit(1)

    with (
        tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".txt", encoding="utf-8") as passphrase_file,
        tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".txt", encoding="utf-8") as private_key_file,
    ):
        print("Writing GnuPG key and passphrase to temporary files.")
        private_key_file.write(gpg_private_key)
        private_key_file.flush()
        passphrase_file.write(gpg_passphrase)
        passphrase_file.flush()

        print("Importing GnuPG private key.")
        run_command([str(gpg_exe_path), "--batch", "--import", private_key_file.name])
        print("Successfully imported GnuPG private key.")

        print(f"\nProcessing {len(files_to_process)} files in '{jar_file_directory}'.")

        for file_path in files_to_process:
            print(f"Processing file: {file_path}")

            # GPG Signing (.asc)
            print("  - GnuPG signing...")
            run_command(
                [
                    str(gpg_exe_path),
                    "--pinentry-mode",
                    "loopback",
                    "--passphrase-file",
                    passphrase_file.name,
                    "--detach-sign",
                    "--armor",
                    str(file_path),
                ]
            )

            # SHA-1 and MD5 Checksums
            create_hash_file(file_path, "sha1")
            create_hash_file(file_path, "md5")

    print("\nFile signing and checksum generation completed.")
    print("Temporary GnuPG key files have been deleted.")


if __name__ == "__main__":
    main()
