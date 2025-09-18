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
        Path(f"{file_path}.{algorithm}").write_text(hash_value.lower(), encoding="utf-8")
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
    gpg_passphrase = os.environ.get("JAVA_PGP_PWD")
    gpg_private_key = os.environ.get("JAVA_PGP_KEY")

    if not gpg_passphrase or not gpg_private_key:
        print(
            "Error: GPG passphrase or private key not found in environment variables ('JAVA_PGP_PWD', 'JAVA_PGP_KEY').",
            file=sys.stderr,
        )
        sys.exit(1)

    gpg_exe_path = get_gpg_path()
    if not gpg_exe_path.is_file():
        print(f"Error: GPG executable not found at '{gpg_exe_path}'.", file=sys.stderr)
        sys.exit(1)

    agent_temp_dir = os.environ.get("AGENT_TEMPDIRECTORY")

    # Use a single temporary directory to manage all temporary files
    with tempfile.TemporaryDirectory(dir=agent_temp_dir) as temp_dir:
        temp_dir_path = Path(temp_dir)
        print(f"Created temporary directory: {temp_dir_path}")

        private_key_file = temp_dir_path / "private.key"
        passphrase_file = temp_dir_path / "passphrase.txt"

        print("Writing GnuPG key and passphrase to temporary files.")
        private_key_file.write_text(gpg_private_key, encoding="utf-8")
        passphrase_file.write_text(gpg_passphrase, encoding="utf-8")

        print("Importing GnuPG private key.")
        run_command([str(gpg_exe_path), "--batch", "--import", str(private_key_file)])
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
                    str(passphrase_file),
                    "--detach-sign",
                    "--armor",
                    str(file_path),
                ]
            )

            # SHA-1 and MD5 Checksums
            create_hash_file(file_path, "sha1")
            create_hash_file(file_path, "md5")

    print("\nFile signing and checksum generation completed.")
    print("Temporary directory and its contents have been deleted.")


if __name__ == "__main__":
    main()
