import glob
import os
import shutil
import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path


def untar_archives(input_path: Path, output_path: Path):
    archives = glob.glob(str(input_path / '*.tar.gz'))
    for archive in archives:
        tar = tarfile.open(input_path / archive, "r:gz")
        tar.extractall(output_path)
        tar.close()


parser = ArgumentParser(
    description="Untar archive files from input path to output path")
parser.add_argument("--input_path",
                    type=str,
                    required=True,
                    help="Input path to search for archives")
parser.add_argument(
    "--output_path",
    type=str,
    required=False,
    default=None,
    help="output path to save extracted archives, default to input path")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path
    untar_archives(input_path, output_path)
