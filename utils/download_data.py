import pyzipper
import gdown
import os
from typing import Optional

def compress_file_with_password(file_path: str, output_path: str, password: str) -> None:
    """
    Compress a file with a password using AES encryption.

    Args:
        file_path (str): Path to the file to be compressed.
        output_path (str): Path to save the compressed file.
        password (str): Password for encryption.
    """
    with pyzipper.AESZipFile(output_path, 'w', compression=pyzipper.ZIP_DEFLATED) as zf:
        zf.setpassword(password.encode())
        zf.setencryption(pyzipper.WZ_AES, nbits=256)
        zf.write(file_path, os.path.basename(file_path))

def download_file_from_drive(file_id: str, output: str) -> None:
    """
    Download a file from Google Drive.

    Args:
        file_id (str): Google Drive file ID.
        output (str): Path to save the downloaded file.
    """
    url: str = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

def decompress_file(zip_file: str, output_folder: str, password: Optional[str] = None) -> None:
    """
    Decompress a password-protected zip file.

    Args:
        zip_file (str): Path to the zip file.
        output_folder (str): Folder to extract the contents.
        password (Optional[str], optional): Password for decryption. Defaults to None.
    """
    try:
        with pyzipper.AESZipFile(zip_file, 'r') as zf:
            if password:
                zf.setpassword(password.encode())
            zf.extractall(path=output_folder)
        print(f"Decompression successful. Deleting the zip file: {zip_file}")
    except Exception as e:
        print(f"An error occurred during decompression: {e}")
    finally:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"Compressed file {zip_file} has been deleted.")