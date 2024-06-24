import pyzipper
import gdown
import os

def compress_file_with_password(file_path, output_path, password):
    with pyzipper.AESZipFile(output_path, 'w', compression=pyzipper.ZIP_DEFLATED) as zf:
        zf.setpassword(password.encode())
        zf.setencryption(pyzipper.WZ_AES, nbits=256)
        zf.write(file_path, os.path.basename(file_path))

def download_file_from_drive(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

def decompress_file(zip_file, output_folder, password):
    try:
        with pyzipper.AESZipFile(zip_file, 'r') as zf:
            zf.setpassword(password.encode())
            zf.extractall(path=output_folder)
        print(f"Decompression successful. Deleting the zip file: {zip_file}")
    except Exception as e:
        print(f"An error occurred during decompression: {e}")
    finally:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"Compressed file {zip_file} has been deleted.")
