import os
import argparse
import requests
import zipfile

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Morpho-MNIST downloader',
        description='Download the morpho-mnist dataset from https://github.com/dccastro/Morpho-MNIST.',
    )
    parser.add_argument('-d', '--directory', type=str, help='path to dataset directory')
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()

    for file_id, name in zip(
        ["1fFGJW0IHoBmLuD6CEKCB8jz3Y5LJ5Duk", "1q3Bfl1oraKZcIPLHnqkU0whnTiz-AVSP", "1Uy-SmnEkwq_dptTFuoUtmO9rn2FAbNb8"], 
        ["global", "thin", "thick"],
    ):
        if args.verbose:
            print(f"Load {name} with file-id {file_id}.")
        zip_path = os.path.join(args.directory, f"{name}.zip")
        download_file_from_google_drive(file_id, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(args.directory))
        os.remove(zip_path)

