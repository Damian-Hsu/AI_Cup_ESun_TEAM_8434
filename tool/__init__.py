from ._dataset_store import DatasetStore 
from .config import Config
import gdown

__all__ = ["Config", "download_dataset"]

def download_dataset():
    """
    Download the competition dataset from a public Google Drive folder into
    the local project directory using `gdown`.

    This function retrieves all files contained in the specified Google Drive
    folder link and stores them under the local `./.data` directory. It is
    commonly used to automate dataset preparation in the project setup stage.

    The function prints a completion message once all files have been
    successfully downloaded.

    Returns
    -------
    None
        This function performs I/O operations only and does not return a value.
    """

    folder_url = "https://drive.google.com/drive/folders/1QHp8hYPKWiBmGfOFLQrHMttbpB9rh6GZ?usp=sharing"
    gdown.download_folder(folder_url, quiet=False, use_cookies=False, output= "./.data")
    print("Dataset downloaded successfully.")