import os
from typing import List, Optional, Dict, Literal
import pandas as pd
from pathlib import Path
import json
from tool._dataset_store import DatasetStore
class Config:
    """
    Global configuration settings for data management and project structure.

    This class provides:
    - A shared base directory for storing datasets.
    - A nested `Data` helper class that defines dataset paths and quick-access
      functions for loading, saving, listing, and updating versioned datasets
      managed through the DatasetStore.
    """

    # Base data directory
    data_base = "./.data"

    class Data:
        """
        Data configuration and dataset version management helper.

        This class defines path utilities for locating raw datasets (CSV files),
        and exposes convenience wrappers around DatasetStore for managing
        versioned datasets—including saving, loading, listing versions, updating,
        and pruning old versions.
        """

        # Directories and file names
        dataset_dir = "dataset"
        acct_alert = "acct_alert.csv"
        acct_predict = "acct_predict.csv"
        acct_transaction = "acct_transaction.csv"

        @property
        def dataset_dir_path(self):
            """
            Return the absolute path to the dataset directory.

            Returns
            -------
            str
                Path to `<data_base>/dataset`.
            """
            return os.path.join(Config.data_base, self.dataset_dir)

        @property
        def acct_alert_path(self):
            """
            Return the full file path of `acct_alert.csv`.

            Returns
            -------
            str
                Full path to the alert-account CSV file.
            """
            return os.path.join(Config.data_base, self.dataset_dir, self.acct_alert)

        @property
        def acct_predict_path(self):
            """
            Return the full file path of `acct_predict.csv`.

            Returns
            -------
            str
                Full path to the prediction-account CSV file.
            """
            return os.path.join(Config.data_base, self.dataset_dir, self.acct_predict)

        @property
        def acct_transaction_path(self):
            """
            Return the full file path of `acct_transaction.csv`.

            Returns
            -------
            str
                Full path to the transaction CSV file.
            """
            return os.path.join(Config.data_base, self.dataset_dir, self.acct_transaction)

        def __init__(self, root: Optional[str] = None):
            """
            Initialize the data manager and attach a DatasetStore instance.

            Parameters
            ----------
            root : str, optional
                Custom root directory for data storage. If omitted, defaults to
                `Config.data_base`.
            """
            base = root or Config.data_base
            self.store = DatasetStore(base)

        def save(self, name: str,
                 df: pd.DataFrame,
                 description: str = "",
                 created_by: str = "",
                 fmt: str = "parquet") -> str:
            """
            Save a DataFrame as a versioned dataset with metadata.

            Parameters
            ----------
            name : str
                Dataset name.
            df : pd.DataFrame
                DataFrame to store.
            description : str, optional
                Description of the dataset contents.
            created_by : str, optional
                Identifier of the creator or process.
            fmt : {"parquet", "csv"}, optional
                Preferred storage format.

            Returns
            -------
            str
                Version identifier generated or reused by DatasetStore.
            """
            meta = {
                "description": description,
                "created_by": created_by
            }
            return self.store.save(name, df, meta=meta, fmt=fmt)

        def list(self, name: str):
            """
            Display all versions of a dataset and their metadata.

            Parameters
            ----------
            name : str
                Dataset name.

            Returns
            -------
            List[str]
                List of version identifiers.
            """
            dataset_dir = Path(self.dataset_dir_path) / name
            if not dataset_dir.exists():
                print(f"找不到資料集：{name}")
                return []

            versions = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
            if not versions:
                print(f"尚未儲存任何版本：{name}")
                return []

            print(f"\n{name}版本列表：")
            print("-" * 40)
            for vdir in versions:
                meta_path = vdir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {}
                print(f"{vdir.name} : {meta}")
            print("-" * 40)
            print(f"共 {len(versions)} 個版本")

            return [vdir.name for vdir in versions]

        def list_versions(self, name: str) -> List[str]:
            """
            Return all version identifiers for a dataset.

            Parameters
            ----------
            name : str
                Dataset name.

            Returns
            -------
            List[str]
                Sorted version identifiers.
            """
            return self.store.list_versions(name)

        def load(self, name: str, version: str = "latest") -> pd.DataFrame:
            """
            Load dataset content as a pandas DataFrame.

            Parameters
            ----------
            name : str
                Dataset name.
            version : str, optional
                Version identifier or "latest" (default).

            Returns
            -------
            pd.DataFrame
                Loaded dataset.
            """
            return self.store.load(name, version)

        def load_meta(self, name: str, version: str = "latest") -> Dict:
            """
            Load metadata for a dataset version.

            Parameters
            ----------
            name : str
                Dataset name.
            version : str, optional
                Version identifier or "latest".

            Returns
            -------
            Dict
                Metadata stored in `meta.json`.
            """
            return self.store.load_meta(name, version)

        def update(self, name: str, version: str,
                   df: Optional[pd.DataFrame] = None,
                   description: str = "",
                   created_by: str = "",
                   fmt: Optional[str] = None):
            """
            Update an existing dataset version, modifying data and/or metadata.

            Parameters
            ----------
            name : str
                Dataset name.
            version : str
                Version identifier to update.
            df : pd.DataFrame, optional
                New data to replace the existing content.
            description : str, optional
                New description text for metadata.
            created_by : str, optional
                Updated creator identifier.
            fmt : {"parquet", "csv"}, optional
                Optional explicit output format for data rewrite.

            Returns
            -------
            None
            """
            meta = {
                "description": description,
                "created_by": created_by
            }
            return self.store.update(name, version, df=df, meta=meta, fmt=fmt)

        def delete(self, name: str, version: Optional[str] = None):
            """
            Delete a dataset or a specific version.

            Parameters
            ----------
            name : str
                Dataset name.
            version : str, optional
                Version identifier to delete. If omitted, the entire dataset
                directory is removed.

            Returns
            -------
            None
            """
            return self.store.delete(name, version)

        def prune(self, name: str, keep: int = 3) -> List[str]:
            """
            Remove older versions, keeping only the most recent ones.

            Parameters
            ----------
            name : str
                Dataset name.
            keep : int, optional
                Number of versions to retain. Defaults to 3.

            Returns
            -------
            List[str]
                Version identifiers that were deleted.
            """
            return self.store.prune(name, keep=keep)
