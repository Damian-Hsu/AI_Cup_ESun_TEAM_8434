import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(
    df_train : pd.DataFrame,
    df_test : pd.DataFrame,
    sample_0_rate: float = 0.8,
    sample: bool = True,
    split_val_rate: float = 0.2,
    minmax_scaler: bool = True,
    random_state: int = 815
):  
    """
    Load, preprocess, split, and optionally scale the training and testing datasets.

    This function performs the complete preparation pipeline for model training.
    It supports negative-class downsampling, train/validation splitting, and
    optional MinMax scaling. The function returns a structured dictionary containing
    all processed datasets required for XGBoost or other ML models.

    Steps performed:
    1. Removes non-feature columns ("acct", "is_alert") from raw data.
    2. Prints dataset statistics and alert distribution.
    3. Optionally samples the negative class (label 0) using `sample_0_rate`.
    4. Splits data into training and validation sets using stratified sampling.
    5. Applies MinMaxScaler (optional) to train, validation, and test sets.
    6. Returns all processed DataFrames and the scaler instance.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset containing features and the `is_alert` label.
    df_test : pd.DataFrame
        Testing dataset containing features only.
    sample_0_rate : float, optional
        Fraction of negative-class (0) samples to keep. Default is 0.8.
    sample : bool, optional
        Whether to apply sampling to class 0. Default is True.
    split_val_rate : float, optional
        Ratio of validation set size in train/val split. Default is 0.2.
    minmax_scaler : bool, optional
        Whether to apply MinMaxScaler to all feature matrices. Default is True.
    random_state : int, optional
        Random seed for reproducibility. Default is 815.

    Returns
    -------
    dict
        A dictionary containing:
        - "df_train_sampled": DataFrame used for splitting (after sampling if enabled)
        - "x_train": Feature matrix for training
        - "y_train": Training labels
        - "x_val": Feature matrix for validation
        - "y_val": Validation labels
        - "x_test": Processed test features
        - "scaler": The fitted MinMaxScaler instance (or None)
    """

    x_train, x_val, y_train, y_val = None, None, None, None
    x_test = None
    def _p(text: str):
        print(f"[ load data ] {text}")
    _p("載入參數：")
    _p(f"  sample_0_rate: {sample_0_rate}")
    _p(f"  sample: {sample}")
    _p(f"  split_val_rate: {split_val_rate}")
    _p(f"  minmax_scaler: {minmax_scaler}")
    _p(f"  random_state: {random_state}")

    df_train = df_train.copy().drop(columns=["acct",])
    df_test = df_test.copy().drop(columns=["acct","is_alert"])
    _p(f"已載入 train 與 test 資料：訓練 => {df_train.shape}, 測試 => {df_test.shape}")

    _alert_counts = df_train["is_alert"].value_counts().to_dict()
    _p(f"目前警示帳戶數量：{_alert_counts[1]}")
    _p(f"目前非警示帳戶數量：{_alert_counts[0]}")

    if sample:
        # 抽樣訓練資料，減少0類別
        df_train_0 = df_train[df_train["is_alert"] == 0]
        df_train_1 = df_train[df_train["is_alert"] == 1]
        # 0類別抽樣
        df_train_0_sampled = df_train_0.sample(frac=sample_0_rate, random_state=random_state)
        # 合併1類別與抽樣後的0類別
        df_train_sampled = pd.concat([df_train_0_sampled, df_train_1])
        _p(f"抽樣後 train 資料：{df_train_sampled.shape}")
    
        x_train, x_val, y_train, y_val = train_test_split(
            df_train_sampled.drop(columns=["is_alert"]),
            df_train_sampled["is_alert"],
            test_size=split_val_rate,
            random_state=random_state,
            stratify=df_train_sampled["is_alert"])
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            df_train.drop(columns=["is_alert"]),
            df_train["is_alert"],
            test_size=split_val_rate,
            random_state=random_state,
            stratify=df_train["is_alert"])
    _p(f"切割訓練與驗證資料：訓練 => {x_train.shape}, 驗證 => {x_val.shape}")
    _p("開始前處理...")


    if minmax_scaler:
        scaler = MinMaxScaler()
        x_res = pd.DataFrame(scaler.fit_transform(x_res), columns=x_train.columns)
        x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
        x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(df_test), columns=x_train.columns)
        _p("已進行 MinMaxScaler")
    else:
        x_test = df_test.copy()
        scaler = None
        _p("未進行 MinMaxScaler，直接使用原始資料")
    _p("前處理完成")
    return {
        "df_train_sampled": df_train_sampled if sample else df_train,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "scaler":scaler,
    }
def split_txn_by_n(
    df: pd.DataFrame,
    n:int = 2
):
    """
    Split accounts into two groups based on transaction count threshold.

    This function calculates each account's total number of transactions
    (send_count + recv_count) and separates the input DataFrame into:
    1. Accounts with <= n transactions.
    2. Accounts with > n transactions.

    This is useful for scenarios where low-transaction accounts exhibit
    different patterns and require separate handling in modeling or feature
    engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing transaction features, including
        'send_count' and 'recv_count'.
    n : int, optional
        Threshold number of transactions used to split accounts.
        Default is 2.

    Returns
    -------
    tuple of pd.DataFrame
        (df_n_txn, df_multi_txn), where:
        - df_n_txn: Accounts with <= n transactions
        - df_multi_txn: Accounts with > n transactions

    Both returned DataFrames exclude the auxiliary column `txn_count`.
    """

    def _p(text: str):
        print(f"[ split data by txt ] {text}")
    df["txn_count"] = df[["send_count","recv_count"]].sum(axis=1)
    df_n_txn = df[df["txn_count"] <= n ].reset_index(drop=True)
    df_multi_txn = df[df["txn_count"] > n].reset_index(drop=True)
    total_accts = len(df)
    single_tx = df_n_txn.shape[0]
    multi_tx = total_accts - single_tx
    _p("帳號交易數統計")
    _p(f"  總帳號數：{total_accts:,}")
    _p(f"  僅 {n} 筆交易帳號：{single_tx:,}（{single_tx / total_accts:.2%}）")
    _p(f"  超過 {n} 筆帳號：{multi_tx:,}（{multi_tx / total_accts:.2%}）")
    return df_n_txn.drop(columns=["txn_count"]), df_multi_txn.drop(columns=["txn_count"])
