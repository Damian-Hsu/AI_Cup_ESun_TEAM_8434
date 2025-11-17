from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
from Preprocess.according_to_account import load_data, split_txn_by_n
def print_evaluation(model, x, y, threshold ,prefix="") -> None:
    """Print evaluation metrics for a given model and dataset."""
    def _p(text: str):
        """Print progress messages for print_evaluation function."""
        print(f"[ evaluation ({prefix}) ]  {text}")
    y_pred = (model.predict_proba(x)[:, 1] >= threshold).astype(int)
    _p("Confusion Matrix:")
    print(f"\n{confusion_matrix(y, y_pred)}")
    _p("Classification Report:")
    print(f"\n{classification_report(y, y_pred)}")

def get_low_importances_df(del_n:int = 150,
                           best_params:dict = None,
                           x_train = None,
                           y_train = None,
                           x_val = None,
                           y_val = None) -> list:
    """
    Compute feature importances using XGBoost and return the least important features.

    This function trains an XGBoost model with the provided parameters and datasets,
    extracts the feature importance scores, and returns the bottom `del_n` features
    (rank-sorted by ascending importance). It is typically used for feature pruning
    or iterative feature selection where low-value features are removed to simplify
    the model or improve generalization.

    Parameters
    ----------
    del_n : int, optional
        Number of lowest-importance features to return. Default is 150.
    best_params : dict
        Dictionary of optimized XGBoost hyperparameters, including keys:
        {"n_estimators", "learning_rate", "max_depth", "scale_pos_weight",
        "subsample", "threshold"}.
    x_train : pandas.DataFrame
        Training feature matrix.
    y_train : array-like
        Training labels.
    x_val : pandas.DataFrame
        Validation feature matrix.
    y_val : array-like
        Validation labels.

    Returns
    -------
    list
        A list containing the names of the least important `del_n` features.
    """

    def _p(text: str):
        """Print progress messages for get_low_importances_df function."""
        print(f"[ get low importances ] {text}")

    _p("第一次訓練 XGBClassifier 以取得特徵重要性...")
    xgb_m_org = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_jobs=-1,
        scale_pos_weight=best_params['scale_pos_weight'],
        subsample=best_params['subsample']
    )
    xgb_m_org.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False)
    _p("取得特徵重要性完成，開始評估模型...")
    # 預測本次訓練的訓練集
    print_evaluation(xgb_m_org, x_train, y_train,best_params["threshold"], prefix="Train Multi-Txn")
    # 預測本次訓練的驗證集
    print_evaluation(xgb_m_org, x_val, y_val, best_params["threshold"], prefix="Validation Multi-Txn")
    columns = x_train.columns
    importances = xgb_m_org.feature_importances_
    importances_df = pd.DataFrame({
        "feature": columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    _p(f"移除重要性最低的 {del_n} 個特徵")
    return importances_df.tail(del_n)["feature"].tolist()

def train_model(best_params:dict = None,
                x_train = None,
                y_train = None,
                x_val = None,
                y_val = None)-> XGBClassifier:
    """
    Train an XGBoost model using the best-found hyperparameters and return it.

    This function creates and fits an XGBoost classifier using the provided
    training and validation datasets. The intended use is to train the final
    model after hyperparameter optimization or feature selection has been
    completed. The model is evaluated silently (no verbose output) and the
    trained model instance is returned for downstream inference or evaluation.

    Parameters
    ----------
    best_params : dict
        Dictionary of optimized XGBoost hyperparameters, including:
        {"n_estimators", "learning_rate", "max_depth", "scale_pos_weight",
        "subsample"}.
    x_train : pandas.DataFrame
        Training feature matrix.
    y_train : array-like
        Training labels.
    x_val : pandas.DataFrame
        Validation feature matrix used for eval_set monitoring.
    y_val : array-like
        Validation labels.

    Returns
    -------
    XGBClassifier
        The fully trained XGBoost model.
    """

    def _p(text: str):
        """Print progress messages for train_model function."""
        print(f"[ train model ] {text}")

    _p("開始訓練最終模型...")
    model = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_jobs=-1,
        scale_pos_weight=best_params['scale_pos_weight'],
        subsample=best_params['subsample'],
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False)
    _p("最終模型訓練完成")
    return model

