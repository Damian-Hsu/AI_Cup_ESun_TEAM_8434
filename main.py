import tool
import gc
import pandas as pd
from Preprocess.according_to_account import load_data, split_txn_by_n
from Model import train_model, get_low_importances_df

# 檢查.data 中是否有dataset，沒有的話就下載
import os
data_dir = "./.data/dataset"

if not os.path.isdir(data_dir):
    print("Dataset folder not found. Downloading...")
    tool.download_dataset()
else:
    print("Dataset folder exists.")

cfg_data = tool.Config.Data()
_p = lambda text: print(f"[ main ] {text}")
USE_EXISTING_DATA = True # 使用現有的前處理資料，重新前處理要花3個小時以上

if USE_EXISTING_DATA: # 使用現有的前處理資料
    _p("使用現有的前處理資料...")
    try:
        _p("載入資料中...")
        df_user_train = cfg_data.load("df_user_train_gf_all")
        df_user_test = cfg_data.load("df_user_test_gf_all")
    except Exception as e:
        print(f"載入資料失敗: {e}")
        exit(1)
else:
    _p("重新前處理資料...")
    # 載入according_to_transaction的函式
    from Preprocess.according_to_transaction import get_preprocessed_data, get_feature_engineering_data
    # 基本前處理
    _p("載入基礎資料集")
    df_txn = pd.read_csv(cfg_data.acct_transaction_path) # 交易資料
    df_alert = pd.read_csv(cfg_data.acct_alert_path) # 警示帳戶資料
    # 特徵工程
    df_txn_prep = get_preprocessed_data(df_txn)
    df_user_features = get_feature_engineering_data(
                                df_alert= df_alert,
                                df_txn= df_txn_prep,
                                verbose= True,
                                include_event_day= True
                            )
    # 儲存使用者層級的彙總資料
    cfg_data.save(f"df_user_features_gf_all",
                  df_user_features,
                    description=f"使用者層級的彙總資料，包含事件當天的交易，新特徵包括爆發/節奏、對手集中度、金額分布形狀等，重寫圖特徵，以及對手帳戶特徵的二次聚合。圖特徵使用 Networkit 計算，包含 PageRank、Katz、Connected Components、Degree、Approximate Betweenness、k-core、2-hop/3-hop neighborhood size 等。",
                    created_by="HungChi"
                    )
    # 標註警示帳戶
    df_user_features = df_user_features.merge(df_alert[['acct', 'event_date']].drop_duplicates(),
                        on='acct', how='left')
    df_user_features['is_alert'] = df_user_features['event_date'].notna().astype(int)
    df_user_features.drop(columns=['event_date'], inplace=True)
    df_test = pd.read_csv(cfg_data.acct_predict_path)
    df_user_test = df_user_features[df_user_features['acct'].isin(df_test['acct'])].copy()
    df_user_train = df_user_features[~df_user_features['acct'].isin(df_test['acct'])].copy()
    # 儲存最終的train與test資料
    cfg_data.save("df_user_train_gf_all",
              df_user_train,
                description="訓練資料，包含當天交易以及圖特徵(新)，以及對手帳戶特徵的二次聚合。",
                created_by="HungChi"
                )
    cfg_data.save("df_user_test_gf_all",
                df_user_test,
                    description="測試資料，包含當天交易以及圖特徵(新)，以及對手帳戶特徵的二次聚合。",
                    created_by="HungChi"
                    )
    del df_txn, df_alert, df_txn_prep, df_user_features
    gc.collect()
_p(f"已載入 train 與 test 資料：訓練 => {df_user_train.shape}, 測試 => {df_user_test.shape}")
_p("根據交易筆數切割資料...(預設兩筆)")
# 切割成單筆交易與多筆交易的資料
df_train_one_txn, df_train_multi_txn = split_txn_by_n(df_user_train)
df_test_one_txn, df_test_multi_txn = split_txn_by_n(df_user_test)
data_dict_multi = load_data(
    df_train=df_train_multi_txn,
    df_test=df_test_multi_txn,
    sample=False,
    split_val_rate=0.2,
    minmax_scaler=False,
    random_state= 999
)

x_m_train = data_dict_multi["x_train"]
y_m_train = data_dict_multi["y_train"]
x_m_val = data_dict_multi["x_val"]
y_m_val = data_dict_multi["y_val"]
x_m_test = data_dict_multi["x_test"]


del data_dict_multi
gc.collect()

_p(f"Multi-Txn Data Shapes: x_m_train: {x_m_train.shape}, y_m_train: {y_m_train.shape}, x_m_val: {x_m_val.shape}, y_m_val: {y_m_val.shape}, x_m_test: {x_m_test.shape}")
_p("前處理完成，進入模型訓練階段...")

# 這是使用optuna調參後的最佳參數
best_params= {'n_estimators': 1154,
              'learning_rate': 0.12929902001851637,
              'max_depth': 4,
              'scale_pos_weight': 19.932066980434552,
              'subsample': 0.7355483010609939,
              'threshold': 0.30158814999761946}

low_importance_features = get_low_importances_df(
    del_n=150,
    best_params=best_params,
    x_train=x_m_train,
    y_train=y_m_train,
    x_val=x_m_val,
    y_val=y_m_val
)
_p(f"低重要性特徵共 {len(low_importance_features)} 個")
_p(f"前10個為：{low_importance_features[:10]}...")

# 載入比賽測試集並使用全部樣本訓練模型
_p("使用全部樣本訓練最終模型...")

df_user_train = cfg_data.load("df_user_train_gf_all") # 全部的train資料
df_user_test = cfg_data.load("df_user_test_gf_all") # 全部的test資料
_, df_train_multi_txn_all = split_txn_by_n(df_user_train) # 只取3筆以上的交易資料(2筆以下的被丟掉)

# 切成特徵與標籤
x_all, y_all = df_train_multi_txn_all.drop(columns=["is_alert","acct"]), df_train_multi_txn_all["is_alert"]

# 取得測試集中<=2筆交易的帳號清單
n_txn_acct = df_user_test["acct"][df_user_test["total_txn_count"] <= 2 ].tolist()

# 測試集特徵
test_feature = df_user_test.drop(columns=["acct","is_alert"]).copy()

# 測試集帳號
test_ids = df_user_test["acct"].copy()

# 移除低重要性特徵
x_all_drop_low_importance = x_all.drop(columns=low_importance_features)
test_all_drop_low_importance = test_feature.drop(columns=low_importance_features)

# 記憶體控制
del df_user_train, df_user_test, df_train_multi_txn_all, test_feature, _, x_all
gc.collect()

model = train_model(
    best_params=best_params,
    x_train=x_all_drop_low_importance,
    y_train=y_all,
    x_val=x_all_drop_low_importance,
    y_val=y_all
)

_p("模型訓練完成，開始預測比賽測試集...")

proba = model.predict_proba(test_all_drop_low_importance)
p1 = proba[:, 1]
df_user_test_pred = (p1 >= 0.0006).astype(int)
submission = pd.DataFrame({
    "acct": test_ids,
    "label": df_user_test_pred
})
submission.loc[submission["acct"].isin(n_txn_acct), "label"] = 0
# 存檔
submission.to_csv("submission_no_res_p0_0006_no_lteq2txn_v3data_D150.csv", index=False)
_p("預測完成，提交檔案已儲存為 submission_no_res_p0_0006_no_lteq2txn_v3data_D150.csv")
_p("提交檔案中各類別數量：")
print(submission["label"].value_counts())

