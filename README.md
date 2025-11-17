
# AI_Cup_ESun_TEAM_8434

2025 玉山人工智慧公開挑戰賽 — 初賽資格審查
可復現模型程式碼（Python 3.11）

---

## 📌 專案介紹

本專案為 **TEAM_8434** 參加 **2025 玉山人工智慧公開挑戰賽 – 初賽** 所使用的完整可復現程式碼。

主旨為 **偵測警示帳戶（Fraudulent Account Detection）**，透過大量交易記錄建構 **使用者層級（Account-level）特徵**，並以 **XGBoost** 完成二元分類。

本 Repository 包含：

✔ 資料前處理  
✔ 使用者層級特徵工程（數百維 Feature，包括 Networkit 圖特徵）  
✔ 訓練流程與模型產出  
✔ 最終提交檔案產生  
✔ 可直接執行的 `main.py`  

---

## 🏗️ 專案架構

```
AI_Cup_ESun_TEAM_8434/
│── main.py                     # 主執行檔（可直接復現結果）
│── requirements.txt            # 套件版本
│── README.md                   # 專案說明文件（本檔案）
│── .gitignore
│
├── Preprocess/                 # 前處理模組
│   ├── according_to_account.py # 依帳戶彙整的特徵前處理
│   ├── according_to_transaction.py # 依交易紀錄生成長表與特徵工程（包含 Networkit 圖特徵）
│   ├── README.md
│   └── __init__.py
│
├── Model/
│   ├── get_low_importances_df # 特徵重要性過濾
│   ├── train_model.py         # 模型訓練腳本
│   ├── README.md
│   └── __init__.py
│
└── tool/
    ├── config.py              # DatasetStore 與資料存取
    ├── _dataset_store.py      # 資料版本管理（自動儲存 dataset/ 版本）
    ├── README.md
    └── __init__.py
```

---

## 🧰 使用環境

* **Python：3.11**
* **CPU:** Intel i9-14900K
* **RAM:** 32 GB
* **GPU:** NVIDIA RTX A1000（僅部分步驟加速，非必要）

> ⚠️ 注意：完整特徵工程（according_to_transaction）耗時約 2~3 小時，若使用者需要，可改為讀取本專案所儲存的 processed dataset。

---

## 📦 套件需求

請先安裝：

```
pip install -r requirements.txt
```

（內含 pandas、numpy、networkit、xgboost、scikit-learn、psutil、tqdm 等）

---

## 🚀 如何復現結果（直接執行即可）

本專案已設計好完整流程，只需要執行：

```
python main.py
```

主程式會自動完成：

1. 檢查 `.data/dataset` 是否存在
2. 若無 → 自動從 Google Drive **下載比賽原始資料**
3. 若已存在 → 直接載入前處理後的 `df_user_train_gf_all`、`df_user_test_gf_all`
4. 依帳號交易筆數切割（≤2 筆不訓練，≥3 筆進模型）
5. 執行 `get_low_importances_df()` 移除低重要性特徵
6. 使用全部資料 **訓練最終 XGBoost 模型**
7. 預測 test set
8. 產生 **submission.csv**

輸出檔案：

```
submission_no_res_p0_0006_no_lteq2txn_v3data_D150.csv
```

---

## 🤖 模型結構與訓練流程

### ✔ 模型：XGBoost (binary:logistic)

本隊使用 Optuna 搜尋後得到最佳參數：

```python
best_params = {
    'n_estimators': 1154,
    'learning_rate': 0.12929902001851637,
    'max_depth': 4,
    'scale_pos_weight': 19.932066980434552,
    'subsample': 0.7355483010609939,
    'threshold': 0.30158814999761946
}
```

### ✔ 特徵工程亮點

建立 **超過 500+ account-level features**，包含：

#### 🔹 基礎統計

* send/recv count
* active days
* txn interval entropy
* 3-hour time bin distribution

#### 🔹 金額行為

* 金額均值、標準差、IQR、CV
* skew/kurt
* 尾端比例（q90/q95/q99）
* Benford’s Law 偏離度

#### 🔹 對手特徵

* counterparty unique
* HHI concentration
* Top-1 / Top-3 amount share
* bidirectional ratio

#### 🔹 Session 行為

* 5/15/30/60 min sessions
* burst features

#### 🔹 Networkit 圖特徵

基於帳號間交易金額作為 weighted edges：

* PageRank
* Katz Centrality
* Weighted In/Out Degree
* Approx Betweenness
* K-core decomposition
* Connected component size
* 2-hop / 3-hop neighbor size

> **注意**：本步驟為主要運算瓶頸，需大量 RAM 與 CPU 時間。

#### 🔹 對手帳戶二次聚合 (Account-of-Counterparty Aggregation)

將 **對手帳戶特徵再次聚合**，產生二階資訊
如：平均對手 burst、平均對手 entropy、加權 mean 等。

---

## 📊 實驗結果

本專案最終模型於 public leaderboard 取得：

* **Public F1 ≈ 0.445 ~ 0.46**（依不同資料切割模擬）
* 已成功產生可交付之 submission
* 與原始上傳結果完全一致，可 100% 復現
---

## 📝 如何使用你訓練好的模型

本專案採「程式內部直接訓練」方式，**不需要附模型檔**。
若需存模型，可自行加入：

```python
import joblib
joblib.dump(model, "model.pkl")
```

---

## 📁 資料來源（DATA）

若 `.data/dataset/` 不存在，程式會自動下載：

```
https://drive.google.com/drive/folders/1QHp8hYPKWiBmGfOFLQrHMttbpB9rh6GZ?usp=sharing
```

