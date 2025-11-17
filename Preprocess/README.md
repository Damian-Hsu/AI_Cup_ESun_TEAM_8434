Preprocess
==========

簡介
----
`Preprocess` 資料夾包含專案中負責資料前處理的腳本，將 raw transaction / account 資料整理成模型可用的 features。主要檔案：

- `according_to_account.py`：以帳戶（account）為單位進行彙整與特徵工程，適合產生每個帳戶的聚合特徵（例如交易數、總金額、平均交易間隔等）。
- `according_to_transaction.py`：以交易（transaction）為單位進行轉換或衍生特徵，適合保留交易層級的紀錄與相依特徵。

使用方式（一般建議）
------------------
1. 讀入原始交易或帳戶資料（CSV / DataFrame）。
2. 使用 `according_to_transaction` 處理交易層特徵（若需要）。
3. 使用 `according_to_account` 產生每個帳戶的聚合特徵（若模型需要帳戶級別的輸入）。

範例（Python）
----------------
```python
import pandas as pd
from Preprocess import according_to_transaction as pat
from Preprocess import according_to_account as paa

# 讀入 raw data
# df = pd.read_csv('data/transactions.csv')

# 建議流程：先做交易層級的轉換
# df_tx = pat.process_transactions(df)  # 範例函式名稱，請依實際實作替換

# 再根據交易資料彙整成帳戶特徵
# df_account = paa.aggregate_by_account(df_tx)  # 範例函式名稱
```

