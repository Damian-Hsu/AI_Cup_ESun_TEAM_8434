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

輸入 / 輸出
-------------
- 假設輸入為原始交易紀錄（含帳戶 ID、時間戳、金額、交易類型等），輸出為可直接交給模型的 feature 表（Pandas DataFrame）。
- 若需要將中間結果儲存為檔案（CSV / parquet），請在呼叫端處理 I/O，或在腳本中加入明確的 `save` 參數。

注意事項與假設
-----------------
- 範例中函式名稱僅為示意，實際 API 請以程式碼中定義之 function/class 為準。
- 請確保時間欄位已正確解析為 datetime 型別，以便計算時間差、滾動統計等。
- 輸出特徵應包含清楚命名，避免與原始欄位衝突（例如用 `agg_` 前綴）。

測試與驗證
---------------
- 建議針對每個處理步驟撰寫單元測試（小型 DataFrame 驗證欄位、資料類型與統計值）。
- 確認處理後的 NA 行為（填補 / 刪除）符合模型需求。

延伸建議
---------
- 若需要大量資料處理，可將步驟拆成小函式並支援 chunk / streaming。
- 可以加入 CLI 介面（例如 `python -m Preprocess.according_to_account --input x.csv --output y.csv`）以便在 pipeline 中使用。

