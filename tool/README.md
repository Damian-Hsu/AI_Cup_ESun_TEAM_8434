tool
=====

簡介
----
`tool` 資料夾包含專案中供資料載入、儲存與共用設定的輔助模組，常見檔案如下：

- `_dataset_store.py`：封裝資料集的讀寫介面（load / save / cache）。可做為集中管理資料來源的地方。
- `config.py`：放置路徑、常數、參數等設定，供專案其他模組引用。

主要功能
--------
- 集中管理資料讀寫（例如 train / val / test 資料集）。
- 提供統一的配置入口，減少硬編碼路徑。

使用範例
---------
（範例為示意，實際函式名稱請依程式碼調整）

```python
from tool import _dataset_store as ds
from tool import config

# 讀取資料
# df = ds.load_dataset('train')

# 或指定路徑
# df = ds.load_from_path(config.DATA_DIR / 'train.csv')

# 儲存處理後資料
# ds.save_dataset(df, 'processed_train')
```

如何新增資料集
---------------
1. 在 `config.py` 新增對應的路徑或 key（例如：DATA_PATHS['newset'] = 'path/to/newset'）。
2. 在 `_dataset_store.py` 新增對應的 loader / saver 或在現有的 load 接口中加入 key 處理。
3. 加上必要的文件說明與測試，確保在 CI / pipeline 中可以正確載入。

建議
-----
- 在 `_dataset_store.py` 中實作容錯與日誌（例如檔案不存在時拋出清楚錯誤、或回傳空 DataFrame）。
- 將敏感或環境相關設定（如絕對路徑）抽到 `config.py`，並使用環境變數或 .env 管理不同環境的設定。

