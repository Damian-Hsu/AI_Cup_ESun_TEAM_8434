import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import time, os, gc
from typing import Dict
import networkit as nk
import psutil
from pandas.api.types import CategoricalDtype
from collections import defaultdict
def get_preprocessed_data( df_txn: pd.DataFrame ) -> pd.DataFrame:
    """
    Preprocess transaction data by converting categorical fields, normalizing
    transaction amounts to TWD, transforming time representations, and producing
    a cleaned, consistently structured DataFrame for downstream feature engineering.

    This function performs several key preprocessing steps:

    1. Convert selected columns to categorical dtype to reduce memory footprint
    and improve processing efficiency.
    2. Map transaction amounts into New Taiwan Dollars (TWD) using predefined
    exchange rates and remove the original raw amount column.
    3. Convert transaction time into an integer representing seconds since midnight
    and compute an absolute timestamp (`time_abs`) combining date and time.
    4. Rename columns for consistency and reorder them to a standardized schema.
    5. Convert account-type codes into boolean indicators representing whether an
    account belongs to E.SUN Bank.

    Parameters
    ----------
    df_txn : pd.DataFrame
        Raw transaction table containing fields such as from_acct, to_acct,
        account types, transaction amount, currency_type, channel_type,
        txn_date, and txn_time.

    Returns
    -------
    pd.DataFrame
        A cleaned and transformed DataFrame ready for feature extraction,
        containing standardized column names, normalized transaction amounts,
        categorical types, temporal encodings, and boolean account indicators.
    """

    def _p(text):
        print(f"[Preprocess] {text}")
    # 轉換類別欄位
    _p("Convert categorical columns")
    convert_target = ['from_acct', 'from_acct_type', 'to_acct', 'to_acct_type', 'is_self_txn', 'currency_type', 'channel_type']
    for col in convert_target:
        df_txn[col] = df_txn[col].astype('category')
    # 設定匯率（兌換成 TWD）
    exchange_rates = {
        'USD': 30.0,
        'JPY': 0.21,
        'AUD': 20.0,
        'CNY': 4.3,
        'EUR': 36.0,
        'SEK': 2.7,
        'GBP': 41.0,
        'HKD': 3.8,
        'THB': 0.85,
        'NZD': 18.0,
        'CAD': 22.5,
        'CHF': 33.5,
        'SGD': 22.2,
        'ZAR': 1.7,
        'MXN': 1.8,
        'TWD': 1.0 
    }
    _p("Convert transaction amount to TWD")
    # 建立新的欄位 txn_amt_twd
    rates = pd.Series(exchange_rates, dtype='float64')
    rate = df_txn['currency_type'].astype(str).map(rates)
    df_txn['txn_amt_twd'] = pd.to_numeric(df_txn['txn_amt'], errors='coerce') * rate
    df_txn['txn_amt_twd'] = df_txn['txn_amt_twd'].astype(float)
    df_txn = df_txn.drop(columns=['txn_amt']) # 去除原始交易金額欄位
    _p("Convert time to integer categories")
    # 將時間轉為整數類別，把時間欄位轉成「一天中的秒數」
    df_txn['time'] = pd.to_timedelta(df_txn['txn_time']).dt.total_seconds().astype(int)
    df_txn.drop(columns=['txn_time'], inplace=True)
    # 從起始點開始的總秒數
    day_to_second = 24*60*60
    df_txn['time_abs'] = df_txn['txn_date'] * day_to_second + df_txn['time']
    _p("Rename and reorder columns")
    # 重新命名欄位
    df_txn.rename(columns={
        "txn_date": "date",
        "txn_amt_twd" : "amount",
        "is_self_txn" : "is_self"
    }, inplace=True)
    _p("Finalize column order")
    # 重新排序欄位
    df_txn = df_txn[[
        "from_acct", "from_acct_type", "to_acct", "to_acct_type",
        "is_self", "date", "time", "time_abs", "currency_type",
        "amount", "channel_type"
    ]]
    _p("Convert account types to boolean indicators")
    # 將 acct_type 變成是否為 esun 帳戶
    df_txn['from_acct_type'] = (df_txn['from_acct_type'] == '01').astype("bool")  
    df_txn['to_acct_type'] = (df_txn['to_acct_type'] == '01').astype("bool")
    df_txn.rename(columns={
        "from_acct_type": "from_is_esun",
        "to_acct_type": "to_is_esun"
    }, inplace=True)
    _p("Preprocessing complete")
    return df_txn


def get_feature_engineering_data(
        df_txn: pd.DataFrame ,
        df_alert: pd.DataFrame ,
        include_event_day: bool = True,
        verbose: bool = True
):
    """
    Compute per-account feature matrix from raw transaction logs and alert dates for fraud modeling.

    This function performs a full feature-engineering pipeline that converts
    transaction-level data (`df_txn`) into an account-level feature table
    (`df_user_features`). It integrates temporal patterns, amount distributions,
    counterparty structure, session behavior, and graph-based connectivity
    into a single DataFrame suitable for downstream modeling (e.g., fraud
    detection on accounts).

    ## Main steps and feature groups

    1. **Type alignment & numeric safety**

    * Aligns categorical types for `from_acct` / `to_acct` / `acct`.
    * Normalizes `amount`, `date`, `time`, and `time_abs` into numeric types.
    * Ensures `is_self` has consistent categories (Y/N/UNK).

    2. **Long-format transformation (send/recv view)**

    * Builds a long table with one row per direction:

        * `acct`, `counterparty`, `role` ("send" / "recv"), `date`, `abs_sec`,
        `hour`, `currency`, `channel`, `amount`.
    * Aligns account categories across all views.

    3. **Observation window control using `df_alert`**

    * Merges `event_date` from `df_alert` by `acct`.
    * Filters transactions to only include those before (or up to) the event
        day, depending on `include_event_day`.

    4. **Basic activity statistics**

    * `send_count`, `recv_count`, `total_txn_count`.
    * Per-day stats: mean/max/min/std of daily counts, CV, `active_days`.
    * Inter-transaction intervals: mean/min gap, proportion of short gaps.

    5. **Time-of-day structure**

    * 3-hour bins (8 segments per day) transaction counts.
    * Night-time (hour < 6) transaction ratio.
    * Hour-of-day entropy (`txn_hour_entropy`).

    6. **Amount distribution & shape**

    * `amount_max`, `amount_mean`, `amount_std`, `amount_sum`,
        `amount_median`, IQR, CV, skewness, kurtosis.
    * Gini coefficient for amount concentration.
    * Tail proportions above q90/q95/q99 (`amount_prop_gt_q*`).
    * Proportion of near-round amounts (e.g. near multiples of 1,000).
    * Last-digit entropy of amounts.
    * Benford distance on first digits (`benford_l1`).

    7. **Currency / channel diversity & interactions**

    * Unique counts: `currency_unique`, `channel_unique`.
    * One-hot counts: `currency_count__*`, `channel_count__*`.
    * Entropies: `currency_entropy`, `channel_entropy`.
    * Channel-specific amount statistics (mean / median).
    * Currency-specific amount proportions (`currency_amount_prop__*`).
    * Channel / currency transaction share (`ch_txn_prop__*`, `currency_txn_prop__*`).

    8. **Self-transfer structure**

    * Self-transfer count based on labeled `is_self == "Y"` across both
        from/to directions (`self_txn_count`).
    * One-hot counts and ratios per `is_self` category
        (`is_self_count__*`, `is_self_ratio__*`).
    * Global self-transfer ratio at account level.

    9. **Burstiness & Fano factor**

    * Burst statistics over multiple time granularities (5/15/30/60 minutes):
        `burst_*_max`, `burst_*_mean`, `burst_*_p95`.
    * Per-day Fano factor of transaction counts (`txn_per_day_fano`).

    10. **Counterparty diversity & concentration**

        * Unique counterparties: total/send/recv.
        * Counterparty count entropy (`cp_entropy`).
        * Count-based concentration metrics: HHI, top1/top3 share.
        * Amount-based concentration metrics: top1/top3 amount share.
        * Bidirectional counterparty ratio (`bidir_counterparty_ratio`).

    11. **New counterparty ratios (recency-based)**

        * For each window (7/14/30 days), ratio of counterparties that are
        newly seen near the account’s last observed date:
        `new_counterparty_ratio_7d`, `_14d`, `_30d`.

    12. **Session-based features (multi-gap)**

        * Session segmentation using gaps of 5/15/30/60 minutes.
        * For each gap size, aggregates over session lengths per account:
        `sess_*_max`, `sess_*_mean`, `sess_*_p95`.

    13. **Quick-out / quick-in style features**

        * From recv→next send sequence, within windows 5/15/30/60 minutes:

        * `quick_out_rate_*`: fraction of deposits quickly followed by an outflow.
        * `quick_out_large_ratio_rate_*`: fraction where the outgoing amount
            is a large proportion of the incoming (e.g. ≥ 80%).

    14. **Graph-based features (networkit)**

        * Builds directed and undirected graphs with log(1+amount) edge weights:

        * Nodes: accounts (including both acct and counterparty).
        * Directed graph: send edges only.
        * Undirected graph: combined transactions.
        * Features:

        * PageRank (`pr`) on directed, weighted graph.
        * Connected component size (`component_size`).
        * In/out degrees and weighted degrees (`in_deg`, `out_deg`,
            `win_deg`, `wout_deg`).
        * Approximate/estimated betweenness (`betweenness_approx`).
        * Katz centrality (`katz`) with adaptive alpha.
        * Core number (k-core) on undirected graph (`ud_core_number`).
        * 2-hop and 3-hop neighborhood sizes (`ud_hop2_size`, `ud_hop3_size`).
        * All centrality-like scores are z-normalized within connected components.

    15. **Account-level aggregation base (`base`)**

        * All above per-account features are joined into a single base table.
        * Dtypes are downcast to `float32` / `int32` when possible.

    16. **Second-order counterparty aggregation**

        * Selects a subset of informative base features as "counterparty reference".
        * For each account, aggregates over its counterparties:

        * Unweighted mean / std of counterparty reference features.
        * Amount-weighted mean of counterparty reference features.
        * Coverage metrics: number of unique counterparties.
        * Edge strength statistics: sum/mean of transaction counts and amounts.
        * Results are joined back as `cpref__*`-based aggregated features.

    ## Parameters

    df_txn : pd.DataFrame
    Transaction-level data including at least:
    `from_acct`, `to_acct`, `amount`, `date`, `time`, `time_abs`,
    `currency_type`/`currency`, `channel_type`/`channel`, and `is_self`.
    df_alert : pd.DataFrame
    Account-level alert info, must contain:
    - `acct`: account identifier.
    - `event_date`: label date used to define observation windows.
    include_event_day : bool, optional
    If True, include transactions on the `event_date` in the observation
    window (`date <= event_date`); if False, only include days strictly
    before the event (`date < event_date`). Defaults to True.
    verbose : bool, optional
    If True, prints step-wise logs and memory usage via StepTimer and `_log`.
    Defaults to True.

    ## Returns

    pd.DataFrame
    An account-level feature table `df_user_features` where:
    - Each row corresponds to one account (`acct`).
    - Columns collect all engineered features described above.
    - Numeric columns are downcast to efficient dtypes when possible.

    ## Notes

    * This function is computationally heavy and memory intensive, intended
    for offline feature generation.
    * It relies on external libraries such as `networkit`, `psutil`, `tqdm`,
    `numpy`, and `pandas`, and assumes that transaction times and dates
    have been preprocessed into consistent units.

    
    """
    tqdm.pandas()

    # ---------------------------- 記憶體監控 ----------------------------
    _PROC = psutil.Process(os.getpid())

    # ---------------------------- 小工具 ----------------------------
    def _log(msg, verbose=True):
        if verbose:
            print(f"[FE] {msg}")

    def _mem():
        if _PROC is None: return ""
        rss = _PROC.memory_info().rss / (1024**3)
        return f"{rss:,.2f} GB"

    class StepTimer:
        def __init__(self, name, verbose=True):
            self.name = name; self.verbose = verbose
        def __enter__(self):
            self.t0 = time.perf_counter()
            if self.verbose: print(f"[FE] ▶ {self.name} ... (mem={_mem()})")
            return self
        def __exit__(self, exc_type, exc, tb):
            t = time.perf_counter() - self.t0
            if self.verbose: print(f"[FE] ✓ {self.name} done in {t:,.2f}s (mem={_mem()})")

    def _safe_entropy_from_counts(counts: np.ndarray) -> float:
        s = counts.sum()
        if s <= 0: return 0.0
        p = counts / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _safe_entropy_from_series(cat_series: pd.Series) -> float:
        if cat_series.size == 0: return 0.0
        vc = cat_series.value_counts(dropna=False)
        return _safe_entropy_from_counts(vc.values.astype(float))

    def _safe_entropy_from_hours(hours_series: pd.Series) -> float:
        if hours_series.size == 0: return 0.0
        counts = np.bincount(hours_series.astype(int), minlength=24)
        return _safe_entropy_from_counts(counts.astype(float))

    def _mean_interval(abs_sec_series: pd.Series) -> float:
        n = abs_sec_series.size
        if n <= 1: return 0.0
        arr = np.sort(abs_sec_series.values)
        diffs = np.diff(arr)
        return float(np.mean(diffs)) if diffs.size > 0 else 0.0

    def _min_interval(abs_sec_series: pd.Series) -> float:
        n = abs_sec_series.size
        if n <= 1: return 0.0
        arr = np.sort(abs_sec_series.values)
        diffs = np.diff(arr)
        return float(np.min(diffs)) if diffs.size > 0 else 0.0

    def _prop_short_intervals(abs_sec_series: pd.Series, thr_sec: int = 300) -> float:
        """比例：相鄰兩筆間隔 <= thr_sec（預設5分鐘）"""
        n = abs_sec_series.size
        if n <= 1: return 0.0
        arr = np.sort(abs_sec_series.values)
        diffs = np.diff(arr)
        return float((diffs <= thr_sec).mean()) if diffs.size > 0 else 0.0

    def _gini_from_values(x: np.ndarray) -> float:
        """Gini 係數：金額集中度，越大代表越不均"""
        x = np.asarray(x, dtype=float)
        x = x[x >= 0]
        if x.size == 0: return 0.0
        if np.allclose(x.sum(), 0.0): return 0.0
        # 以常見公式計算 Gini
        x_sorted = np.sort(x)
        n = x_sorted.size
        cumx = np.cumsum(x_sorted)
        g = (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n
        return float(g) if np.isfinite(g) else 0.0

    def _herfindahl_from_counts(counts: np.ndarray) -> float:
        """Herfindahl-Hirschman Index：分配集中度（平方和）"""
        s = counts.sum()
        if s <= 0: return 0.0
        p = counts / s
        return float(np.sum(p * p))

    def _topk_share_from_counts(counts: np.ndarray, k: int) -> float:
        if counts.size == 0: return 0.0
        s = counts.sum()
        if s <= 0: return 0.0
        topk = np.sort(counts)[::-1][:k].sum()
        return float(topk / s)


    def _prop_near_round(amount_series: pd.Series, base: float = 1000.0, tol: float = 10.0) -> float:
        """比例：金額是否落在 base 的倍數 ± tol 之內（例如接近千元整）"""
        x = amount_series.values
        if x.size == 0: return 0.0
        x = np.abs(x)
        mod = np.abs(np.mod(x, base))
        near = (mod <= tol) | (np.abs(base - mod) <= tol)
        return float(near.mean())

    def _last_digit_entropy(amount_series: pd.Series) -> float:
        """金額尾數（以整數金額的個位數）分布熵"""
        x = np.floor(np.abs(amount_series.values)).astype(int)
        if x.size == 0: return 0.0
        dig = x % 10
        counts = np.bincount(dig, minlength=10).astype(float)
        return _safe_entropy_from_counts(counts)

    def _amount_tail_props(amount_series: pd.Series, qs=(0.9, 0.95, 0.99)) -> Dict[str, float]:
        """尾端比例：> q-quantile 的筆數占比"""
        x = amount_series.values
        out = {}
        if x.size == 0:
            for q in qs: out[f"amount_prop_gt_q{int(q*100)}"] = 0.0
            return out
        qv = np.quantile(x, qs)
        for q, thr in zip(qs, qv):
            prop = float((x > thr).mean()) if x.size > 0 else 0.0
            out[f"amount_prop_gt_q{int(q*100)}"] = prop
        return out

    def _skew_safe(x: pd.Series) -> float:
        if x.size == 0: return 0.0
        val = float(pd.Series(x).skew())
        return val if np.isfinite(val) else 0.0

    def _kurt_safe(x: pd.Series) -> float:
        if x.size == 0: return 0.0
        val = float(pd.Series(x).kurt())
        return val if np.isfinite(val) else 0.0

    def _benford_distance(amount_series: pd.Series) -> float:
        x = np.abs(amount_series.values)
        x = x[x > 0]
        if x.size == 0: return 0.0
        pow10 = np.power(10.0, np.floor(np.log10(x)))
        first_digit = (x // pow10).astype(int)
        first_digit = first_digit[(first_digit >= 1) & (first_digit <= 9)]
        if first_digit.size == 0: return 0.0
        counts = np.bincount(first_digit, minlength=10)[1:10].astype(float)
        p = counts / counts.sum()
        ben = np.log10(1 + 1/np.arange(1,10))
        return float(np.abs(p - ben).sum())

    def _ensure_idx_cat(df_like, _acct_dtype):
        df_like.index = df_like.index.astype(_acct_dtype)
        return df_like
    
    gc.collect()
    _log("複製資料 + 類別欄位對齊", verbose)
    df = df_txn.copy()

    # ---- 對齊 Categorical 類別（避免 from_acct/to_acct 比較時 categories 不同）----
    if isinstance(df['from_acct'].dtype, CategoricalDtype) or isinstance(df['to_acct'].dtype, CategoricalDtype):
        fa_cats = pd.Index(df['from_acct'].astype('category').cat.categories)
        ta_cats = pd.Index(df['to_acct'].astype('category').cat.categories)
        acct_union = fa_cats.union(ta_cats)
        _acct_dtype = CategoricalDtype(categories=acct_union, ordered=False)
        df['from_acct'] = df['from_acct'].astype(_acct_dtype)
        df['to_acct']   = df['to_acct'].astype(_acct_dtype)
    else:
        # 先都轉成字串
        df['from_acct'] = df['from_acct'].astype(str)
        df['to_acct']   = df['to_acct'].astype(str)
        # 用全部帳號建一個 category，後面才有 _acct_dtype 可以用
        acct_union = pd.Index(df['from_acct'].unique()).union(df['to_acct'].unique())
        _acct_dtype = CategoricalDtype(categories=acct_union, ordered=False)
        df['from_acct'] = df['from_acct'].astype(_acct_dtype)
        df['to_acct']   = df['to_acct'].astype(_acct_dtype)

    # is_self 對齊（含 Y/N/UNK）
    if 'is_self' in df.columns:
        df['is_self'] = df['is_self'].astype('category')
        need = pd.Index(['Y','N','UNK'])
        cats = pd.Index(df['is_self'].cat.categories)
        miss = list(need.difference(cats))
        if len(miss) > 0:
            df['is_self'] = df['is_self'].cat.add_categories(miss)

    # 數值欄位保險
    df['amount']   = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['date']     = pd.to_numeric(df['date'],   errors='coerce').fillna(0).astype('int64')
    df['time']     = pd.to_numeric(df['time'],   errors='coerce').fillna(0).astype('int64')
    df['time_abs'] = pd.to_numeric(df['time_abs'], errors='coerce').fillna(0).astype('int64')
        # ---- 建立帳戶視角長表（send/recv）----
    _log("建立長表（send/recv 視角）", verbose)
    with StepTimer("concat send/recv", verbose):
        send_long = (
            df[['from_acct','to_acct','date','time','time_abs','currency_type','amount','channel_type','is_self']]
            .rename(columns={
                'from_acct':'acct',
                'to_acct':'counterparty',
                'time':'sec_in_day',
                'time_abs':'abs_sec',
                'currency_type':'currency',
                'channel_type':'channel'
            })
            .assign(role='send')
        )
        recv_long = (
            df[['to_acct','from_acct','date','time','time_abs','currency_type','amount','channel_type','is_self']]
            .rename(columns={
                'to_acct':'acct',
                'from_acct':'counterparty',
                'time':'sec_in_day',
                'time_abs':'abs_sec',
                'currency_type':'currency',
                'channel_type':'channel'
            })
            .assign(role='recv')
        )
        long = pd.concat([send_long, recv_long], ignore_index=True)
        long['hour'] = (long['sec_in_day'] // 3600).astype('int64')

        # 這裡用 acct ∪ counterparty 建「真正的」全集，再覆蓋掉前面的 _acct_dtype
        acct_cats = long['acct'].astype('category').cat.categories.union(
            long['counterparty'].astype('category').cat.categories
        )
        _acct_dtype = CategoricalDtype(categories=acct_cats, ordered=False)

        long['acct'] = long['acct'].astype(_acct_dtype)
        long['counterparty'] = long['counterparty'].astype(_acct_dtype)

        del send_long, recv_long
        gc.collect()
    _log(f"長表 rows = {len(long):,}", verbose)

    # ---- 合併 event_date 並切割觀測期間 ----
    _log("合併 event_date 並切割觀測期間", verbose)
    with StepTimer("merge event_date & filter", verbose):
        alert_map = df_alert[['acct', 'event_date']].copy()
        # 這裡也用同一個 _acct_dtype
        alert_map['acct'] = alert_map['acct'].astype(_acct_dtype)
        alert_map['event_date'] = pd.to_numeric(alert_map['event_date'], errors='coerce').astype('Int64')

        long = long.merge(alert_map, on='acct', how='left')
        if include_event_day:
            keep = (long['event_date'].isna()) | (long['date'] <= long['event_date'])
        else:
            keep = (long['event_date'].isna()) | (long['date'] <  long['event_date'])
        long = long.loc[keep].drop(columns=['event_date'])
    _log(f"切割後長表 rows = {len(long):,}", verbose)
    gc.collect()
    # ---------------------------- 基礎統計（原有 + 強化） ----------------------------

    _log("自轉次數 self_txn_count (from is_self)", verbose)
    with StepTimer("self_txn_count (is_self)", verbose):
        # 只拿標成自轉的交易
        self_txn = df[df["is_self"] == "Y"].copy()

        # 兩個方向都要算：from_acct、to_acct
        part_from = (
            self_txn[["from_acct", "date"]]
            .rename(columns={"from_acct": "acct"})
        )
        part_to = (
            self_txn[["to_acct", "date"]]
            .rename(columns={"to_acct": "acct"})
        )

        self_all = pd.concat([part_from, part_to], ignore_index=True)

        # 跟 long / alert 一樣的 dtype
        self_all["acct"] = self_all["acct"].astype(_acct_dtype)

        # 合 event_date，跟其他特徵一樣的過濾邏輯
        self_all = self_all.merge(alert_map, on="acct", how="left")
        if include_event_day:
            self_all = self_all[
                self_all["event_date"].isna() | (self_all["date"] <= self_all["event_date"])
            ]
        else:
            self_all = self_all[
                self_all["event_date"].isna() | (self_all["date"] < self_all["event_date"])
            ]

        # 每個帳號真正的自轉筆數
        self_txn_per_acct = (
            self_all.groupby("acct", observed=True, sort=False)
                    .size()
                    .rename("self_txn_count")
        )
        del self_txn, part_from, part_to, self_all
        gc.collect()

    # send / recv 次數
    _log("send/recv 次數", verbose)
    with StepTimer("send/recv 計數", verbose):
        send_count = (
            long[long['role']=='send']
            .groupby('acct', observed=True, sort=False)
            .size().rename('send_count')
        )
        recv_count = (
            long[long['role']=='recv']
            .groupby('acct', observed=True, sort=False)
            .size().rename('recv_count')
        )

    # 每日統計與活躍日
    _log("每日統計 & 活躍日", verbose)
    with StepTimer("per_day_counts → per_day_agg", verbose):
        per_day_counts = (
            long.groupby(['acct','date'], observed=True, sort=False)
                .size().rename('cnt').reset_index()
        )
        per_day_agg = (
            per_day_counts.groupby('acct', observed=True, sort=False)['cnt']
            .agg(
                txn_per_day_mean='mean',
                txn_per_day_max='max',
                txn_per_day_min='min',
                txn_per_day_std=lambda s: float(np.std(s, ddof=0)),
            )
        )
        per_day_agg['txn_per_day_cv'] = (
            per_day_agg['txn_per_day_std'] / (per_day_agg['txn_per_day_mean'] + 1e-9)
        )
        per_day_agg = per_day_agg.astype({
            'txn_per_day_mean':'float32',
            'txn_per_day_max':'float32',
            'txn_per_day_min':'float32',
            'txn_per_day_std':'float32',
            'txn_per_day_cv':'float32',
        })
        active_days = (
            per_day_counts.groupby('acct', observed=True)['date']
                .nunique()
                .rename('active_days')
                .astype('int32')
        )
        del per_day_counts
        gc.collect()
    # 交易間隔統計（平均/最小/短間隔比例）
    _log("交易間隔統計", verbose)
    with StepTimer("txn intervals", verbose):
        interval_agg = (long.groupby('acct', observed=True, sort=False)['abs_sec']
                        .agg(txn_interval_mean=_mean_interval,
                                txn_interval_min=_min_interval,
                                prop_short_interval=lambda s: _prop_short_intervals(s, 300)))
        interval_agg = interval_agg.astype({'txn_interval_mean':'float32',
                                            'txn_interval_min':'float32',
                                            'prop_short_interval':'float32'})
    # 3 小時分段（8 段）
    _log("3 小時分段次數", verbose)
    with StepTimer("3 小時分段", verbose):
        hour_bin = (long['hour'] // 3).astype('int8')   # 0..7
        ct = pd.crosstab(long['acct'], hour_bin)
        ct = ct.rename(columns={i:f"txn_count_{3*i}_{3*(i+1)}" for i in range(8)}).astype('int32')
        bins3h = ct
        del ct, hour_bin; gc.collect()

        # 夜間比例 / 金額統計 / 時段熵
    _log("夜間比例 / 金額統計 / 時段熵", verbose)
    with StepTimer("夜間/金額/熵", verbose):
        night_ratio = ((long['hour'] < 6).groupby(long['acct'], observed=True)
                        .mean().rename('night_txn_ratio').astype('float32'))

        # 金額分布統計
        amt = long.groupby('acct', observed=True)['amount'].agg(
            amount_max='max',
            amount_mean='mean',
            amount_std=lambda s: float(s.std(ddof=0)),
            amount_q25=lambda s: float(s.quantile(0.25)),
            amount_q75=lambda s: float(s.quantile(0.75)),
            amount_sum='sum',
            amount_median='median',
            amount_skew=_skew_safe,
            amount_kurt=_kurt_safe
        )
        amt['amount_iqr'] = amt['amount_q75'] - amt['amount_q25']
        amt['amount_cv']  = amt['amount_std'] / (amt['amount_mean'] + 1e-9)
        amt_agg = (amt.drop(columns=['amount_q25','amount_q75'])
                        .astype({'amount_max':'float32','amount_mean':'float32','amount_std':'float32',
                                'amount_sum':'float32','amount_median':'float32',
                                'amount_iqr':'float32','amount_cv':'float32',
                                'amount_skew':'float32','amount_kurt':'float32'}))

        # 時段（小時）熵
        hour_entropy = (long.groupby('acct', observed=True)['hour']
                            .agg(txn_hour_entropy=lambda s: _safe_entropy_from_hours(s))
                            .astype('float32'))

    # 幣別/通路/自轉 多樣性 + OHE 次數
    _log("幣別/通路/自轉 多樣性 + 次數", verbose)
    with StepTimer("幣別/通路/自轉 多樣性 + crosstab", verbose):
        # 幣別 / 通路 unique 數
        currency_unique = (
            long.groupby('acct', observed=True)['currency']
                .nunique().rename('currency_unique').astype('int16')
        )
        channel_unique  = (
            long.groupby('acct', observed=True)['channel']
                .nunique().rename('channel_unique').astype('int16')
        )

        # 幣別次數 OHE
        cc = pd.crosstab(long['acct'], long['currency'])
        cc.columns = [f"currency_count__{str(c)}" for c in cc.columns]
        cc = cc.astype('int32')
        currency_counts = cc

        # 通路次數 OHE
        ch = pd.crosstab(long['acct'], long['channel'])
        ch.columns = [f"channel_count__{str(c)}" for c in ch.columns]
        ch = ch.astype('int32')
        channel_counts = ch

        # 自轉次數 OHE（Y / N / UNK 都留著，UNK 也算一類）
        # long 這時候已經有 is_self 了
        self_ct = pd.crosstab(long['acct'], long['is_self'])
        # 欄名長一點，避免跟別的撞
        self_ct.columns = [f"is_self_count__{str(c)}" for c in self_ct.columns]
        self_ct = self_ct.astype('int32')
        self_txn_ohe = self_ct


        # 自轉比例
        self_txn_ratio = self_txn_ohe.div(
            self_txn_ohe.sum(axis=1).replace(0, np.nan),
            axis=0
        ).fillna(0.0)
        self_txn_ratio.columns = [c.replace("count__", "ratio__") for c in self_txn_ratio.columns]

        del cc, ch, self_ct
        gc.collect()
    # ---------------------------- 爆發/節奏特徵（5/15/30/60 分鐘） ----------------------------
    _log("爆發/節奏特徵（5/15/30/60min）", verbose)
    with StepTimer("burst & fano (multi-scale)", verbose):
        # 比賽顆粒是 5 分鐘，所以往上聚 15/30/60 分鐘
        bucket_specs = {
            300:  "5min",    # 5 分鐘
            900:  "15min",   # 3 個 5 分鐘
            1800: "30min",   # 6 個 5 分鐘
            3600: "60min",   # 12 個 5 分鐘
        }

        burst_feat_list = []

        for sec_per_bucket, label in bucket_specs.items():
            col_bucket = f"_bucket_{label}"
            # 以秒數除以粒度，得到這筆交易落在哪個時間桶
            long[col_bucket] = (long['abs_sec'] // sec_per_bucket).astype('int64')

            # 每個帳戶在這個粒度下，每個時間桶的交易筆數
            tmp = (
                long.groupby(['acct', col_bucket], observed=True)
                    .size().rename('cnt').reset_index()
            )

            # 做這個粒度下的爆發統計
            tmp_agg = (
                tmp.groupby('acct', observed=True)['cnt']
                    .agg(**{
                        f"burst_{label}_max":  'max',
                        f"burst_{label}_mean": 'mean',
                        f"burst_{label}_p95":  lambda s: float(np.quantile(s, 0.95)),
                    })
                    .astype({
                        f"burst_{label}_max":  'float32',
                        f"burst_{label}_mean": 'float32',
                        f"burst_{label}_p95":  'float32',
                    })
            )

            burst_feat_list.append(tmp_agg)
            long.drop(columns=[col_bucket], inplace=True)  # 減少記憶體占用

            del tmp, tmp_agg
            gc.collect()

        # 把 5/15/30/60min 四組爆發特徵合併
        burst_feats = pd.concat(burst_feat_list, axis=1)

        # 每日計數的 Fano factor（保留原本的日級爆發感）
        daily_cnt = (
            long.groupby(['acct', 'date'], observed=True)
                .size().rename('d_cnt').reset_index()
        )

        def _fano(s: pd.Series) -> float:
            m = float(s.mean())
            v = float(np.var(s, ddof=0))
            return float(v / (m + 1e-9))

        fano = (
            daily_cnt.groupby('acct', observed=True)['d_cnt']
                .agg(txn_per_day_fano=_fano)
                .astype('float32')
        )

        del daily_cnt, burst_feat_list
        gc.collect()

    # ---------------------------- 對手/關係集中度 & 熵 ----------------------------
    _log("對手/關係集中度 & 熵", verbose)
    with StepTimer("counterparty diversity", verbose):
        # 對手唯一數（收/付/總）
        cp_total_unique = long.groupby('acct', observed=True)['counterparty'].nunique().rename('cp_total_unique').astype('int32')
        cp_send_unique  = long[long['role']=='send'].groupby('acct', observed=True)['counterparty'].nunique().rename('cp_send_unique').astype('int32')
        cp_recv_unique  = long[long['role']=='recv'].groupby('acct', observed=True)['counterparty'].nunique().rename('cp_recv_unique').astype('int32')

        # 對手分布的熵（以出入合併的對手計數）
        cp_entropy = (long.groupby(['acct','counterparty'], observed=True).size().rename('cnt').reset_index()
                            .groupby('acct', observed=True)['cnt']
                            .apply(lambda s: float(_safe_entropy_from_counts(s.values.astype(float)))))
        cp_entropy = cp_entropy.rename('cp_entropy').astype('float32')

        # 對手集中度（HHI / Top-k by count）
        cp_counts = (
            long.groupby(['acct','counterparty'], observed=True)
                .size().rename('cnt').reset_index()
        )

        def _cp_conc_from_counts(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            hhi  = _herfindahl_from_counts(vals)
            top1 = _topk_share_from_counts(vals, 1)
            top3 = _topk_share_from_counts(vals, 3)
            return pd.Series({
                'cp_hhi_count': hhi,
                'cp_top1_count_share': top1,
                'cp_top3_count_share': top3,
            })

        cp_conc = cp_counts.groupby('acct', observed=True)['cnt'].apply(_cp_conc_from_counts)

        # 兼容不同 pandas：可能得到 Series(MultiIndex)；統一攤平成 DataFrame
        if isinstance(cp_conc, pd.Series):
            cp_conc = cp_conc.unstack()

        # 若某版型少欄位（極端資料），補齊再轉型
        for col in ['cp_hhi_count', 'cp_top1_count_share', 'cp_top3_count_share']:
            if col not in cp_conc.columns:
                cp_conc[col] = 0.0

        cp_conc = cp_conc.astype({
            'cp_hhi_count': 'float32',
            'cp_top1_count_share': 'float32',
            'cp_top3_count_share': 'float32'
        })
        cp_conc.index.name = 'acct'  # join 前保險

        # 以金額聚合的集中度（Top-k share by amount）
        cp_amt = (
            long.groupby(['acct','counterparty'], observed=True)['amount']
                .sum().rename('amt').reset_index()
        )

        def _cp_conc_from_amounts(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float)
            denom = vals.sum() + 1e-9
            vsort = np.sort(vals)[::-1]
            top1  = float(vsort[:1].sum() / denom)
            top3  = float(vsort[:3].sum() / denom)
            return pd.Series({
                'cp_top1_amount_share': top1,
                'cp_top3_amount_share': top3,
            })

        cp_amt_conc = cp_amt.groupby('acct', observed=True)['amt'].apply(_cp_conc_from_amounts)

        if isinstance(cp_amt_conc, pd.Series):
            cp_amt_conc = cp_amt_conc.unstack()

        for col in ['cp_top1_amount_share', 'cp_top3_amount_share']:
            if col not in cp_amt_conc.columns:
                cp_amt_conc[col] = 0.0

        cp_amt_conc = cp_amt_conc.astype({
            'cp_top1_amount_share': 'float32',
            'cp_top3_amount_share': 'float32'
        })
        cp_amt_conc.index.name = 'acct'

        # 互惠比例：同一對手是否雙向有交易
        # 標記每個帳戶的「雙向對手」數量
        bi_df = (long.groupby(['acct','counterparty','role'], observed=True)
                        .size().unstack(fill_value=0))
        bi_df['has_send'] = (bi_df.get('send', 0) > 0).astype(int)
        bi_df['has_recv'] = (bi_df.get('recv', 0) > 0).astype(int)
        bi_df['is_bidir'] = ((bi_df['has_send'] + bi_df['has_recv']) == 2).astype(int)
        bidir_rate = (bi_df.groupby('acct', observed=True)['is_bidir'].mean()
                            .rename('bidir_counterparty_ratio').astype('float32'))
        del  bi_df
        gc.collect()
        
    # ---------------------------- 金額分布形狀/臨界/尾數 ----------------------------
    _log("金額分布形狀/臨界/尾數", verbose)
    with StepTimer("amount shape stats", verbose):
        # Gini（單帳戶的交易金額不均衡度）
        gini = (long.groupby('acct', observed=True)['amount']
                    .apply(lambda s: float(_gini_from_values(s.values)))
                    .rename('amount_gini').astype('float32'))

        # ---- 修正重點：尾端比例（>q90/q95/q99）避免 MultiIndex ----
        def _tail_props_series(s: pd.Series) -> pd.Series:
            # 回傳一個以欄名為 key 的 Series：{'amount_prop_gt_q90':..., ...}
            return pd.Series(_amount_tail_props(s))

        tp = long.groupby('acct', observed=True)['amount'].apply(_tail_props_series)

        # 有些 pandas 版本會回 Series 且 index 變成 (acct, 指標名)；統一攤平成寬表
        if isinstance(tp, pd.Series):
            tail_props = tp.unstack(level=-1)
        else:
            # 已是 DataFrame（index=acct, columns=指標名）
            tail_props = tp

        tail_props.columns.name = None
        #（可選）確保欄位順序固定
        for col in ['amount_prop_gt_q90', 'amount_prop_gt_q95', 'amount_prop_gt_q99']:
            if col not in tail_props.columns:
                tail_props[col] = 0.0
        tail_props = tail_props[['amount_prop_gt_q90', 'amount_prop_gt_q95', 'amount_prop_gt_q99']].astype('float32')

        # 接近千元整比例（可改 base=1000, tol=10）
        near_round = (long.groupby('acct', observed=True)['amount']
                        .apply(lambda s: float(_prop_near_round(s, base=1000.0, tol=10.0)))
                        .rename('prop_near_1k').astype('float32'))

        # 金額尾數熵（個位數分布）
        last_digit_ent = (long.groupby('acct', observed=True)['amount']
                            .apply(_last_digit_entropy)
                            .rename('amount_last_digit_entropy').astype('float32'))
        del tp
        gc.collect()
    # ---------------------------- 通路/幣別 熵與金額交互 ----------------------------
    _log("通路/幣別 熵與金額交互", verbose)
    with StepTimer("channel/currency mix", verbose):
        channel_entropy = (long.groupby('acct', observed=True)['channel']
                                .apply(_safe_entropy_from_series)
                                .rename('channel_entropy').astype('float32'))
        currency_entropy = (long.groupby('acct', observed=True)['currency']
                                    .apply(_safe_entropy_from_series)
                                    .rename('currency_entropy').astype('float32'))

        # 各通路金額的平均/中位數（只取 top 使用量的通路可再精簡，這裡全做）
        ch_amt_mean = long.pivot_table(index='acct', columns='channel', values='amount', aggfunc='mean', observed=True)
        ch_amt_mean.columns = [f'ch_amt_mean__{str(c)}' for c in ch_amt_mean.columns]
        ch_amt_median = long.pivot_table(index='acct', columns='channel', values='amount', aggfunc='median', observed=True)
        ch_amt_median.columns = [f'ch_amt_median__{str(c)}' for c in ch_amt_median.columns]

        # 幣別金額占比（雖已換 TWD，但來源幣別仍可能透露模式）
        cur_amt_sum = long.pivot_table(index='acct', columns='currency', values='amount', aggfunc='sum', observed=True)
        cur_amt_sum = cur_amt_sum.fillna(0.0)
        cur_amt_prop = cur_amt_sum.div(cur_amt_sum.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        cur_amt_prop.columns = [f'currency_amount_prop__{str(c)}' for c in cur_amt_prop.columns]
        
    # ---------------------------- 新對手比例（近7/14/30天） ----------------------------
    with StepTimer("new counterparty ratio (7/14/30d)", verbose):
        # 每個帳戶與對手的首次出現日期
        first_seen = (
            long.groupby(['acct', 'counterparty'], observed=True)['date']
                .min().rename('cp_first_date').reset_index()
        )

        # 合回長表
        long3 = long.merge(first_seen, on=['acct', 'counterparty'], how='left')

        # 每個帳戶自己的最後觀測日
        acct_last_date = (
            long3.groupby('acct', observed=True)['date']
                .max().rename('acct_last_date')
        )
        long4 = long3.merge(acct_last_date, on='acct', how='left')

        # 定義視窗
        win_list = [7, 14, 30]
        out_list = []

        for win in win_list:
            in_last = long4['date'] >= (long4['acct_last_date'] - win + 1)
            cp_new = long4['cp_first_date'] >= (long4['acct_last_date'] - win + 1)

            ratio = (cp_new & in_last).groupby(long4['acct'], observed=True).mean()
            ratio = ratio.rename(f'new_counterparty_ratio_{win}d').astype('float32')
            out_list.append(ratio)

        # 合併三個時間窗
        new_cp_ratios = pd.concat(out_list, axis=1)

        del long3, long4, first_seen, acct_last_date, out_list
        gc.collect()
    # ---------------------------- Session 特徵（多間隔） ----------------------------
    _log("Session 特徵（5/15/30/60 分鐘間隔）", verbose)
    with StepTimer("session features (multi-gap)", verbose):
        # 先把 long 排好，後面每個門檻都可以重用
        long = long.sort_values(['acct', 'abs_sec'])
        session_gap_specs = {
            300:  "5min",    # 5 分鐘
            900:  "15min",   # 15 分鐘
            1800: "30min",   # 30 分鐘
            3600: "60min",   # 60 分鐘
        }

        sess_feat_list = []

        for thr_sec, label in session_gap_specs.items():
            g = long.groupby('acct', observed=True)

            # 與前一筆時間差 > 門檻 → 新 session
            is_new = g['abs_sec'].diff().gt(thr_sec) | g['abs_sec'].diff().isna()

            # 依帳戶累計 session id
            sess_id = is_new.groupby(long['acct'], observed=True).cumsum().astype('int32')

            long_sess = long.assign(_sess=sess_id)

            # 每個帳戶在這個門檻下的所有 session 長度
            sess_len = (
                long_sess.groupby(['acct', '_sess'], observed=True)
                    .size().rename('sess_len')
            )

            sess_agg_one = (
                sess_len.groupby('acct', observed=True)
                    .agg(
                        **{
                            f'sess_{label}_max':  'max',
                            f'sess_{label}_mean': 'mean',
                            f'sess_{label}_p95':  lambda s: float(np.quantile(s, 0.95)),
                        }
                    )
                    .astype({
                        f'sess_{label}_max':  'float32',
                        f'sess_{label}_mean': 'float32',
                        f'sess_{label}_p95':  'float32',
                    })
            )

            sess_feat_list.append(sess_agg_one)

            del g, is_new, sess_id, long_sess, sess_len, sess_agg_one
            gc.collect()

        # 合併四個門檻的 session 特徵
        sess_feats = pd.concat(sess_feat_list, axis=1)

        del sess_feat_list
        gc.collect()
    # ---------------------------- 入後即出（Deposit→Quick Out）/ 出後即入（Quick In） ----------------------------
    _log("入後即出（Deposit→Quick Out）/ 出後即入（Quick In）", verbose)
    with StepTimer("deposit→quick-out features (1-pass, multi-window)", verbose):
        # 只抓需要的欄位並排序（這個排序花一下是值得的，後面就可以純 numpy 走）
        long_sr = (
            long[['acct', 'role', 'abs_sec', 'amount']]
            .copy()
            .sort_values(['acct', 'abs_sec'], kind='mergesort')
        )
        long_sr['acct'] = long_sr['acct'].astype(_acct_dtype)
        long_sr['abs_sec'] = pd.to_numeric(long_sr['abs_sec'], errors='coerce')
        long_sr = long_sr.dropna(subset=['abs_sec'])

        # 轉成 numpy，加快迴圈
        accts  = long_sr['acct'].to_numpy()
        roles  = long_sr['role'].to_numpy()
        times  = long_sr['abs_sec'].to_numpy(dtype=np.int64)
        amounts = long_sr['amount'].to_numpy(dtype=np.float64)

        # 我們要的時間窗（秒）→ 欄名
        quick_specs = {
            300:  "5min",
            900:  "15min",
            1800: "30min",
            3600: "60min",
        }

        results = {}

        # 找出每個帳戶在這四個 array 裡的起訖 index
        # ex: acct_boundaries = [0, 120, 240, ...]
        change_idx = np.flatnonzero(accts[1:] != accts[:-1]) + 1
        boundaries = np.concatenate(([0], change_idx, [accts.size]))

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end   = boundaries[i+1]

            acct_id = accts[start]
            t = times[start:end]
            a = amounts[start:end]
            r = roles[start:end]

            # 拆 recv / send
            recv_mask = (r == 'recv')
            send_mask = (r == 'send')

            rt = t[recv_mask]; ra = a[recv_mask]
            st = t[send_mask]; sa = a[send_mask]

            # 沒有其中一種，就全部 0
            if rt.size == 0 or st.size == 0:
                res = {}
                for _, label in quick_specs.items():
                    res[f'quick_out_rate_{label}'] = 0.0
                    res[f'quick_out_large_ratio_rate_{label}'] = 0.0
                results[acct_id] = res
                continue

            # 對 recv 找下一筆 send
            idx = np.searchsorted(st, rt, side='right')
            valid = (idx < st.size)

            dt = np.full(rt.shape, np.nan, dtype=np.float64)
            ratio = np.full(rt.shape, np.nan, dtype=np.float64)
            dt[valid] = st[idx[valid]] - rt[valid]
            ratio[valid] = sa[idx[valid]] / (ra[valid] + 1e-9)

            res = {}
            # 在同一組 dt / ratio 上面滾 4 個窗
            for thr_sec, label in quick_specs.items():
                quick_mask = (dt <= thr_sec)
                if np.all(~np.isfinite(dt)):
                    q_rate = 0.0
                    q_large = 0.0
                else:
                    q_rate = float(np.nanmean(quick_mask))
                    q_large = float(np.nanmean(quick_mask & (ratio >= 0.8)))
                res[f'quick_out_rate_{label}'] = q_rate
                res[f'quick_out_large_ratio_rate_{label}'] = q_large

            results[acct_id] = res

        # dict → DataFrame
        quick_feats = pd.DataFrame.from_dict(results, orient='index')
        quick_feats.index.name = 'acct'
        quick_feats = quick_feats.astype('float32')

        del long_sr, accts, roles, times, amounts, results, boundaries
        gc.collect()


    _log("Benford's Law 金額首位分布偏離度", verbose)
    with StepTimer("benford", verbose):
        benford_l1 = long.groupby('acct', observed=True)['amount'] \
                        .apply(_benford_distance).rename('benford_l1').astype('float32')



    _log("對手金額集中度 HHI", verbose)
    with StepTimer("cp amount HHI", verbose):
        cp_amt_sum = long.groupby(['acct','counterparty'], observed=True)['amount'].sum().rename('amt').reset_index()
        amt_hhi = (cp_amt_sum.groupby('acct', observed=True)['amt']
                        .apply(lambda s: _herfindahl_from_counts(s.to_numpy(dtype=float)))
                        .rename('cp_hhi_amount').astype('float32'))

    _log("通路/幣別 交易比例", verbose)
    with StepTimer("channel/currency proportions", verbose):
        ch_cnt = pd.crosstab(long['acct'], long['channel']).astype('float32')
        ch_prop = ch_cnt.div(ch_cnt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        ch_prop.columns = [f'ch_txn_prop__{str(c)}' for c in ch_prop.columns]

        cur_cnt = pd.crosstab(long['acct'], long['currency']).astype('float32')
        cur_prop = cur_cnt.div(cur_cnt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        cur_prop.columns = [f'currency_txn_prop__{str(c)}' for c in cur_prop.columns]

    # ---------------------------- 金額為權重的圖特徵 ----------------------------
    _log("金額為權重的圖特徵（networkit: PR/Component/Degree + Bet/Katz/Core + hop2/3）", verbose)
    with StepTimer("graph features (networkit)", verbose):
        # 基本檢查
        required_cols = {"acct", "counterparty", "amount"}
        missing = required_cols - set(long.columns)
        if missing:
            raise ValueError(f"long 缺少必要欄位: {missing}")

        # 1) 分出「有向要用的資料」跟「無向要用的資料」
        if "role" in long.columns:
            # 有 role → 有向圖只用送出的那一半，避免一進一出成對
            long_dir = long[long["role"] == "send"]
        else:
            # 沒 role → 就用原本全部
            long_dir = long

        long_undir = long  # 無向圖還是用全部

        # 2) 各自做聚合
        # 有向：只聚 send 的
        edges_dir = (
            long_dir.groupby(['acct', 'counterparty'], observed=True)['amount']
                .sum()
                .reset_index()
        )
        edges_dir = edges_dir[(edges_dir['acct'] != edges_dir['counterparty']) & (edges_dir['amount'] > 0)]

        # 無向：聚全部的
        edges_ud = (
            long_undir.groupby(['acct', 'counterparty'], observed=True)['amount']
                .sum()
                .reset_index()
        )
        edges_ud = edges_ud[(edges_ud['acct'] != edges_ud['counterparty']) & (edges_ud['amount'] > 0)]

        # 3) 節點全集
        acct_nodes = pd.Index(long['acct'].unique())
        cp_nodes   = pd.Index(long['counterparty'].unique())
        all_nodes  = acct_nodes.union(cp_nodes)  # CategoricalIndex

        node_labels = all_nodes.astype(str).tolist()
        id_of = {lab: i for i, lab in enumerate(node_labels)}
        n = len(node_labels)

        # 4) 建圖
        G_dir   = nk.graph.Graph(n, directed=True,  weighted=True)
        G_undir = nk.graph.Graph(n, directed=False, weighted=True)

        # 有向圖資料
        w_arr_dir  = np.log1p(edges_dir['amount'].to_numpy())
        s_acct_dir = edges_dir['acct'].astype(str).to_numpy()
        s_cp_dir   = edges_dir['counterparty'].astype(str).to_numpy()

        # 無向圖資料
        w_arr_ud  = np.log1p(edges_ud['amount'].to_numpy())
        s_acct_ud = edges_ud['acct'].astype(str).to_numpy()
        s_cp_ud   = edges_ud['counterparty'].astype(str).to_numpy()

        # in / out 的累計
        weighted_in  = defaultdict(float)
        weighted_out = defaultdict(float)

        # 4a) 建有向圖 + 累計出入權重
        for u_lab, v_lab, w in tqdm(zip(s_acct_dir, s_cp_dir, w_arr_dir),
                                    total=len(s_acct_dir), desc="  Build G_dir"):
            u_id = id_of[u_lab]; v_id = id_of[v_lab]
            G_dir.addEdge(u_id, v_id, w)
            weighted_out[u_id] += w
            weighted_in[v_id]  += w

        # 4b) 建無向圖（加總權重）
        for u_lab, v_lab, w in tqdm(zip(s_acct_ud, s_cp_ud, w_arr_ud),
                                    total=len(s_acct_ud), desc="  Build G_undir"):
            u_id = id_of[u_lab]; v_id = id_of[v_lab]
            if not G_undir.hasEdge(u_id, v_id):
                G_undir.addEdge(u_id, v_id, w)
            else:
                old_w = G_undir.weight(u_id, v_id)
                G_undir.setWeight(u_id, v_id, old_w + w)

        # 5) PageRank（有向、加權）
        _log("  PageRank...", verbose)
        pr = nk.centrality.PageRank(G_dir, damp=0.85)
        pr.run()
        pr_vals = pr.scores()

        # 6) Connected Components（無向）
        _log("  Connected components...", verbose)
        comp = nk.components.ConnectedComponents(G_undir); comp.run()
        comp_ids = comp.getComponents()
        comp_size = np.ones(n, dtype=np.int32)
        for comp_nodes in comp_ids:
            sz = len(comp_nodes)
            for nid in comp_nodes:
                comp_size[nid] = sz
        comp_label = -1 * np.ones(n, dtype=np.int32)
        for cid, nodes in enumerate(comp_ids):
            for nid in nodes:
                comp_label[nid] = cid

        # 7) 度數（有向；含加權）
        _log("  Degrees...", verbose)
        in_deg   = np.fromiter((G_dir.degreeIn(u)  for u in range(n)), dtype=np.int32,   count=n)
        out_deg  = np.fromiter((G_dir.degreeOut(u) for u in range(n)), dtype=np.int32,   count=n)

        win_deg  = np.fromiter((weighted_in[u]  for u in range(n)), dtype=np.float32, count=n)
        wout_deg = np.fromiter((weighted_out[u] for u in range(n)), dtype=np.float32, count=n)

    # 可調整 epsilon/delta（epsilon 越小越精準但越慢）
    bet_vals = None
    try:
        # G_undir — 要計算的圖（NetworKit Graph，你這裡是無向加權圖）。
        # epsilon = 0.02 — 加法誤差上限。演算法保證每個節點的估計介數中心性，與真值的差距 ≤ ε 的機率至少為 1−δ。ε 越小 → 需要的抽樣數越多、越慢，但更準。
        # delta = 0.1 — 失敗機率（風險容忍度）。結果不達到上述誤差界的機率 ≤ δ。δ 越小 → 需要的抽樣數越多。
        # universalConstant = 1.0 — 理論常數，控制抽樣數的比例係數；總抽樣數約與 universalConstant / epsilon^2 成正比（再受 δ 影響）。調大它會線性增加抽樣數（更慢、略更準）。
        # 新版/常見接口：ApproxBetweenness( G, epsilon, delta, universalConstant )
        bet = nk.centrality.ApproxBetweenness(G_undir, 0.03, 0.1, 1.0)
        _log("  Betweenness (approx)...", verbose)
        bet.run()
        bet_vals = bet.scores()
        
    except Exception:
        # 舊版/另一支線：EstimateBetweenness( G, nSamples, normalized, parallel_flag )
        # 注意：只能用「位置參數」，沒有 k= / normalized= 這類關鍵字
        # G_undir — 要計算的圖（networkit.Graph；你這裡是無向、加權圖）。
        # 256 — nSamples：抽樣起點的數量。樣本越多近似越準，但時間線性變慢。
        # True — normalized：是否把介數中心性正規化到 [0, 1]。
        # False — parallel_flag：是否啟用平行版本（會用更多記憶體；關閉=單執行緒）。
        bet = nk.centrality.EstimateBetweenness(G_undir, 256, True, False)
        _log("  Betweenness (estimate)...", verbose)
        bet.run()
        bet_vals = bet.scores()
        
    # Katz（有向）
    _log("  Katz...", verbose)
    alpha = 1.5e-3  # 初始 alpha 值
    for _ in range(3):  # 最多退火 3 次
        try:
            katz = nk.centrality.KatzCentrality(G_dir, alpha=alpha, beta=1.0)
            katz.run()
            katz_vals = katz.scores()
            break
        except Exception:
            alpha *= 0.8
    else:
        # 萬一還是失敗就給 0
        katz_vals = np.zeros(n, dtype=np.float32)

    # Core number（k-core；無向）
    _log("Core number... ")
    core = nk.centrality.CoreDecomposition(
            G_undir,
            normalized=False  # 你原本就是要拿真正的 k-core 整數，這裡就設 False
        )
    core.run()
    # 每個節點的 core number
    core_vals = np.array(core.scores(), dtype=np.int32)

    # 2-hop 3-hop（無向，忽略權重；以鄰接表展開）
    _log("  2-hop 3-hop neighborhood size...", verbose)
    hop2_vals = np.zeros(n, dtype=np.int32)
    hop3_vals = np.zeros(n, dtype=np.int32)
    for u in tqdm(range(n)):
        # 1-hop
        nbr1 = list(G_undir.iterNeighbors(u))
        nbr1_set = set(nbr1)
        # 2-hop
        nbr2_set = set()
        for x in nbr1:
            nx1 = list(G_undir.iterNeighbors(x))
            nbr2_set.update(nx1)
        hop2_nodes = {u} | nbr1_set | nbr2_set
        hop2_vals[u] = len(hop2_nodes)
        # 3-hop
        nbr3_set = set()
        for x in nbr2_set:
            nx2 = list(G_undir.iterNeighbors(x))
            nbr3_set.update(nx2)
        hop3_nodes = hop2_nodes | nbr3_set
        hop3_vals[u] = len(hop3_nodes)
    # 轉成 float32
    pr_f   = np.asarray(pr_vals,  dtype=np.float32)
    bet_f  = np.asarray(bet_vals, dtype=np.float32)
    katz_f = np.asarray(katz_vals, dtype=np.float32)
    core_f = np.asarray(core_vals, dtype=np.float32)  # 原本是 int，這裡轉 float 方便計算

    # 連通元件數；避免除0
    K   = int(comp_label.max()) + 1
    cnt = np.bincount(comp_label, minlength=K).astype(np.float32)
    cnt = np.clip(cnt, 1.0, None)

    # 對某一個向量做群組 z-score（純 numpy、逐向量展開；不做 log）
    def _z_by_comp(x):
        s1  = np.bincount(comp_label, weights=x,      minlength=K).astype(np.float32)
        s2  = np.bincount(comp_label, weights=x * x,  minlength=K).astype(np.float32)
        mu  = s1 / cnt
        var = s2 / cnt - mu * mu
        var[var < 0.0] = 0.0
        sd  = np.sqrt(var)
        sd[sd == 0.0] = 1.0
        return (x - mu[comp_label]) / sd[comp_label]

    pr_f   = _z_by_comp(pr_f)
    bet_f  = _z_by_comp(bet_f)
    katz_f = _z_by_comp(katz_f)
    core_f = _z_by_comp(core_f)

    # 組回 DataFrame
    graph_feat = pd.DataFrame({
        'pr': pr_f,
        'component_size': comp_size,
        'out_deg': out_deg,
        'in_deg': in_deg,
        'wout_deg': wout_deg,
        'win_deg': win_deg,
        'betweenness_approx': bet_f,
        'katz': katz_f,
        'ud_core_number': core_f,
        'ud_hop2_size': hop2_vals,
        'ud_hop3_size': hop3_vals,
    }, index=pd.Index(node_labels, name='acct'))

    # 換回 _acct_dtype
    graph_feat.index = graph_feat.index.astype(_acct_dtype)
    graph_feat = graph_feat.reindex(acct_nodes, fill_value=0)

    graph_feat['component_size'] = graph_feat['component_size'].astype('int32')
    graph_feat['out_deg']        = graph_feat['out_deg'].astype('int32')
    graph_feat['in_deg']         = graph_feat['in_deg'].astype('int32')
    graph_feat['ud_core_number'] = graph_feat['ud_core_number'].astype('float32')
    graph_feat['ud_hop2_size']   = graph_feat['ud_hop2_size'].astype('int32')
    graph_feat['katz']         = graph_feat['katz'].astype('float32')
    graph_feat['ud_hop3_size']   = graph_feat['ud_hop3_size'].astype('int32')
    for c in ['pr','wout_deg','win_deg','betweenness_approx','katz']:
        graph_feat[c] = graph_feat[c].astype('float32')

    del w_arr_dir, s_acct_dir, s_cp_dir
    del w_arr_ud,  s_acct_ud,  s_cp_ud
    del G_dir, G_undir, edges_dir, edges_ud, weighted_in, weighted_out
    del pr, comp, bet, core, katz, K, cnt
    del pr_f, bet_f, katz_f, core_f, hop2_vals, hop3_vals, comp_label, comp_size, in_deg, out_deg, wout_deg, win_deg
    gc.collect()
    # ---------------------------- 帳戶層彙整（基礎表 base） ----------------------------
    _log("合併帳戶層基礎特徵", verbose)
    with StepTimer("build base", verbose):
        base = pd.DataFrame(index=long['acct'].astype(_acct_dtype).unique())
        base.index = base.index.astype(_acct_dtype)  # 強制一致
        base.index.name = 'acct'

        for _df in [bins3h, night_ratio, amt_agg, hour_entropy, currency_unique, channel_unique,
            currency_counts, channel_counts,self_txn_ohe, self_txn_ratio, burst_feats, fano, cp_total_unique, cp_send_unique,
            cp_recv_unique, cp_entropy, cp_conc, cp_amt_conc, bidir_rate, gini, tail_props,
            near_round, last_digit_ent, channel_entropy, currency_entropy, ch_amt_mean,
            ch_amt_median, cur_amt_prop, new_cp_ratios, benford_l1, amt_hhi, ch_prop, cur_prop,
            sess_feats, quick_feats, graph_feat]:
            _ensure_idx_cat(_df, _acct_dtype)


        base = base.join(send_count, how='left').join(recv_count, how='left')
        base[['send_count','recv_count']] = base[['send_count','recv_count']].fillna(0).astype('int32')
        base['total_txn_count'] = (base['send_count'] + base['recv_count']).astype('int32')

        base = base.join(active_days, how='left').fillna({'active_days':0}).astype({'active_days':'int32'})
        base = base.join(per_day_agg, how='left').fillna({
            'txn_per_day_mean':0.0,'txn_per_day_max':0.0,'txn_per_day_min':0.0,
            'txn_per_day_std':0.0,'txn_per_day_cv':0.0
        })

        base = base.join(self_txn_per_acct, how='left').fillna({'self_txn_count':0}).astype({'self_txn_count':'int32'})
        base['self_txn_ratio'] = (base['self_txn_count'] / base['total_txn_count'].clip(lower=1)).astype('float32')
        base['send_recv_cnt_ratio'] = ((base['send_count'] + 1) / (base['recv_count'] + 1)).astype('float32')

        base = base.join(interval_agg, how='left').fillna({'txn_interval_mean':0.0,'txn_interval_min':0.0,'prop_short_interval':0.0})

        base = (base
            .join(bins3h,        how='left').fillna(0)
            .join(night_ratio,   how='left').fillna({'night_txn_ratio':0.0})
            .join(amt_agg,       how='left')
            .join(hour_entropy,  how='left').fillna({'txn_hour_entropy':0.0})
            .join(currency_unique, how='left').fillna({'currency_unique':0})
            .join(channel_unique,  how='left').fillna({'channel_unique':0})
            .join(currency_counts, how='left').fillna(0)
            .join(channel_counts,  how='left').fillna(0)
            .join(self_txn_ohe,    how='left').fillna(0)         # is_self_count__Y / __N / __UNK
            .join(self_txn_ratio,  how='left').fillna(0.0)       # is_self_ratio__Y / ...
        )

        # 爆發/節奏
        base = (base
            .join(burst_feats, how='left').fillna(0.0)
            .join(fano,       how='left').fillna({'txn_per_day_fano': 0.0})
        )

        # 對手/關係
        base = (base
            .join(cp_total_unique, how='left').fillna({'cp_total_unique':0})
            .join(cp_send_unique,  how='left').fillna({'cp_send_unique':0})
            .join(cp_recv_unique,  how='left').fillna({'cp_recv_unique':0})
            .join(cp_entropy,      how='left').fillna({'cp_entropy':0.0})
            .join(cp_conc,         how='left').fillna({'cp_hhi_count':0.0,'cp_top1_count_share':0.0,'cp_top3_count_share':0.0})
            .join(cp_amt_conc,     how='left').fillna({'cp_top1_amount_share':0.0,'cp_top3_amount_share':0.0})
            .join(bidir_rate,      how='left').fillna({'bidir_counterparty_ratio':0.0})
        )

        # 金額形狀/臨界/尾數
        base = (base
            .join(gini,        how='left').fillna({'amount_gini':0.0})
            .join(tail_props,  how='left').fillna({'amount_prop_gt_q90':0.0,'amount_prop_gt_q95':0.0,'amount_prop_gt_q99':0.0})
            .join(near_round,  how='left').fillna({'prop_near_1k':0.0})
            .join(last_digit_ent, how='left').fillna({'amount_last_digit_entropy':0.0})
        )

        # 通路/幣別 熵 + 金額交互
        base = (base
            .join(channel_entropy, how='left').fillna({'channel_entropy':0.0})
            .join(currency_entropy, how='left').fillna({'currency_entropy':0.0})
            .join(ch_amt_mean, how='left').fillna(0.0)
            .join(ch_amt_median, how='left').fillna(0.0)
            .join(cur_amt_prop, how='left').fillna(0.0)
        )
        # 新對手比例（近30天）
        base = (base
            .join(new_cp_ratios, how='left').fillna({
                'new_counterparty_ratio_7d': 0.0,
                'new_counterparty_ratio_14d': 0.0,
                'new_counterparty_ratio_30d': 0.0
            })
            .join(benford_l1, how='left').fillna({'benford_l1': 0.0})
            .join(amt_hhi, how='left').fillna({'cp_hhi_amount': 0.0})
            .join(ch_prop, how='left').fillna(0.0)
            .join(cur_prop, how='left').fillna(0.0)
        )
        def _align_to_base_index(df_like, base_index):
            """
            將 df_like 的 index 對齊到 base_index（保留列順序與 pool）。
            實作：以字串做 reindex，最後把 index 指回 base_index。
            """
            df2 = df_like.copy()
            # 以字串做鍵，避免 category pool 不同的對不上
            df2.index = df2.index.astype(str)
            base_idx_str = base_index.astype(str)
            df2 = df2.reindex(base_idx_str)
            # 關鍵：把 index 指回 base 的 CategoricalIndex（類別池一致）
            df2.index = base_index
            return df2

        # 在你 join graph_feat 之前插入：
        graph_feat = _align_to_base_index(graph_feat, base.index)
        base = base.join(graph_feat, how='left').fillna({
            'pr': 0.0,
            'component_size': 1,
            'out_deg': 0, 'in_deg': 0,
            'wout_deg': 0.0, 'win_deg': 0.0,
            'betweenness_approx': 0.0,
            'katz': 0.0,
            'hits_hub': 0.0, 'hits_auth': 0.0,
            'ud_core_number': 0,
            'ud_hop2_size': 0,
            'ud_hop3_size': 0,
        })
        
        base = (base
                .join(sess_feats, how='left').fillna({
                'sess_5min_max': 0.0, 'sess_5min_mean': 0.0, 'sess_5min_p95': 0.0,
                'sess_15min_max': 0.0, 'sess_15min_mean': 0.0, 'sess_15min_p95': 0.0,
                'sess_30min_max': 0.0, 'sess_30min_mean': 0.0, 'sess_30min_p95': 0.0,
                'sess_60min_max': 0.0, 'sess_60min_mean': 0.0, 'sess_60min_p95': 0.0,
            })
        )
        base = (base
                .join(quick_feats, how='left').fillna({
                'quick_out_rate_5min': 0.0,
                'quick_out_large_ratio_rate_5min': 0.0,
                'quick_out_rate_15min': 0.0,
                'quick_out_large_ratio_rate_15min': 0.0,
                'quick_out_rate_30min': 0.0,
                'quick_out_large_ratio_rate_30min': 0.0,
                'quick_out_rate_60min': 0.0,
                'quick_out_large_ratio_rate_60min': 0.0,
            })
        )

        # 壓 dtype
        for c in base.columns:
            if str(base[c].dtype).startswith('int64'):
                base[c] = base[c].astype('int32')
            if str(base[c].dtype).startswith('float64'):
                base[c] = base[c].astype('float32')
        gc.collect()

    # ---------------------------- 對手帳戶二次聚合（強化版）----------------------------
    _log("對手帳戶特徵的二次聚合（無裁邊；精簡統計；向量化）", verbose)
    with StepTimer("counterparty aggregation (no-topk, fast)", verbose):

        # 只挑計算成本低、訊息量高的欄位（可自行增減）
        wanted_cp_cols = [
            'send_count',
            'recv_count',
            'total_txn_count',
            'active_days',
            'txn_per_day_mean',
            'txn_per_day_max',
            'txn_per_day_min',
            'txn_per_day_std',
            'txn_per_day_cv',
            'self_txn_count',
            'self_txn_ratio',
            'send_recv_cnt_ratio',
            'txn_interval_mean',
            'txn_interval_min',
            'prop_short_interval',
            'txn_count_0_3',
            'txn_count_3_6',
            'txn_count_6_9',
            'txn_count_9_12',
            'txn_count_12_15',
            'txn_count_15_18',
            'txn_count_18_21',
            'txn_count_21_24',
            'night_txn_ratio',
            'amount_max',
            'amount_mean',
            'amount_std',
            'amount_sum',
            'amount_median',
            'amount_skew',
            'amount_kurt',
            'amount_iqr',
            'amount_cv',
            'txn_hour_entropy',
            'currency_unique',
            'channel_unique',
            'burst_5min_max',
            'burst_5min_mean',
            'burst_5min_p95',
            'burst_15min_max',
            'burst_15min_mean',
            'burst_15min_p95',
            'burst_30min_max',
            'burst_30min_mean',
            'burst_30min_p95',
            'burst_60min_max',
            'burst_60min_mean',
            'burst_60min_p95',
            'txn_per_day_fano',
            'cp_total_unique',
            'cp_send_unique',
            'cp_recv_unique',
            'cp_entropy',
            'cp_hhi_count',
            'cp_top1_count_share',
            'cp_top3_count_share',
            'cp_top1_amount_share',
            'cp_top3_amount_share',
            'bidir_counterparty_ratio',
            'gini',
            'amount_prop_gt_q90',
            'amount_prop_gt_q95',
            'amount_prop_gt_q99',
            'prop_near_1k',
            'amount_last_digit_entropy',
            'channel_entropy',
            'currency_entropy',
            'new_counterparty_ratio_7d',
            'new_counterparty_ratio_14d',
            'new_counterparty_ratio_30d',
            'benford_l1',
            'cp_hhi_amount',
            'pr',
            'component_size',
            'out_deg',
            'in_deg',
            'wout_deg',
            'win_deg',
            'betweenness_approx',
            'katz',
            'ud_core_number',
            'ud_hop2_size',
            'ud_hop3_size',
            'sess_5min_max',
            'sess_5min_mean',
            'sess_5min_p95',
            'sess_15min_max',
            'sess_15min_mean',
            'sess_15min_p95',
            'sess_30min_max',
            'sess_30min_mean',
            'sess_30min_p95',
            'sess_60min_max',
            'sess_60min_mean',
            'sess_60min_p95',
            'quick_out_rate_5min',
            'quick_out_large_ratio_rate_5min',
            'quick_out_rate_15min',
            'quick_out_large_ratio_rate_15min',
            'quick_out_rate_30min',
            'quick_out_large_ratio_rate_30min',
            'quick_out_rate_60min',
            'quick_out_large_ratio_rate_60min'

        ]
        cp_cols = [c for c in wanted_cp_cols if c in base.columns]

        if len(cp_cols) == 0:
            cp_agg = pd.DataFrame(index=base.index)
        else:
            # 1) 對手帳戶參考表（統一 float32，避免聚合時升為 float64）
            _log(f"  建立對手帳戶參考表，欄位 = {len(cp_cols)}", verbose)
            cp_ref = base[cp_cols].copy()
            for c in cp_ref.columns:
                if str(cp_ref[c].dtype).startswith('int'):
                    cp_ref[c] = cp_ref[c].astype('float32')
                elif str(cp_ref[c].dtype).startswith('float'):
                    cp_ref[c] = cp_ref[c].astype('float32')

            cp_ref = cp_ref.add_prefix('cpref__').reset_index()
            cp_ref['acct'] = cp_ref['acct'].astype(_acct_dtype)
            _log("  建立交易邊（acct–counterparty）聚合表", verbose)
            # 2) 交易彙成唯一邊（acct–counterparty），保留全部邊（不做 Top-K）
            edges_cp = (
                long.groupby(['acct', 'counterparty'], observed=True, sort=False)
                    .agg(txn_count=('amount', 'size'),
                        amount_sum=('amount', 'sum'))
                    .reset_index()
            )
            edges_cp['acct'] = edges_cp['acct'].astype(_acct_dtype)
            edges_cp['counterparty'] = edges_cp['counterparty'].astype(_acct_dtype)
            _log("  合併對手帳戶特徵並做二次聚合", verbose)
            # 3) 合併對手帳戶特徵
            long_with_cp = edges_cp.merge(
                cp_ref,
                left_on='counterparty',
                right_on='acct',
                how='left',
                suffixes=('', '_cpref')
            ).drop(columns=['acct_cpref'], errors='ignore')

            cpref_cols = [c for c in long_with_cp.columns if c.startswith('cpref__')]
            if len(cpref_cols) == 0:
                cp_agg = pd.DataFrame(index=base.index)
            else:
                g = long_with_cp.groupby('acct', observed=True, sort=False)
                _log("  計算對手帳戶特徵聚合值", verbose)
                # ---- (A) 無權重：mean / std（便宜、穩定） ----
                unw_mean = g[cpref_cols].mean().add_suffix('_mean')
                unw_std  = g[cpref_cols].std(ddof=1).fillna(0.0).add_suffix('_std')
                _log("    計算金額加權 mean", verbose)
                # ---- (B) 金額加權 mean（向量化；不算加權 std 以省時）----
                denom = g['amount_sum'].sum().rename('w_denom')  # 每個 acct 的權重總和
                X = long_with_cp[cpref_cols]
                WX = X.mul(long_with_cp['amount_sum'].values, axis=0)  # element-wise 權重乘法
                w_num = WX.groupby(long_with_cp['acct'], observed=True, sort=False).sum()
                w_mean = (w_num.div(denom.replace(0, np.nan), axis=0)
                                .fillna(0.0)
                                .add_suffix('_wmean_amt'))

                # ---- (C) 覆蓋度與強度（便宜）----
                _log("    計算覆蓋度與強度", verbose)
                cp_unique = g['counterparty'].nunique().rename('cpref__n_unique_cp')
                strength = g[['amount_sum', 'txn_count']].agg(['sum', 'mean'])
                strength.columns = [f"edge_{c}_{stat}" for (c, stat) in strength.columns]

                # ---- (D) 組合 / 壓 dtype / 對齊 index ----
                _log("    組合對手帳戶聚合特徵", verbose)
                cp_agg = pd.concat([unw_mean, unw_std, w_mean, cp_unique, strength], axis=1)

                for c in cp_agg.columns:
                    if str(cp_agg[c].dtype).startswith('float64'):
                        cp_agg[c] = cp_agg[c].astype('float32')

                cp_agg.index = cp_agg.index.astype(str)
                base_idx_str = base.index.astype(str)
                cp_agg = cp_agg.reindex(base_idx_str)
                cp_agg.index = base.index

            del long_with_cp, edges_cp
            gc.collect()

    with StepTimer("merge base + cp_agg → df_user_features", verbose):
        _log("  合併基礎特徵與對手帳戶特徵", verbose)
        df_user_features = base.join(cp_agg, how='left').fillna(0).reset_index()
        for c in df_user_features.columns:
            if df_user_features[c].dtype.kind in 'iuf' and df_user_features[c].isna().any():
                df_user_features[c] = df_user_features[c].fillna(0)
    _log(f"完成！帳戶 = {df_user_features.shape[0]:,} ，欄位 = {df_user_features.shape[1]:,}", verbose)

    return df_user_features

