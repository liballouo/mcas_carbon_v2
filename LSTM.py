"""
LSTM 資料前處理腳本
對人群計數 JSON 資料進行預處理，生成適用於 LSTM 模型的 3D Numpy 陣列

資料流程:
1. 載入多個 JSON 檔案
2. 時間索引與重採樣
3. 缺失值插補
4. 時間區間過濾 (08:00-16:45)
5. 特徵工程（正規化）
6. 序列建構（3D 陣列）
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# 常數設定
# ============================================================================

# 時間區間設定
START_TIME = time(8, 0, 0)    # 開始時間: 08:00
END_TIME = time(16, 45, 0)    # 結束時間: 16:45
INTERVAL_MINUTES = 5          # 採樣間隔: 5 分鐘

# 計算每天應有的時間步數
# 從 08:00 到 16:45 共 8 小時 45 分鐘 = 525 分鐘
# 每 5 分鐘一筆: 525 / 5 + 1 = 106 筆（包含起點和終點）
EXPECTED_STEPS_PER_DAY = 106

# 正規化參數
MAX_CAPACITY = 100  # 假設最大人數容量


# ============================================================================
# 資料載入函數
# ============================================================================

def load_json_files(data_dir: str) -> pd.DataFrame:
    """
    從指定資料夾載入所有 JSON 檔案，合併為單一 DataFrame
    
    Args:
        data_dir: 包含 JSON 檔案的資料夾路徑
        
    Returns:
        合併後的 DataFrame
    """
    all_data = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"在 {data_dir} 中找不到任何 JSON 檔案")
    
    print(f"找到 {len(json_files)} 個 JSON 檔案")
    
    for filename in sorted(json_files):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  載入: {filename} ({len(data)} 筆記錄)")
    
    # 建立 DataFrame
    df = pd.DataFrame(all_data)
    print(f"\n總計載入 {len(df)} 筆記錄")
    
    return df


# ============================================================================
# 時間處理函數
# ============================================================================

def convert_to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 timestamp 欄位轉換為 datetime 並設為索引
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        設定 datetime 索引後的 DataFrame
    """
    df = df.copy()
    
    # 轉換 timestamp 為 datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 設為索引
    df.set_index('timestamp', inplace=True)
    
    # 排序索引
    df.sort_index(inplace=True)
    
    return df


def resample_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    對資料進行重採樣，確保嚴格的 5 分鐘間隔
    並對缺失值進行線性插補
    
    Args:
        df: 時間索引的 DataFrame
        
    Returns:
        重採樣並插補後的 DataFrame
    """
    # 按日期分組處理
    df['date'] = df.index.date
    
    resampled_dfs = []
    
    for date, group in df.groupby('date'):
        # 對每天的資料進行重採樣
        # 使用 '5min' 作為頻率，取每個區間的平均值
        daily_resampled = group[['people_count']].resample('5min').mean()
        
        # 線性插補缺失值
        daily_resampled['people_count'] = daily_resampled['people_count'].interpolate(
            method='linear',
            limit_direction='both'
        )
        
        resampled_dfs.append(daily_resampled)
    
    # 合併所有日期的資料
    result = pd.concat(resampled_dfs)
    
    # 移除可能的重複索引
    result = result[~result.index.duplicated(keep='first')]
    
    return result


def filter_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    過濾資料，只保留 08:00 至 16:45 之間的記錄
    
    Args:
        df: 時間索引的 DataFrame
        
    Returns:
        過濾後的 DataFrame
    """
    # 取得每筆記錄的時間部分
    time_mask = (df.index.time >= START_TIME) & (df.index.time <= END_TIME)
    
    filtered_df = df[time_mask].copy()
    
    original_count = len(df)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    print(f"\n時間過濾: 保留 {filtered_count} 筆, 移除 {removed_count} 筆")
    print(f"時間範圍: {START_TIME} - {END_TIME}")
    
    return filtered_df


# ============================================================================
# 特徵工程函數
# ============================================================================

def create_features(df: pd.DataFrame, max_capacity: int = MAX_CAPACITY) -> pd.DataFrame:
    """
    建立正規化特徵
    
    特徵列表:
    - people_count_norm: 正規化人數 (0-1)
    - hour_norm: 正規化小時 (0-1)
    - minute_norm: 正規化分鐘 (0-1)
    - weekday_norm: 正規化星期 (0-1)
    
    Args:
        df: 時間索引的 DataFrame
        max_capacity: 最大人數容量（用於正規化）
        
    Returns:
        包含特徵欄位的 DataFrame
    """
    df = df.copy()
    
    # 正規化人數 (使用 max_capacity 進行正規化)
    df['people_count_norm'] = df['people_count'] / max_capacity
    # 確保值在 0-1 之間
    df['people_count_norm'] = df['people_count_norm'].clip(0, 1)
    
    # 正規化小時 (0-23 -> 0-1)
    df['hour_norm'] = df.index.hour / 23.0
    
    # 正規化分鐘 (0-59 -> 0-1)
    df['minute_norm'] = df.index.minute / 59.0
    
    # 正規化星期 (0-6 -> 0-1)
    df['weekday_norm'] = df.index.weekday / 6.0
    
    print(f"\n特徵工程完成:")
    print(f"  - people_count_norm (max_capacity={max_capacity})")
    print(f"  - hour_norm (0-23)")
    print(f"  - minute_norm (0-59)")
    print(f"  - weekday_norm (0-6)")
    
    return df


# ============================================================================
# 序列建構函數
# ============================================================================

def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    expected_steps: int = EXPECTED_STEPS_PER_DAY
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    將資料按日期分組，建立 3D Numpy 陣列
    
    Args:
        df: 包含特徵的 DataFrame
        feature_columns: 要包含的特徵欄位名稱列表
        expected_steps: 每天預期的時間步數
        
    Returns:
        - 3D Numpy 陣列 (n_days, time_steps, n_features)
        - 有效日期列表
        - 被丟棄的日期列表
    """
    # 按日期分組
    df['date'] = df.index.date
    grouped = df.groupby('date')
    
    valid_sequences = []
    valid_dates = []
    discarded_dates = []
    
    print(f"\n序列建構:")
    print(f"預期每天時間步數: {expected_steps}")
    print("-" * 50)
    
    for date, group in grouped:
        n_steps = len(group)
        
        if n_steps == expected_steps:
            # 完整的一天資料
            features = group[feature_columns].values
            valid_sequences.append(features)
            valid_dates.append(str(date))
            print(f"  ✓ {date}: {n_steps} 步 (有效)")
        elif n_steps < expected_steps:
            # 資料不足
            discarded_dates.append(str(date))
            print(f"  ✗ {date}: {n_steps} 步 (不足，丟棄)")
        else:
            # 資料過多（不應發生，但以防萬一取前 expected_steps 筆）
            features = group[feature_columns].values[:expected_steps]
            valid_sequences.append(features)
            valid_dates.append(str(date))
            print(f"  ! {date}: {n_steps} 步 (過多，截斷至 {expected_steps})")
    
    if not valid_sequences:
        raise ValueError("沒有任何有效的日期資料！")
    
    # 堆疊為 3D 陣列
    sequences = np.array(valid_sequences)
    
    print("-" * 50)
    print(f"有效天數: {len(valid_dates)}")
    print(f"丟棄天數: {len(discarded_dates)}")
    if discarded_dates:
        print(f"被丟棄的日期: {discarded_dates}")
    
    return sequences, valid_dates, discarded_dates


# ============================================================================
# 主要前處理流程
# ============================================================================

def preprocess_data(
    data_dir: str,
    max_capacity: int = MAX_CAPACITY,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    主要資料前處理流程
    
    Args:
        data_dir: JSON 資料檔案所在資料夾
        max_capacity: 最大人數容量（用於正規化）
        save_path: 可選，儲存處理後陣列的路徑
        
    Returns:
        - X: 3D Numpy 陣列 (n_days, time_steps, n_features)
        - valid_dates: 有效日期列表
        - processed_df: 處理後的 DataFrame
    """
    print("=" * 60)
    print("LSTM 資料前處理開始")
    print("=" * 60)
    
    # 1. 載入資料
    print("\n[步驟 1] 載入 JSON 資料...")
    df = load_json_files(data_dir)
    
    # 2. 轉換時間索引
    print("\n[步驟 2] 轉換時間索引...")
    df = convert_to_datetime_index(df)
    print(f"時間範圍: {df.index.min()} ~ {df.index.max()}")
    
    # 3. 重採樣與插補
    print("\n[步驟 3] 重採樣與缺失值插補...")
    df = resample_and_interpolate(df)
    print(f"重採樣後記錄數: {len(df)}")
    
    # 4. 時間區間過濾
    print("\n[步驟 4] 時間區間過濾...")
    df = filter_time_window(df)
    
    # 5. 特徵工程
    print("\n[步驟 5] 特徵工程...")
    df = create_features(df, max_capacity=max_capacity)
    
    # 6. 建構序列
    print("\n[步驟 6] 建構 3D 序列...")
    feature_columns = ['people_count_norm', 'hour_norm', 'minute_norm', 'weekday_norm']
    X, valid_dates, discarded_dates = create_sequences(df, feature_columns)
    
    # 輸出最終結果
    print("\n" + "=" * 60)
    print("前處理完成!")
    print("=" * 60)
    print(f"\n最終輸出陣列形狀: {X.shape}")
    print(f"  - 天數 (samples): {X.shape[0]}")
    print(f"  - 時間步數 (time_steps): {X.shape[1]}")
    print(f"  - 特徵數 (features): {X.shape[2]}")
    print(f"\n特徵順序: {feature_columns}")
    
    # 儲存處理後的陣列
    if save_path:
        np.save(save_path, X)
        print(f"\n已儲存至: {save_path}")
    
    return X, valid_dates, df


# ============================================================================
# 時序分割函數（嚴格時間順序，無資料洩漏）
# ============================================================================

def time_based_split(
    data: np.ndarray,
    valid_dates: List[str],
    train_days: int = 10,
    test_days: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    嚴格的時序分割：前 N 天訓練，後 M 天測試
    
    Args:
        data: 3D 陣列 (n_days, time_steps, n_features)
        valid_dates: 有效日期列表
        train_days: 訓練集天數（預設: 10 天 = 2 週）
        test_days: 測試集天數（預設: 5 天 = 1 週）
        
    Returns:
        train_data, test_data, 分割資訊字典
    """
    n_total_days = data.shape[0]
    
    # 驗證天數
    if train_days + test_days > n_total_days:
        raise ValueError(f"訓練天數({train_days}) + 測試天數({test_days}) > 總天數({n_total_days})")
    
    # 嚴格切片
    train_data = data[:train_days]  # Day 0 to Day 9
    test_data = data[train_days:train_days + test_days]  # Day 10 to Day 14
    
    split_info = {
        'n_total': n_total_days,
        'n_train': train_days,
        'n_test': test_days,
        'train_dates': valid_dates[:train_days],
        'test_dates': valid_dates[train_days:train_days + test_days]
    }
    
    print("\n" + "=" * 60)
    print("時序分割結果 (Time-Based Split)")
    print("=" * 60)
    print(f"總天數: {n_total_days}")
    print(f"訓練集: Day 0 - Day {train_days - 1} ({train_days} 天)")
    print(f"測試集: Day {train_days} - Day {train_days + test_days - 1} ({test_days} 天)")
    print(f"\n訓練集形狀: {train_data.shape}")
    print(f"測試集形狀: {test_data.shape}")
    print(f"\n訓練日期: {split_info['train_dates']}")
    print(f"測試日期: {split_info['test_dates']}")
    
    return train_data, test_data, split_info


# ============================================================================
# MinMaxScaler 正規化函數（僅在訓練資料上擬合）
# ============================================================================

def fit_scaler_on_train(
    train_data: np.ndarray,
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    在訓練資料上擬合 MinMaxScaler
    
    Args:
        train_data: 3D 訓練陣列 (n_days, time_steps, n_features)
        feature_range: 正規化範圍
        
    Returns:
        scaler: 擬合後的 MinMaxScaler
        train_scaled: 正規化後的訓練資料
    """
    n_days, time_steps, n_features = train_data.shape
    
    # 展平為 2D 以擬合 scaler: (n_days * time_steps, n_features)
    train_flat = train_data.reshape(-1, n_features)
    
    # 建立並擬合 scaler（僅用訓練資料！）
    scaler = MinMaxScaler(feature_range=feature_range)
    train_scaled_flat = scaler.fit_transform(train_flat)
    
    # 還原為 3D
    train_scaled = train_scaled_flat.reshape(n_days, time_steps, n_features)
    
    print("\n[MinMaxScaler] 已在訓練資料上擬合")
    print(f"  特徵最小值: {scaler.data_min_}")
    print(f"  特徵最大值: {scaler.data_max_}")
    
    return scaler, train_scaled


def transform_with_scaler(
    scaler: MinMaxScaler,
    data: np.ndarray
) -> np.ndarray:
    """
    使用已擬合的 scaler 轉換資料
    
    Args:
        scaler: 已擬合的 MinMaxScaler
        data: 3D 陣列 (n_days, time_steps, n_features)
        
    Returns:
        scaled_data: 正規化後的資料
    """
    n_days, time_steps, n_features = data.shape
    
    # 展平、轉換、還原
    data_flat = data.reshape(-1, n_features)
    scaled_flat = scaler.transform(data_flat)
    scaled_data = scaled_flat.reshape(n_days, time_steps, n_features)
    
    return scaled_data


# ============================================================================
# 滑動視窗序列生成函數
# ============================================================================

def create_sliding_window_sequences(
    data: np.ndarray,
    lookback: int = 12,
    forecast_horizon: int = 1,
    target_feature_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    對單一資料集建立滑動視窗序列
    
    Args:
        data: 3D 陣列 (n_days, time_steps, n_features)
        lookback: 回顧視窗大小（使用多少個時間步來預測）
        forecast_horizon: 預測未來幾個時間步
        target_feature_idx: 目標特徵索引（預設 0 = people_count）
        
    Returns:
        X: (n_samples, lookback, n_features)
        y: (n_samples,) 或 (n_samples, forecast_horizon)
    """
    n_days, time_steps, n_features = data.shape
    
    X_list = []
    y_list = []
    
    for day in range(n_days):
        daily_data = data[day]
        # 滑動視窗：從 lookback 位置開始，到 (time_steps - forecast_horizon) 結束
        for i in range(lookback, time_steps - forecast_horizon + 1):
            # X: 過去 lookback 個時間步的所有特徵
            X_list.append(daily_data[i - lookback:i])
            # y: 未來 forecast_horizon 個時間步的目標特徵
            y_list.append(daily_data[i:i + forecast_horizon, target_feature_idx])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 如果只預測一步，將 y 壓縮為 1D
    if forecast_horizon == 1:
        y = y.squeeze()
    
    return X, y


# ============================================================================
# 完整的時序分割與準備流程
# ============================================================================

def prepare_lstm_data(
    data_path: str = './lstm_data.npy',
    train_days: int = 10,
    test_days: int = 5,
    lookback: int = 12,
    forecast_horizon: int = 1,
    use_scaler: bool = True
) -> dict:
    """
    完整的 LSTM 資料準備流程
    
    Args:
        data_path: 預處理後的 .npy 檔案路徑
        train_days: 訓練天數
        test_days: 測試天數
        lookback: 回顧視窗大小
        forecast_horizon: 預測範圍
        use_scaler: 是否使用 MinMaxScaler
        
    Returns:
        包含所有準備好資料的字典
    """
    print("=" * 60)
    print("LSTM 資料準備開始")
    print("=" * 60)
    
    # 1. 載入資料
    print(f"\n[步驟 1] 載入資料: {data_path}")
    data = np.load(data_path)
    print(f"原始資料形狀: {data.shape}")
    print(f"  - 天數: {data.shape[0]}")
    print(f"  - 時間步數: {data.shape[1]}")
    print(f"  - 特徵數: {data.shape[2]}")
    
    # 生成日期列表（假設檔案中沒有，這裡模擬）
    valid_dates = [f"Day_{i}" for i in range(data.shape[0])]
    
    # 2. 時序分割（先分割後正規化，避免資料洩漏）
    print("\n[步驟 2] 時序分割...")
    train_data, test_data, split_info = time_based_split(
        data, valid_dates, train_days=train_days, test_days=test_days
    )
    
    # 3. 正規化（僅在訓練資料上擬合 scaler）
    scaler = None
    if use_scaler:
        print("\n[步驟 3] MinMaxScaler 正規化...")
        scaler, train_data = fit_scaler_on_train(train_data)
        test_data = transform_with_scaler(scaler, test_data)
        print("  測試資料已使用訓練資料的 scaler 轉換（無資料洩漏）")
    
    # 4. 建立滑動視窗序列
    print("\n[步驟 4] 建立滑動視窗序列...")
    print(f"  回顧視窗 (lookback): {lookback} 步 = {lookback * 5} 分鐘")
    print(f"  預測範圍 (forecast): {forecast_horizon} 步")
    
    # 分別對訓練和測試資料建立序列
    X_train, y_train = create_sliding_window_sequences(
        train_data, lookback=lookback, forecast_horizon=forecast_horizon
    )
    X_test, y_test = create_sliding_window_sequences(
        test_data, lookback=lookback, forecast_horizon=forecast_horizon
    )
    
    # 5. 輸出最終形狀
    print("\n" + "=" * 60)
    print("最終資料形狀驗證")
    print("=" * 60)
    print(f"\n訓練資料:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  樣本數: {X_train.shape[0]} (來自 {train_days} 天)")
    
    print(f"\n測試資料:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  樣本數: {X_test.shape[0]} (來自 {test_days} 天)")
    
    # 計算理論樣本數
    steps_per_day = data.shape[1]
    samples_per_day = steps_per_day - lookback - forecast_horizon + 1
    expected_train_samples = train_days * samples_per_day
    expected_test_samples = test_days * samples_per_day
    
    print(f"\n理論計算:")
    print(f"  每天可產生樣本數: {samples_per_day}")
    print(f"  預期訓練樣本: {expected_train_samples}")
    print(f"  預期測試樣本: {expected_test_samples}")
    
    # 驗證無資料洩漏
    print("\n" + "=" * 60)
    print("資料洩漏檢查")
    print("=" * 60)
    print("✓ 時序分割: 訓練集 (Day 0-9) 在測試集 (Day 10-14) 之前")
    print("✓ Scaler: 僅在訓練資料上擬合，測試資料使用相同 scaler 轉換")
    print("✓ 滑動視窗: 訓練和測試分別獨立生成，無跨集序列")
    
    print("\n" + "=" * 60)
    print("準備完成！可用於 Keras/TensorFlow LSTM 模型訓練")
    print("=" * 60)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'split_info': split_info,
        'config': {
            'lookback': lookback,
            'forecast_horizon': forecast_horizon,
            'train_days': train_days,
            'test_days': test_days
        }
    }


# ============================================================================
# 輔助函數：分割訓練/驗證/測試集（舊版，保留相容性）
# ============================================================================

def train_val_test_split(
    X: np.ndarray,
    valid_dates: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    將資料分割為訓練集、驗證集、測試集（比例分割，舊版）
    
    Args:
        X: 3D 陣列 (n_days, time_steps, n_features)
        valid_dates: 有效日期列表
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        
    Returns:
        X_train, X_val, X_test, 分割資訊字典
    """
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    
    split_info = {
        'n_total': n_samples,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'train_dates': valid_dates[:n_train],
        'val_dates': valid_dates[n_train:n_train + n_val],
        'test_dates': valid_dates[n_train + n_val:]
    }
    
    print(f"\n資料分割:")
    print(f"  訓練集: {split_info['n_train']} 天")
    print(f"  驗證集: {split_info['n_val']} 天")
    print(f"  測試集: {split_info['n_test']} 天")
    
    return X_train, X_val, X_test, split_info


# ============================================================================
# 輔助函數：建立 X, y 對（舊版，保留相容性）
# ============================================================================

def create_xy_pairs(
    X: np.ndarray,
    lookback: int = 12,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    建立 LSTM 監督式學習用的 X, y 配對（舊版）
    使用滑動視窗方式建立序列
    
    Args:
        X: 3D 陣列 (n_days, time_steps, n_features)
        lookback: 回顧視窗大小（使用多少個時間步來預測）
        forecast_horizon: 預測未來幾個時間步
        
    Returns:
        X_seq: (n_samples, lookback, n_features)
        y_seq: (n_samples, forecast_horizon) - 只預測 people_count_norm
    """
    n_days, time_steps, n_features = X.shape
    
    X_seq = []
    y_seq = []
    
    for day in range(n_days):
        daily_data = X[day]
        for i in range(lookback, time_steps - forecast_horizon + 1):
            X_seq.append(daily_data[i - lookback:i])
            # 只取 people_count_norm (index 0) 作為目標
            y_seq.append(daily_data[i:i + forecast_horizon, 0])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"\n滑動視窗序列建立:")
    print(f"  回顧視窗: {lookback} 步")
    print(f"  預測範圍: {forecast_horizon} 步")
    print(f"  X_seq 形狀: {X_seq.shape}")
    print(f"  y_seq 形狀: {y_seq.shape}")
    
    return X_seq, y_seq


# ============================================================================
# 主程式入口
# ============================================================================

def main():
    """
    主程式：使用時序分割準備 LSTM 資料
    """
    # 方法一：如果已有 lstm_data.npy，直接使用新的時序分割
    DATA_PATH = './lstm_data.npy'
    
    if os.path.exists(DATA_PATH):
        print("使用已存在的預處理資料...")
        result = prepare_lstm_data(
            data_path=DATA_PATH,
            train_days=10,      # 前 2 週（10 個工作日）
            test_days=5,        # 第 3 週（5 個工作日）
            lookback=12,        # 回顧 1 小時 (12 * 5min)
            forecast_horizon=1, # 預測下一個時間步
            use_scaler=True     # 使用 MinMaxScaler
        )
        
        # 返回準備好的資料供後續使用
        return result
    
    # 方法二：如果沒有 npy 檔案，先執行前處理
    else:
        print("未找到預處理資料，開始前處理流程...")
        config = {
            'data_dir': './output',
            'max_capacity': 100,
            'save_path': DATA_PATH
        }
        
        # 執行前處理
        X, valid_dates, processed_df = preprocess_data(**config)
        
        print("\n預處理完成，開始準備 LSTM 資料...")
        
        # 使用新的時序分割
        result = prepare_lstm_data(
            data_path=DATA_PATH,
            train_days=10,
            test_days=5,
            lookback=12,
            forecast_horizon=1,
            use_scaler=True
        )
        
        return result


if __name__ == "__main__":
    result = main()
    
    # 解包結果供後續使用
    X_train = result['X_train']
    y_train = result['y_train']
    X_test = result['X_test']
    y_test = result['y_test']
    scaler = result['scaler']