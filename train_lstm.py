"""
LSTM 模型訓練與評估腳本
使用 Keras/TensorFlow 建立、訓練並評估人群計數預測模型
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 日誌

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 導入資料準備函數
from LSTM import prepare_lstm_data


# ============================================================================
# 步驟 1: 建立 LSTM 模型
# ============================================================================

def build_lstm_model(
    input_shape: tuple = (12, 4),
    lstm_units: int = 32,
    dropout_rate: float = 0.2
) -> Sequential:
    """
    建立 LSTM 模型
    
    Args:
        input_shape: 輸入形狀 (time_steps, features)
        lstm_units: LSTM 層單元數
        dropout_rate: Dropout 比例
        
    Returns:
        編譯後的 Keras Sequential 模型
    """
    model = Sequential([
        # Layer 1: LSTM
        LSTM(units=lstm_units, input_shape=input_shape),
        
        # Layer 2: Dropout（防止過擬合）
        Dropout(rate=dropout_rate),
        
        # Layer 3: Dense 輸出層
        Dense(units=1)
    ])
    
    # 編譯模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'  # Mean Squared Error
    )
    
    print("\n" + "=" * 60)
    print("LSTM 模型架構")
    print("=" * 60)
    model.summary()
    
    return model


# ============================================================================
# 步驟 2: 訓練模型
# ============================================================================

def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10
) -> dict:
    """
    訓練 LSTM 模型
    
    Args:
        model: Keras 模型
        X_train, y_train: 訓練資料
        X_test, y_test: 驗證資料
        batch_size: 批次大小
        epochs: 最大訓練週期數
        patience: EarlyStopping 耐心值
        
    Returns:
        訓練歷史記錄
    """
    print("\n" + "=" * 60)
    print("開始訓練")
    print("=" * 60)
    print(f"批次大小: {batch_size}")
    print(f"最大週期數: {epochs}")
    print(f"EarlyStopping patience: {patience}")
    
    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # 訓練模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"\n訓練完成！最終 epochs: {len(history.history['loss'])}")
    print(f"最佳 val_loss: {min(history.history['val_loss']):.6f}")
    
    return history


# ============================================================================
# 步驟 3: 反正規化函數（修正版：使用 dummy array 處理 4 特徵 scaler）
# ============================================================================

def inverse_transform_target(
    scaled_values: np.ndarray,
    scaler,
    target_feature_idx: int = 0,
    n_features: int = 4,
    max_capacity: int = 100
) -> np.ndarray:
    """
    將正規化的目標值反轉換為真實人數
    
    處理流程:
    1. 建立 dummy array (n_samples, n_features) 以符合 scaler 的期望形狀
    2. 將目標值填入對應的特徵位置
    3. 執行 scaler.inverse_transform
    4. 提取目標特徵的反轉換結果
    5. 再乘以 max_capacity 還原到真實人數
    
    Args:
        scaled_values: 正規化後的預測值或真實值 (1D array)
        scaler: MinMaxScaler 物件（擬合於 4 個特徵）
        target_feature_idx: 目標特徵在 scaler 中的索引（預設 0 = people_count）
        n_features: 特徵數量
        max_capacity: 原始正規化時使用的最大容量
        
    Returns:
        真實人數（整數）
    """
    n_samples = len(scaled_values)
    
    # 建立 dummy array，填入 0
    dummy = np.zeros((n_samples, n_features))
    
    # 將目標值填入對應位置
    dummy[:, target_feature_idx] = scaled_values
    
    # 使用 scaler 反轉換
    inversed = scaler.inverse_transform(dummy)
    
    # 提取目標特徵的結果
    result = inversed[:, target_feature_idx]
    
    # 還原到真實人數（乘以 max_capacity）
    # 因為原始資料在 create_features 時除以了 max_capacity
    real_values = result * max_capacity
    
    return real_values


def postprocess_predictions(
    predictions: np.ndarray
) -> np.ndarray:
    """
    後處理預測值：
    1. Clip: 強制為非負數（人數不能為負）
    2. Round: 四捨五入為整數（人數為離散值）
    
    Args:
        predictions: 原始預測值
        
    Returns:
        後處理後的整數預測值
    """
    # Step A: Clip 負值為 0
    clipped = np.maximum(predictions, 0)
    
    # Step B: 四捨五入為整數
    rounded = np.round(clipped).astype(int)
    
    return rounded


# ============================================================================
# 步驟 4: 評估與視覺化（修正版：顯示真實人數）
# ============================================================================

def evaluate_and_plot(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history: dict,
    scaler,
    max_capacity: int = 100,
    save_dir: str = './output'
):
    """
    評估模型並生成視覺化圖表
    
    完整流程:
    1. 預測
    2. 反正規化（使用 dummy array）
    3. 後處理（Clip -> Round）
    4. 計算指標
    5. 繪製圖表
    
    Args:
        model: 訓練好的模型
        X_test, y_test: 測試資料（已正規化）
        history: 訓練歷史
        scaler: MinMaxScaler 物件
        max_capacity: 原始正規化使用的最大容量
        save_dir: 圖表儲存目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ========================================================================
    # 1. 預測
    # ========================================================================
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    print("\n" + "=" * 60)
    print("預測與後處理流程")
    print("=" * 60)
    print(f"原始預測值範圍（正規化後）: {y_pred_scaled.min():.4f} ~ {y_pred_scaled.max():.4f}")
    
    # ========================================================================
    # 2. 反正規化為真實人數
    # ========================================================================
    y_test_real = inverse_transform_target(y_test, scaler, max_capacity=max_capacity)
    y_pred_real = inverse_transform_target(y_pred_scaled, scaler, max_capacity=max_capacity)
    
    print(f"反正規化後預測範圍: {y_pred_real.min():.2f} ~ {y_pred_real.max():.2f}")
    print(f"反正規化後實際範圍: {y_test_real.min():.2f} ~ {y_test_real.max():.2f}")
    
    # ========================================================================
    # 3. 後處理：Clip (非負) + Round (整數)
    # ========================================================================
    y_test_int = postprocess_predictions(y_test_real)
    y_pred_int = postprocess_predictions(y_pred_real)
    
    print(f"\n後處理結果（Clip + Round）:")
    print(f"  實際人數範圍: {y_test_int.min()} ~ {y_test_int.max()} 人")
    print(f"  預測人數範圍: {y_pred_int.min()} ~ {y_pred_int.max()} 人")
    
    # ========================================================================
    # 4. 計算評估指標（基於整數值）
    # ========================================================================
    rmse = np.sqrt(mean_squared_error(y_test_int, y_pred_int))
    mae = mean_absolute_error(y_test_int, y_pred_int)
    
    # 計算百分比誤差
    mask = y_test_int > 0  # 避免除以零
    if mask.sum() > 0:
        mape = np.mean(np.abs(y_test_int[mask] - y_pred_int[mask]) / y_test_int[mask]) * 100
    else:
        mape = 0
    
    print("\n" + "=" * 60)
    print("模型評估結果（真實人數，整數）")
    print("=" * 60)
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} 人")
    print(f"MAE  (Mean Absolute Error):     {mae:.2f} 人")
    print(f"MAPE (Mean Absolute % Error):   {mape:.1f}%")
    print(f"測試樣本數: {len(y_test_int)}")
    
    # 誤差分布統計
    errors = y_pred_int - y_test_int
    print(f"\n誤差分布:")
    print(f"  平均誤差: {errors.mean():.2f} 人")
    print(f"  誤差標準差: {errors.std():.2f} 人")
    print(f"  最大低估: {errors.min()} 人")
    print(f"  最大高估: {errors.max()} 人")
    
    # ========================================================================
    # 5. Plot 1: Loss Curve
    # ========================================================================
    fig = plt.figure(figsize=(14, 10))
    
    # Loss Curve
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Loss Curve (Training vs Validation)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # 6. Plot 2: Prediction vs Actual（整數人數）
    # ========================================================================
    ax2 = fig.add_subplot(2, 2, 2)
    time_steps = np.arange(len(y_test_int))
    
    ax2.plot(time_steps, y_test_int, label='Actual', color='blue', linewidth=1.2, alpha=0.8)
    ax2.plot(time_steps, y_pred_int, label='Predicted', color='red', linewidth=1.2, alpha=0.8)
    
    ax2.set_title(f'Prediction vs Actual (RMSE={rmse:.2f}, MAE={mae:.2f})', fontsize=14)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('People Count', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 設定 Y 軸為整數刻度
    max_y = max(y_test_int.max(), y_pred_int.max())
    ax2.set_ylim(0, max_y + 5)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # ========================================================================
    # 7. Plot 3: 誤差分布直方圖
    # ========================================================================
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(errors, bins=range(int(errors.min()) - 1, int(errors.max()) + 2), 
             color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_title('Prediction Error Distribution', fontsize=14)
    ax3.set_xlabel('Error (Predicted - Actual)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # 8. Plot 4: Scatter Plot (Actual vs Predicted)
    # ========================================================================
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(y_test_int, y_pred_int, alpha=0.5, s=20, c='steelblue')
    
    # 理想線（y = x）
    max_val = max(y_test_int.max(), y_pred_int.max())
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax4.set_title('Actual vs Predicted Scatter', fontsize=14)
    ax4.set_xlabel('Actual People Count', fontsize=12)
    ax4.set_ylabel('Predicted People Count', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_val + 2)
    ax4.set_ylim(0, max_val + 2)
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_path = os.path.join(save_dir, 'lstm_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n圖表已儲存: {plot_path}")
    
    plt.show()
    
    # ========================================================================
    # 9. 分日顯示預測結果
    # ========================================================================
    plot_daily_predictions(y_test_int, y_pred_int, save_dir)
    
    return {
        'y_test_real': y_test_int,
        'y_pred_real': y_pred_int,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def plot_daily_predictions(
    y_test_int: np.ndarray,
    y_pred_int: np.ndarray,
    save_dir: str,
    samples_per_day: int = 94
):
    """
    分日顯示預測結果（整數人數版本）
    """
    n_days = len(y_test_int) // samples_per_day
    
    # 增加畫布高度：每個子圖 4 吋高度，確保有足夠空間
    fig, axes = plt.subplots(n_days, 1, figsize=(15, 4 * n_days))
    
    for day in range(n_days):
        start_idx = day * samples_per_day
        end_idx = start_idx + samples_per_day
        
        ax = axes[day] if n_days > 1 else axes
        
        # 時間標籤：從 09:00 開始（因為 lookback=12，損失前 12 個時間步 = 1 小時）
        time_labels = np.arange(samples_per_day)
        
        actual = y_test_int[start_idx:end_idx]
        predicted = y_pred_int[start_idx:end_idx]
        
        ax.plot(time_labels, actual, label='Actual', color='blue', linewidth=1.5, marker='o', markersize=2)
        ax.plot(time_labels, predicted, label='Predicted', color='red', linewidth=1.5, alpha=0.8, marker='x', markersize=2)
        
        day_rmse = np.sqrt(mean_squared_error(actual, predicted))
        day_mae = mean_absolute_error(actual, predicted)
        
        ax.set_title(f'Day {day + 11} (Test Day {day + 1}) - RMSE: {day_rmse:.2f}, MAE: {day_mae:.2f}', fontsize=12, pad=10)
        ax.set_xlabel('Time Step (5-min intervals from 09:00)', fontsize=10)
        ax.set_ylabel('People Count', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 設定 Y 軸為整數刻度
        max_y = max(actual.max(), predicted.max())
        ax.set_ylim(0, max_y + 3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 增加子圖之間的垂直間距
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'lstm_daily_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Daily prediction results saved: {plot_path}")
    
    plt.show()


# ============================================================================
# 主程式
# ============================================================================

def main():
    """
    完整的 LSTM 訓練與評估流程
    """
    print("=" * 60)
    print("LSTM 人群計數預測模型")
    print("=" * 60)
    
    # ========================================================================
    # 1. 準備資料
    # ========================================================================
    print("\n[階段 1] 資料準備")
    
    data = prepare_lstm_data(
        data_path='./lstm_data.npy',
        train_days=10,
        test_days=5,
        lookback=12,
        forecast_horizon=1,
        use_scaler=True
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['scaler']
    
    # ========================================================================
    # 2. 建立模型
    # ========================================================================
    print("\n[階段 2] 建立模型")
    
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),  # (12, 4)
        lstm_units=32,
        dropout_rate=0.2
    )
    
    # ========================================================================
    # 3. 訓練模型
    # ========================================================================
    print("\n[階段 3] 訓練模型")
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    # ========================================================================
    # 4. 評估與視覺化
    # ========================================================================
    print("\n[階段 4] 評估與視覺化")
    
    results = evaluate_and_plot(
        model=model,
        X_test=X_test,
        y_test=y_test,
        history=history,
        scaler=scaler,
        save_dir='./output'
    )
    
    # ========================================================================
    # 5. 儲存模型
    # ========================================================================
    model_path = './output/lstm_model.keras'
    model.save(model_path)
    print(f"\n模型已儲存: {model_path}")
    
    print("\n" + "=" * 60)
    print("訓練與評估完成！")
    print("=" * 60)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
