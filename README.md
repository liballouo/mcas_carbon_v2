# mcas_carbon_v2

人流追蹤與熱力圖生成系統 + LSTM 人流預測模型

## 環境安裝

### 1. 安裝 Python
建議使用 Python 3.9 或以上版本

### 2. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 3. 下載 YOLO 模型
將 `yolov9e.pt` 模型檔案放置於專案根目錄

---

## 使用方法

### 圖片處理 (main_image.py)

處理單張圖片並生成追蹤結果和熱力圖：

```bash
python main_image.py --image <圖片路徑> --config <場域配置>
```

**參數說明：**
- `--image`：輸入圖片路徑
- `--config`：場域配置名稱 (見下方場域對應)
- `--output_prefix`：輸出檔案前綴名

**範例：**
```bash
python main_image.py --image input/images/1104/1.jpg --config hall_1
```

**輸出檔案：**
- `<prefix>_track_out.png`：追蹤結果圖
- `<prefix>_heat_out.png`：熱力圖
- `<prefix>_count.txt`：人數統計
- `<prefix>_result.json`：JSON 格式結果

---

## LSTM 人流預測

### 步驟 1: 批次人物檢測 (batch_detection.py)

對 `input/images/日期/` 資料夾中的圖片進行批次人物檢測：

```bash
python batch_detection.py
```

**配置參數（在 `main()` 中修改）：**
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `input_base_dir` | `"input/images"` | 輸入目錄 |
| `output_dir` | `"output"` | 輸出目錄 |
| `space_config` | `"hall_1"` | 場域配置 |
| `start_time` | `"08:00:00"` | 每天開始時間 |
| `interval_minutes` | `5` | 時間間隔（分鐘）|
| `year` | `2025` | 年份 |

**輸出格式（JSON）：**
```json
[
  {"timestamp": "2025-11-03 08:00:00", "filename": "1.jpg", "people_count": 5},
  {"timestamp": "2025-11-03 08:05:00", "filename": "2.jpg", "people_count": 8}
]
```

---

### 步驟 2: 資料前處理 (LSTM.py)

將 JSON 資料轉換為 LSTM 訓練用的 3D Numpy 陣列：

```bash
python LSTM.py
```

**處理流程：**
1. 載入 JSON 資料
2. 時間過濾 (08:00-16:45)
3. 重採樣 & 缺失值插補
4. 特徵工程（正規化 people_count, hour, minute, weekday）
5. 時序分割（前 10 天訓練，後 5 天測試）
6. MinMaxScaler（僅在訓練資料上擬合）
7. 滑動視窗序列生成

**輸出：**
- `lstm_data.npy`：3D 陣列 `(n_days, 106, 4)`

---

### 步驟 3: 模型訓練與評估 (train_lstm.py)

訓練 LSTM 模型並評估預測效果：

```bash
python train_lstm.py
```

**模型架構：**
```
LSTM(32) → Dropout(0.2) → Dense(1)
```

**訓練配置：**
- Optimizer: Adam
- Loss: MSE
- EarlyStopping: patience=10
- Batch Size: 32
- Lookback: 12 步（1 小時）

**輸出檔案：**
- `output/lstm_model.keras`：訓練好的模型
- `output/lstm_results.png`：Loss 曲線 + 預測結果圖
- `output/lstm_daily_results.png`：每日預測對比圖

**評估指標：**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

---

## 場域對應

| 場域名稱 | 配置名稱 | 參考圖片 |
|---------|---------|---------|
| 穿堂1 | `hall_1` | ![hall_1](field/hall_A.jpg) |
| 穿堂2 | `hall_2` | ![hall_2](field/hall_B.jpg) |
| 階梯教室 | `lecture_room` | ![lecture_room](field/lecture_room.jpg) |
| 電腦教室1 | `computer_room_1` | ![computer_room_1](field/computer_room_A.jpg) |
| 電腦教室2 | `computer_room_2` | ![computer_room_2](field/computer_room_B.jpg) |

---

## 專案結構

```
mcas_carbon_v2/
├── main_image.py          # 圖片處理主程式
├── main.py                # 影片處理主程式
├── person_detection.py    # 人物檢測模組
├── heatmap.py             # 熱力圖生成模組
├── config.py              # 場域配置檔
├── rename_files.py        # 圖片重新命名工具
├── batch_detection.py     # 批次人物檢測（產生 JSON）
├── LSTM.py                # LSTM 資料前處理
├── train_lstm.py          # LSTM 模型訓練與評估
├── requirements.txt       # 依賴套件清單
├── yolov9e.pt             # YOLO 模型檔案
├── lstm_data.npy          # 預處理後的 LSTM 資料
├── input/                 # 輸入資料夾
│   └── images/            # 圖片資料（按日期分資料夾）
└── output/                # 輸出資料夾
    ├── *.json             # 人物檢測結果
    ├── lstm_model.keras   # 訓練好的 LSTM 模型
    └── *.png              # 視覺化圖表
```

