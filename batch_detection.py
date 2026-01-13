"""
批次影像人物檢測腳本
對 input/images/日期 中的所有圖片進行人物檢測，並將結果儲存為 JSON 格式
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict
from person_detection import PersonDetector


def natural_sort_key(filename: str) -> int:
    """
    自然排序鍵函數，用於正確排序數字檔名
    例如: 1.jpg, 2.jpg, 10.jpg 而不是 1.jpg, 10.jpg, 2.jpg
    """
    # 提取檔名中的數字部分
    basename = os.path.splitext(filename)[0]
    try:
        return int(basename)
    except ValueError:
        return 0


def process_date_folder(
    detector: PersonDetector,
    date_folder: str,
    space_config: str = 'hall_1',
    start_time: str = "08:00:00",
    interval_minutes: int = 5,
    year: int = 2024
) -> List[Dict]:
    """
    處理單個日期資料夾中的所有圖片
    
    Args:
        detector: PersonDetector 實例
        date_folder: 日期資料夾的完整路徑
        space_config: 空間配置名稱
        start_time: 開始時間 (HH:MM:SS 格式)
        interval_minutes: 每張圖片的時間間隔（分鐘）
        year: 年份
        
    Returns:
        包含所有檢測結果的列表
    """
    results = []
    
    # 獲取資料夾名稱（日期）
    date_str = os.path.basename(date_folder)
    
    # 解析日期（格式：MMDD）
    if len(date_str) == 4:
        month = int(date_str[:2])
        day = int(date_str[2:])
    else:
        print(f"無法解析日期格式: {date_str}")
        return results
    
    # 構建基準日期時間
    base_date = datetime(year, month, day)
    time_parts = start_time.split(':')
    base_datetime = base_date.replace(
        hour=int(time_parts[0]),
        minute=int(time_parts[1]),
        second=int(time_parts[2]) if len(time_parts) > 2 else 0
    )
    
    # 獲取所有圖片檔案
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in os.listdir(date_folder)
        if f.lower().endswith(image_extensions)
    ]
    
    # 按數字自然排序
    image_files.sort(key=natural_sort_key)
    
    print(f"\n處理日期資料夾: {date_str}")
    print(f"找到 {len(image_files)} 張圖片")
    print(f"開始時間: {base_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    for i, filename in enumerate(image_files):
        # 計算當前圖片的時間戳
        current_time = base_datetime + timedelta(minutes=i * interval_minutes)
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 圖片完整路徑
        image_path = os.path.join(date_folder, filename)
        
        try:
            # 執行人物檢測
            people_count, _, _ = detector.detect_persons_full_image(
                image_source=image_path,
                space_config=space_config
            )
            
            # 儲存結果
            result = {
                "timestamp": timestamp,
                "filename": filename,
                "people_count": people_count
            }
            results.append(result)
            
            print(f"  [{i+1}/{len(image_files)}] {filename}: {people_count} 人 @ {timestamp}")
            
        except Exception as e:
            print(f"  [{i+1}/{len(image_files)}] {filename}: 處理失敗 - {e}")
            # 仍然記錄失敗的結果
            result = {
                "timestamp": timestamp,
                "filename": filename,
                "people_count": -1,
                "error": str(e)
            }
            results.append(result)
    
    return results


def batch_process_all_dates(
    input_base_dir: str = "input/images",
    output_dir: str = "output",
    space_config: str = 'hall_1',
    start_time: str = "08:00:00",
    interval_minutes: int = 5,
    year: int = 2024,
    specific_dates: List[str] = None
):
    """
    批次處理所有日期資料夾
    
    Args:
        input_base_dir: 輸入基底目錄
        output_dir: 輸出目錄
        space_config: 空間配置名稱
        start_time: 開始時間
        interval_minutes: 時間間隔
        year: 年份
        specific_dates: 指定要處理的日期列表，若為 None 則處理所有日期
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化檢測器
    print("初始化人物檢測器...")
    detector = PersonDetector(model_path='yolov9e.pt')
    print("檢測器初始化完成")
    
    # 獲取所有日期資料夾
    if not os.path.exists(input_base_dir):
        print(f"錯誤: 輸入目錄不存在: {input_base_dir}")
        return
    
    date_folders = [
        d for d in os.listdir(input_base_dir)
        if os.path.isdir(os.path.join(input_base_dir, d))
    ]
    
    # 如果指定了特定日期，則只處理這些日期
    if specific_dates:
        date_folders = [d for d in date_folders if d in specific_dates]
    
    date_folders.sort()
    
    print(f"\n找到 {len(date_folders)} 個日期資料夾")
    print("=" * 60)
    
    for date_folder_name in date_folders:
        date_folder_path = os.path.join(input_base_dir, date_folder_name)
        
        # 處理該日期資料夾
        results = process_date_folder(
            detector=detector,
            date_folder=date_folder_path,
            space_config=space_config,
            start_time=start_time,
            interval_minutes=interval_minutes,
            year=year
        )
        
        if results:
            # 儲存 JSON 結果
            output_filename = f"{date_folder_name}_detection_results.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n結果已儲存至: {output_path}")
            print(f"總計處理 {len(results)} 張圖片")
        
        print("=" * 60)
    
    print("\n所有處理完成！")


def main():
    """
    主程式入口
    """
    # 配置參數
    config = {
        "input_base_dir": "./input/images",  # 輸入目錄
        "output_dir": "output",             # 輸出目錄
        "space_config": "hall_1",           # 空間配置（可選：hall_1, hall_2, lecture_room, computer_room_1, computer_room_2）
        "start_time": "08:00:00",           # 每天開始時間
        "interval_minutes": 5,              # 每張圖片間隔（分鐘）
        "year": 2025,                       # 年份
        "specific_dates": None              # 指定日期列表，None 表示處理所有日期
        # 例如: ["1103", "1104"] 只處理這兩天
    }
    
    batch_process_all_dates(**config)


if __name__ == "__main__":
    main()
