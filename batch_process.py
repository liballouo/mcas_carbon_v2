"""
批次影像辨識腳本
用於處理指定場域的所有影片檔案

使用方式:
    python batch_process.py --location 穿堂1 --config hall_1
    python batch_process.py --location 穿堂2 --config hall_2
    python batch_process.py --location 電腦教室1 --config computer_room_1
    python batch_process.py --location 電腦教室2 --config computer_room_2
    python batch_process.py --location 階梯教室 --config lecture_room

可用的 config 選項:
    - hall_1 (穿堂A區)
    - hall_2 (穿堂B區)
    - computer_room_1 (電腦教室A區)
    - computer_room_2 (電腦教室B區)
    - lecture_room (階梯教室)
"""

import os
import glob
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ============================================================
# 設定區 - 可依需求調整
# ============================================================

# 場域與 config 的對應關係
# 如需新增場域，請在此處添加
LOCATION_CONFIG_MAP = {
    "穿堂1": "hall_1",
    "穿堂2": "hall_2",
    "電腦教室1": "computer_room_1",
    "電腦教室2": "computer_room_2",
    "階梯教室": "lecture_room",
    # 新增場域範例:
    # "新場域名稱": "對應的config",
}

# 預設的模型路徑
DEFAULT_MODEL = "yolov9e.pt"

# 預設的影片資料夾模式 (glob pattern)
# 這會搜索所有以 202512 開頭的資料夾
DEFAULT_FOLDER_PATTERN = "202512*"


# ============================================================
# 主程式
# ============================================================

def find_video_folders(base_dir: str, pattern: str = DEFAULT_FOLDER_PATTERN) -> list:
    """
    尋找所有符合模式的資料夾
    
    Args:
        base_dir: 基礎目錄路徑
        pattern: 資料夾名稱模式 (glob pattern)
    
    Returns:
        資料夾路徑列表
    """
    search_path = os.path.join(base_dir, pattern)
    folders = glob.glob(search_path)
    folders = [f for f in folders if os.path.isdir(f)]
    return sorted(folders)


def find_videos_by_location(folder: str, location: str) -> list:
    """
    在指定資料夾中尋找特定場域的影片
    
    Args:
        folder: 資料夾路徑
        location: 場域名稱 (如 "穿堂1")
    
    Returns:
        影片檔案路徑列表
    """
    videos = []
    
    # 搜索當前資料夾和所有子資料夾中的影片
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 檢查檔案是否為影片且以指定場域名稱開頭
            if file.startswith(location) and file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                videos.append(os.path.join(root, file))
    
    return sorted(videos)


def process_video(video_path: str, config: str, output_dir: str, 
                  model: str = DEFAULT_MODEL, enable_pose: bool = False,
                  gemini_api_key: str = None) -> bool:
    """
    處理單一影片檔案
    
    Args:
        video_path: 影片檔案路徑
        config: 空間配置名稱
        output_dir: 輸出目錄
        model: YOLO 模型路徑
        enable_pose: 是否啟用姿勢識別
        gemini_api_key: Gemini API 金鑰
    
    Returns:
        是否處理成功
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_prefix = os.path.join(output_dir, video_name)
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立命令
    cmd = [
        "python", "main.py",
        "--video", video_path,
        "--config", config,
        "--model", model,
        "--output_prefix", output_prefix,
    ]
    
    if enable_pose:
        cmd.append("--enable_pose")
        if gemini_api_key:
            cmd.extend(["--gemini_api_key", gemini_api_key])
    
    print(f"\n{'='*60}")
    print(f"處理影片: {video_path}")
    print(f"輸出目錄: {output_dir}")
    print(f"使用配置: {config}")
    print(f"{'='*60}")
    
    try:
        # 取得 main.py 所在的目錄
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 執行命令
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            check=True,
            capture_output=False,  # 讓輸出直接顯示在終端
        )
        print(f"✓ 處理完成: {video_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 處理失敗: {video_name}")
        print(f"  錯誤: {e}")
        return False
    except Exception as e:
        print(f"✗ 發生未預期錯誤: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='批次處理指定場域的影片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--location', 
        type=str, 
        required=True,
        help='場域名稱 (如: 穿堂1, 穿堂2, 電腦教室1, 電腦教室2, 階梯教室)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        choices=['hall_1', 'hall_2', 'computer_room_1', 'computer_room_2', 'lecture_room', '2'],
        help='空間配置名稱 (若不指定，會根據場域名稱自動選擇)'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='..',
        help='影片資料夾所在的基礎目錄 (default: 上層目錄)'
    )
    
    parser.add_argument(
        '--folder_pattern',
        type=str,
        default=DEFAULT_FOLDER_PATTERN,
        help=f'資料夾名稱模式 (default: {DEFAULT_FOLDER_PATTERN})'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='輸出目錄 (default: ./output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'YOLO 模型路徑 (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--enable_pose',
        action='store_true',
        help='是否啟用姿勢識別'
    )
    
    parser.add_argument(
        '--gemini_api_key',
        type=str,
        help='Gemini API 金鑰 (用於姿勢識別)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='僅顯示將要處理的檔案，不實際執行'
    )
    
    args = parser.parse_args()
    
    # 決定 config
    if args.config:
        config = args.config
    elif args.location in LOCATION_CONFIG_MAP:
        config = LOCATION_CONFIG_MAP[args.location]
        print(f"自動選擇配置: {config} (根據場域名稱 '{args.location}')")
    else:
        print(f"錯誤: 未知的場域名稱 '{args.location}'，請使用 --config 手動指定配置")
        print(f"已知的場域: {list(LOCATION_CONFIG_MAP.keys())}")
        return
    
    # 取得腳本目錄作為參考點
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, args.base_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir, args.location))
    
    print(f"\n{'='*60}")
    print(f"批次影像辨識腳本")
    print(f"{'='*60}")
    print(f"場域名稱: {args.location}")
    print(f"空間配置: {config}")
    print(f"基礎目錄: {base_dir}")
    print(f"資料夾模式: {args.folder_pattern}")
    print(f"輸出目錄: {output_dir}")
    print(f"{'='*60}")
    
    # 尋找所有符合模式的資料夾
    folders = find_video_folders(base_dir, args.folder_pattern)
    
    if not folders:
        print(f"警告: 在 '{base_dir}' 中未找到符合 '{args.folder_pattern}' 的資料夾")
        return
    
    print(f"\n找到 {len(folders)} 個資料夾")
    
    # 收集所有影片
    all_videos = []
    for folder in folders:
        videos = find_videos_by_location(folder, args.location)
        all_videos.extend(videos)
    
    if not all_videos:
        print(f"警告: 未找到任何以 '{args.location}' 開頭的影片檔案")
        return
    
    print(f"找到 {len(all_videos)} 個 '{args.location}' 影片檔案:")
    for i, video in enumerate(all_videos, 1):
        print(f"  {i}. {video}")
    
    # Dry run 模式
    if args.dry_run:
        print("\n[Dry Run 模式] 以上是將要處理的檔案，實際執行請移除 --dry_run 參數")
        return
    
    # 確認執行
    print(f"\n準備處理 {len(all_videos)} 個影片...")
    
    # 開始處理
    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    
    for video in all_videos:
        if process_video(
            video_path=video,
            config=config,
            output_dir=output_dir,
            model=args.model,
            enable_pose=args.enable_pose,
            gemini_api_key=args.gemini_api_key
        ):
            success_count += 1
        else:
            fail_count += 1
    
    # 顯示結果摘要
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"批次處理完成")
    print(f"{'='*60}")
    print(f"總計影片數: {len(all_videos)}")
    print(f"成功: {success_count}")
    print(f"失敗: {fail_count}")
    print(f"耗時: {duration}")
    print(f"輸出目錄: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
