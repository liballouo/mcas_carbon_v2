"""
影片影格與JSON資料提取腳本

功能：
    1. 讀取1fps影片檔案與對應的JSON預測檔案
    2. 將每一影格儲存為獨立的圖片檔案
    3. 將每個對應的JSON物件儲存為獨立的JSON檔案
    4. 支援處理 output 資料夾中各場域的影片
    5. 支援 GPU 加速 (需要 CUDA 支援的 OpenCV)

使用方式：
    # 只儲存圖片
    python extract_frames.py --video 影片路徑.mp4 --output 輸出資料夾 --save_images
    
    # 只儲存JSON (需要同時提供影片和JSON檔案)
    python extract_frames.py --video 影片路徑.mp4 --json JSON路徑.json --output 輸出資料夾 --save_json
    
    # 同時儲存圖片和JSON
    python extract_frames.py --video 影片路徑.mp4 --json JSON路徑.json --output 輸出資料夾 --save_images --save_json
    
    # 使用 GPU 加速
    python extract_frames.py --video 影片路徑.mp4 --output 輸出資料夾 --save_images --use_gpu
    
    # 批次處理所有場域的影片
    python extract_frames.py --batch --save_images --save_json

命名規則：
    {場域名稱}_{YYYYMMDDHHMMSS}.jpg
    {場域名稱}_{YYYYMMDDHHMMSS}.json
"""

import os
import cv2
import json
import argparse
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm


# ============================================================
# GPU 支援檢測
# ============================================================

def check_gpu_available() -> bool:
    """
    檢查是否有可用的 CUDA GPU
    
    Returns:
        是否有可用的 GPU
    """
    try:
        # 檢查 OpenCV 是否有 CUDA 支援
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return True
    except:
        pass
    return False


def get_gpu_info() -> str:
    """
    取得 GPU 資訊
    
    Returns:
        GPU 資訊字串
    """
    try:
        if check_gpu_available():
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            return f"找到 {device_count} 個 CUDA GPU"
    except:
        pass
    return "無可用的 CUDA GPU"


GPU_AVAILABLE = check_gpu_available()


# ============================================================
# 設定區 - 可依需求調整
# ============================================================

# 預設的檔案命名前綴
DEFAULT_PREFIX = "frame"

# 零填充位數 (6位數可支援約1百萬張圖片，約11天的1fps影片)
ZERO_PADDING = 6

# 圖片格式
IMAGE_FORMAT = "jpg"

# JPEG 品質 (1-100)
JPEG_QUALITY = 95

# 要處理的場域列表
LOCATIONS = ["穿堂1", "穿堂2", "階梯教室", "電腦教室1", "電腦教室2"]

# 輸出資料夾名稱（在每個影片的同一資料夾中建立）
OUTPUT_FOLDER_NAME = "frames"


# ============================================================
# 主程式
# ============================================================

def parse_video_filename(video_path: str) -> dict:
    """
    解析影片檔名，提取場域名稱和起始時間
    
    檔名格式: 穿堂1_CCTV_CCTV_20251202142602_20251202154241_11260475_track_out.mp4
    其中 20251202142602 表示 2025年12月02日14時26分02秒
    
    Args:
        video_path: 影片檔案路徑
    
    Returns:
        包含場域名稱、起始時間等資訊的字典
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 移除 _track_out 或 _heat_out 後綴
    basename = re.sub(r'_(track_out|heat_out)$', '', basename)
    
    # 使用正則表達式提取資訊
    # 格式: {場域名稱}_CCTV_CCTV_{起始時間}_{結束時間}_{其他}
    pattern = r'^(.+?)_CCTV_CCTV_(\d{14})_(\d{14})_(\d+)$'
    match = re.match(pattern, basename)
    
    if match:
        location = match.group(1)
        start_time_str = match.group(2)
        end_time_str = match.group(3)
        
        # 解析時間字串
        start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
        end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")
        
        return {
            "location": location,
            "start_time": start_time,
            "end_time": end_time,
            "start_time_str": start_time_str,
            "base_name": basename,
            "prefix": f"{location}_{start_time_str}"
        }
    else:
        # 如果無法解析，使用預設值
        return {
            "location": basename,
            "start_time": None,
            "end_time": None,
            "start_time_str": "",
            "base_name": basename,
            "prefix": basename
        }


def format_timestamp(base_time: datetime, frame_index: int) -> str:
    """
    根據基準時間和影格索引計算時間戳記
    
    由於影片為1fps，每個影格對應1秒
    
    Args:
        base_time: 影片起始時間
        frame_index: 影格索引 (0-indexed)
    
    Returns:
        格式化的時間字串 YYYYMMDDHHMMSS
    """
    if base_time is None:
        return ""
    
    frame_time = base_time + timedelta(seconds=frame_index)
    return frame_time.strftime("%Y%m%d%H%M%S")


def extract_frames_from_video(
    video_path: str,
    json_path: str = None,
    output_dir: str = None,
    prefix: str = None,
    use_timestamp: bool = True,
    save_images: bool = True,
    save_json: bool = True,
    use_gpu: bool = False
) -> bool:
    """
    從影片中提取影格並儲存對應的JSON資料
    
    Args:
        video_path: 影片檔案路徑
        json_path: JSON預測檔案路徑 (如果 save_json=True 則必須提供)
        output_dir: 輸出目錄
        prefix: 檔案命名前綴 (如果為None則自動從影片檔名解析)
        use_timestamp: 是否在檔名中使用時間戳記
        save_images: 是否儲存圖片
        save_json: 是否儲存JSON
        use_gpu: 是否使用 GPU 加速
    
    Returns:
        是否成功處理
    """
    # 檢查至少要儲存一種類型
    if not save_images and not save_json:
        print("錯誤: 必須至少選擇儲存圖片或JSON其中之一 (--save_images 或 --save_json)")
        return False
    
    # 檢查檔案是否存在
    if not os.path.exists(video_path):
        print(f"錯誤: 影片檔案不存在 - {video_path}")
        return False
    
    # 如果需要儲存JSON，則必須提供JSON檔案路徑
    if save_json:
        if json_path is None:
            print("錯誤: 儲存JSON需要提供JSON檔案路徑 (--json)")
            return False
        if not os.path.exists(json_path):
            print(f"錯誤: JSON檔案不存在 - {json_path}")
            return False
    
    # 解析影片檔名
    video_info = parse_video_filename(video_path)
    
    # 決定使用的前綴
    if prefix is None:
        prefix = video_info["prefix"]
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取JSON檔案（如果需要）
    results = None
    json_count = 0
    if save_json and json_path:
        print(f"讀取JSON檔案: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"錯誤: JSON解析失敗 - {e}")
            return False
        
        # 確認JSON結構
        if isinstance(json_data, dict) and "results" in json_data:
            results = json_data["results"]
        elif isinstance(json_data, list):
            results = json_data
        else:
            print(f"錯誤: 不支援的JSON結構")
            return False
        
        json_count = len(results)
        print(f"JSON資料筆數: {json_count}")
    
    # 檢查 GPU 支援
    actual_use_gpu = False
    if use_gpu:
        if GPU_AVAILABLE:
            actual_use_gpu = True
            print(f"GPU 加速: 已啟用 ({get_gpu_info()})")
        else:
            print("警告: 無法使用 GPU 加速，將使用 CPU 處理")
            print("  提示: 需要安裝支援 CUDA 的 OpenCV (opencv-contrib-python 或自行編譯)")
    
    # 開啟影片
    print(f"開啟影片: {video_path}")
    
    # 使用 GPU 加速時，嘗試使用硬體解碼
    if actual_use_gpu:
        # 嘗試使用 CUDA 視訊解碼器
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        # 設定硬體加速
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"錯誤: 無法開啟影片 - {video_path}")
        return False
    
    # 取得影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"影片資訊:")
    print(f"  - 總影格數: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - 解析度: {width}x{height}")
    
    # 決定處理數量
    if save_json and results:
        if total_frames != json_count:
            print(f"警告: 影格數 ({total_frames}) 與 JSON資料筆數 ({json_count}) 不符!")
            print(f"將使用較小的數量: {min(total_frames, json_count)}")
        process_count = min(total_frames, json_count)
    else:
        process_count = total_frames
    
    # 顯示輸出模式
    output_mode = []
    if save_images:
        output_mode.append("圖片")
    if save_json:
        output_mode.append("JSON")
    
    print(f"\n開始處理...")
    print(f"輸出目錄: {output_dir}")
    print(f"檔案前綴: {prefix}")
    print(f"輸出類型: {' + '.join(output_mode)}")
    if actual_use_gpu:
        print(f"處理模式: GPU 加速")
    
    # 處理每一影格
    success_count = 0
    fail_count = 0
    
    # GPU 加速時使用的 CUDA 編碼器
    gpu_encoder = None
    if actual_use_gpu and save_images:
        try:
            # 嘗試建立 GPU 編碼器
            gpu_encoder = cv2.cudacodec.createVideoWriter if hasattr(cv2, 'cudacodec') else None
        except:
            gpu_encoder = None
    
    for frame_idx in tqdm(range(process_count), desc="處理進度"):
        # 讀取影格
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告: 無法讀取影格 {frame_idx}")
            fail_count += 1
            continue
        
        # 計算時間戳記 - 新命名格式: {場域名稱}_{YYYYMMDDHHMMSS}
        if use_timestamp and video_info["start_time"]:
            timestamp = format_timestamp(video_info["start_time"], frame_idx)
            filename_base = f"{video_info['location']}_{timestamp}"
        else:
            filename_base = f"{video_info['location']}_{str(frame_idx + 1).zfill(ZERO_PADDING)}"
        
        # 儲存圖片
        if save_images:
            image_path = os.path.join(output_dir, f"{filename_base}.{IMAGE_FORMAT}")
            if actual_use_gpu:
                try:
                    # 上傳到 GPU 並編碼
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    # GPU 編碼 JPEG
                    _, encoded = cv2.imencode(f'.{IMAGE_FORMAT}', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    with open(image_path, 'wb') as f:
                        f.write(encoded.tobytes())
                except Exception as e:
                    # 如果 GPU 編碼失敗，回退到 CPU
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                # 使用 imencode + 檔案寫入來支援中文路徑
                _, encoded = cv2.imencode(f'.{IMAGE_FORMAT}', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                with open(image_path, 'wb') as f:
                    f.write(encoded.tobytes())
        
        # 儲存JSON
        if save_json and results:
            # 準備JSON資料，加入時間戳記
            frame_json = results[frame_idx].copy()
            if video_info["start_time"]:
                frame_json["timestamp"] = format_timestamp(video_info["start_time"], frame_idx)
            
            json_output_path = os.path.join(output_dir, f"{filename_base}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(frame_json, f, ensure_ascii=False, indent=2)
        
        success_count += 1
    
    # 釋放資源
    cap.release()
    
    print(f"\n處理完成!")
    print(f"  - 成功: {success_count}")
    print(f"  - 失敗: {fail_count}")
    print(f"  - 輸出目錄: {output_dir}")
    print(f"  - 輸出類型: {' + '.join(output_mode)}")
    
    return True


def find_video_json_pairs_in_location(output_dir: str, location: str) -> list:
    """
    在指定場域資料夾中尋找影片和JSON配對
    
    Args:
        output_dir: output 資料夾路徑
        location: 場域名稱 (如 "穿堂1")
    
    Returns:
        (video_path, json_path) 配對列表
    """
    pairs = []
    location_dir = os.path.join(output_dir, location)
    
    if not os.path.exists(location_dir):
        print(f"警告: 場域資料夾不存在 - {location_dir}")
        return pairs
    
    # 尋找所有 *_track_out.mp4 影片
    for file in os.listdir(location_dir):
        if file.endswith('_track_out.mp4'):
            video_path = os.path.join(location_dir, file)
            
            # 對應的JSON檔案名稱
            # 從 xxx_track_out.mp4 變成 xxx_result.json
            json_name = file.replace('_track_out.mp4', '_result.json')
            json_path = os.path.join(location_dir, json_name)
            
            if os.path.exists(json_path):
                pairs.append((video_path, json_path))
            else:
                print(f"警告: 找不到對應的JSON檔案 - {json_path}")
    
    return sorted(pairs)


def batch_process(output_dir: str, locations: list = None, 
                  save_images: bool = True, save_json: bool = True, 
                  use_gpu: bool = False) -> None:
    """
    批次處理所有場域的影片
    
    Args:
        output_dir: output 資料夾路徑
        locations: 要處理的場域列表
        save_images: 是否儲存圖片
        save_json: 是否儲存JSON
        use_gpu: 是否使用 GPU 加速
    """
    if locations is None:
        locations = LOCATIONS
    
    print(f"\n{'='*60}")
    print(f"批次處理模式")
    print(f"{'='*60}")
    print(f"來源目錄: {output_dir}")
    print(f"處理場域: {', '.join(locations)}")
    
    # 收集所有影片和JSON配對
    all_pairs = []
    for location in locations:
        pairs = find_video_json_pairs_in_location(output_dir, location)
        all_pairs.extend([(location, video, json_file) for video, json_file in pairs])
    
    if not all_pairs:
        print(f"警告: 未找到任何影片和JSON配對")
        return
    
    print(f"\n找到 {len(all_pairs)} 個影片/JSON配對:")
    for i, (location, video, json_file) in enumerate(all_pairs, 1):
        print(f"  {i}. [{location}] {os.path.basename(video)}")
    
    # 處理每個配對
    success_count = 0
    fail_count = 0
    
    for location, video_path, json_path in all_pairs:
        video_info = parse_video_filename(video_path)
        
        # 在影片所在的資料夾中建立輸出子資料夾
        # 使用影片的 base_name 作為資料夾名稱
        video_dir = os.path.dirname(video_path)
        output_subdir = os.path.join(video_dir, video_info["base_name"])
        
        print(f"\n{'='*60}")
        print(f"處理: {os.path.basename(video_path)}")
        print(f"輸出到: {output_subdir}")
        print(f"{'='*60}")
        
        if extract_frames_from_video(
            video_path, 
            json_path if save_json else None, 
            output_subdir,
            save_images=save_images,
            save_json=save_json,
            use_gpu=use_gpu
        ):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批次處理完成")
    print(f"{'='*60}")
    print(f"成功: {success_count}")
    print(f"失敗: {fail_count}")


def main():
    parser = argparse.ArgumentParser(
        description='從影片中提取影格並儲存對應的JSON資料',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 單一影片處理模式
    parser.add_argument(
        '--video',
        type=str,
        help='影片檔案路徑'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        help='JSON預測檔案路徑'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='輸出目錄 (default: ./output)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default=None,
        help='檔案命名前綴 (若不指定，會從影片檔名自動產生)'
    )
    
    # 批次處理模式
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批次處理所有場域的影片'
    )
    
    parser.add_argument(
        '--locations',
        type=str,
        nargs='+',
        default=None,
        help=f'要處理的場域列表 (default: {LOCATIONS})'
    )
    
    parser.add_argument(
        '--no_timestamp',
        action='store_true',
        help='不在檔名中加入時間戳記'
    )
    
    # 輸出類型選項
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='儲存圖片檔案 (.jpg)'
    )
    
    parser.add_argument(
        '--save_json',
        action='store_true',
        help='儲存JSON檔案 (需要同時提供 --json 或使用 --batch)'
    )
    
    # GPU 加速選項
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help=f'使用 GPU 加速處理 (目前狀態: {"可用" if GPU_AVAILABLE else "不可用"})'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='僅顯示將要處理的檔案，不實際執行'
    )
    
    args = parser.parse_args()
    
    # 如果沒有指定輸出類型，且沒有 JSON 檔案，預設只儲存圖片
    # 如果沒有指定任何輸出類型，預設儲存圖片（保持向後相容）
    save_images = args.save_images
    save_json = args.save_json
    
    # 如果都沒指定，預設儲存圖片
    if not save_images and not save_json:
        save_images = True
        print("提示: 未指定輸出類型，預設儲存圖片 (--save_images)")
    
    # 取得腳本目錄作為參考點
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output))
    
    if args.batch:
        # 批次處理模式
        locations = args.locations if args.locations else LOCATIONS
        
        if args.dry_run:
            print(f"[Dry Run] 搜尋 {output_dir} 中的影片...")
            print(f"輸出類型: {'圖片' if save_images else ''}{' + ' if save_images and save_json else ''}{'JSON' if save_json else ''}")
            print(f"GPU 加速: {'啟用' if args.use_gpu else '停用'}")
            for location in locations:
                pairs = find_video_json_pairs_in_location(output_dir, location)
                for video, json_file in pairs:
                    video_info = parse_video_filename(video)
                    video_dir = os.path.dirname(video)
                    output_subdir = os.path.join(video_dir, video_info["base_name"])
                    print(f"  [{location}]")
                    print(f"    影片: {video}")
                    if save_json:
                        print(f"    JSON: {json_file}")
                    print(f"    輸出: {output_subdir}")
                    print()
            return
        
        batch_process(
            output_dir, 
            locations,
            save_images=save_images,
            save_json=save_json,
            use_gpu=args.use_gpu
        )
    
    elif args.video:
        # 單一影片處理模式
        video_path = os.path.abspath(args.video)
        json_path = os.path.abspath(args.json) if args.json else None
        
        # 如果要儲存 JSON 但沒有提供 JSON 檔案
        if save_json and not json_path:
            print("錯誤: 儲存JSON需要提供 --json 參數")
            return
        
        if args.dry_run:
            print(f"[Dry Run] 將處理:")
            print(f"  影片: {video_path}")
            if json_path:
                print(f"  JSON: {json_path}")
            print(f"  輸出: {output_dir}")
            print(f"  輸出類型: {'圖片' if save_images else ''}{' + ' if save_images and save_json else ''}{'JSON' if save_json else ''}")
            print(f"  GPU 加速: {'啟用' if args.use_gpu else '停用'}")
            return
        
        extract_frames_from_video(
            video_path,
            json_path,
            output_dir,
            prefix=args.prefix,
            use_timestamp=not args.no_timestamp,
            save_images=save_images,
            save_json=save_json,
            use_gpu=args.use_gpu
        )
    
    else:
        parser.print_help()
        print("\n請使用 --video 指定影片，或使用 --batch 進行批次處理")
        print("\n範例:")
        print("  # 只儲存圖片")
        print("  python extract_frames.py --batch --save_images")
        print("")
        print("  # 同時儲存圖片和JSON")
        print("  python extract_frames.py --batch --save_images --save_json")
        print("")
        print("  # 使用GPU加速儲存圖片")
        print("  python extract_frames.py --batch --save_images --use_gpu")


if __name__ == "__main__":
    main()
