"""
重新命名 input/images/[日期]/穿堂1/ 中的所有圖片檔案
按照資料夾時間順序排序後，從 1 開始連續編號

例如：
- 穿堂1_CCTV_CCTV_20251104075959_20251104083006_826688170/ 中會有 1~6
- 穿堂1_CCTV_CCTV_20251104083006_20251104094820_826818968/ 中會從 7 開始
"""

import os
from pathlib import Path


def rename_images_in_date_folder(date_folder: str):
    """
    重新命名指定日期資料夾內 穿堂1 下所有圖片，並儲存到日期根目錄
    
    Args:
        date_folder: 日期資料夾名稱，例如 "1104"
    """
    base_path = Path("input/images") / date_folder 
    output_path = Path("input/images") / date_folder  # 輸出到日期根目錄
    
    if not base_path.exists():
        print(f"錯誤：找不到資料夾 {base_path}")
        return
    
    # 取得所有 CCTV 子資料夾並按名稱排序（名稱中包含時間戳記）
    cctv_folders = sorted([
        folder for folder in base_path.iterdir() 
        if folder.is_dir() and folder.name.startswith("穿堂1_CCTV")
    ])
    
    if not cctv_folders:
        print(f"錯誤：在 {base_path} 中找不到 CCTV 資料夾")
        return
    
    print(f"開始處理日期 {date_folder}...")
    print(f"找到 {len(cctv_folders)} 個 CCTV 資料夾")
    print(f"輸出目錄：{output_path}")
    
    # 全域計數器，用於連續編號
    counter = 1
    
    for cctv_folder in cctv_folders:
        # 取得該資料夾內所有圖片檔案並排序
        image_files = sorted([
            f for f in cctv_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        if not image_files:
            print(f"  跳過空資料夾：{cctv_folder.name}")
            continue
        
        folder_start = counter
        
        # 將每個圖片檔案重新命名並移動到輸出目錄
        for img_file in image_files:
            final_name = output_path / f"{counter}{img_file.suffix}"
            os.rename(img_file, final_name)
            counter += 1
        
        folder_end = counter - 1
        print(f"  {cctv_folder.name}: 編號 {folder_start}~{folder_end}")
    
    print(f"完成！共處理 {counter - 1} 張圖片，儲存於 {output_path}")


def main():
    """主程式：處理指定日期資料夾"""
    import sys
    
    if len(sys.argv) < 2:
        # 顯示可用的日期資料夾
        images_path = Path("input/images")
        if images_path.exists():
            date_folders = sorted([
                folder.name for folder in images_path.iterdir() 
                if folder.is_dir()
            ])
            print(f"可用的日期資料夾：{date_folders}")
        print("\n使用方式：python rename_files.py <日期>")
        print("範例：python rename_files.py 1104")
        return
    
    date_folder = sys.argv[1]
    rename_images_in_date_folder(date_folder)


if __name__ == "__main__":
    main()
