import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import argparse
import os
import json

# 導入我們的新模組
from person_detection import PersonDetector
from heatmap import HeatmapGenerator
from pose_recognition import PoseRecognizer

def parse_args():
    parser = argparse.ArgumentParser(description='人流追蹤和熱力圖生成')
    parser.add_argument('--image', type=str, help='輸入單張圖片路徑 (如: test.png)')
    parser.add_argument('--config', type=str, choices=['2', 'lecture_room', 'computer_room_A', 'computer_room_B', 'hall_A', 'hall_B'], 
                      help='選擇預設的點位配置')
    parser.add_argument('--model', type=str, default='yolov9e.pt', 
                      help='YOLO模型路徑 (default: yolov9e.pt)')
    parser.add_argument('--output_prefix', type=str, help='輸出文件的前綴名')
    parser.add_argument('--conf', type=float, default=0.25, 
                      help='檢測置信度閾值 (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.5, 
                      help='IOU閾值 (default: 0.5)')
    parser.add_argument('--heat_width', type=int, default=500, help='熱力圖寬度 (default: 500)')
    parser.add_argument('--heat_height', type=int, default=500, help='熱力圖高度 (default: 500)')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    # 初始化模組
    print("正在初始化模組...")
    
    # 1. 初始化人物檢測器
    detector = PersonDetector(model_path=args.model)
    
    # 2. 初始化熱力圖生成器
    heatmap_gen = HeatmapGenerator(width=args.heat_width, height=args.heat_height)
    
    # 單張圖片處理流程
    print(f"正在處理圖片: {args.image}")
    
    # 設置輸出文件名
    output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.image))[0]
    track_output = f"{output_prefix}_track_out.png"
    heat_output = f"{output_prefix}_heat_out.png"
    count_file = f"{output_prefix}_count.txt"
    json_output = f"{output_prefix}_result.json"
    
    try:
        # 1. 人物檢測
        print("正在進行人物檢測...")
        total_count, person_coordinates, processed_image = detector.detect_persons_full_image(
            image_source=args.image,
            space_config=args.config
        )
        print(f"檢測到 {total_count} 個人")
        
        # 2. 生成熱力圖
        print("正在生成熱力圖...")
        heatmap_image, transformed_coordinates = heatmap_gen.generate_complete_heatmap(
            person_coordinates=person_coordinates,
            space_config=args.config,
            show_markers=True,
            show_count=True
        )
        
        # 4. 保存結果
        print("正在保存結果...")
        cv2.imwrite(track_output, processed_image)
        cv2.imwrite(heat_output, heatmap_image)
        
        # 創建人數記錄文件
        with open(count_file, 'w', encoding='utf-8') as f:
            f.write("時間(秒),人數")
            f.write("\n")
            
            # 記錄數據
            f.write(f"0.0,{total_count}")

            f.write("\n")
        
        # === 新增 JSON 輸出部分 ===
        json_output = f"{output_prefix}_result.json"
        
        # 獲取空間配置的四個角點
        space_points = heatmap_gen.get_space_points(args.config)
        project_face_coords = [
            (float(space_points['topleft'][0]), float(space_points['topleft'][1])),
            (float(space_points['topright'][0]), float(space_points['topright'][1])),
            (float(space_points['bottomright'][0]), float(space_points['bottomright'][1])),
            (float(space_points['bottomleft'][0]), float(space_points['bottomleft'][1]))
        ]

        results = {
            "results": {
                "time": 0,
                "num_of_people": total_count,
                "project_face_coords": project_face_coords,
                "points": [(float(x), float(y)) for (x, y) in person_coordinates]
            }
        }

        with open(json_output, 'w', encoding='utf-8') as jf:
            json.dump(results, jf, ensure_ascii=False, indent=4)

        print(f"JSON結果已保存至: {json_output}")


        # 輸出結果信息
        print(f"\n處理完成！")
        print(f"檢測到總人數: {total_count}")
        print(f"追蹤圖片已保存至: {track_output}")
        print(f"熱力圖已保存至: {heat_output}")
        print(f"統計數據已保存至: {count_file}")
        
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        return

if __name__ == "__main__":
    main()