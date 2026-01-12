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
    parser.add_argument('--video', type=str, help='輸入視頻文件路徑')
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
    parser.add_argument('--enable_pose', action='store_true', help='是否啟用姿勢識別')
    parser.add_argument('--gemini_api_key', type=str, help='Google Gemini API金鑰')
    args = parser.parse_args()
    if not args.video and not args.image:
        parser.error('需要提供 --video 或 --image 其中之一')
    return args

def main():
    args = parse_args()
    
    # 初始化模組
    print("正在初始化模組...")
    
    # 1. 初始化人物檢測器
    detector = PersonDetector(model_path=args.model)
    
    # 2. 初始化熱力圖生成器
    heatmap_gen = HeatmapGenerator(width=args.heat_width, height=args.heat_height)
    
    # 3. 初始化姿勢識別器（如果啟用）
    pose_recognizer = None
    if args.enable_pose:
        try:
            pose_recognizer = PoseRecognizer(api_key=args.gemini_api_key)
            print("姿勢識別器已啟用")
        except Exception as e:
            print(f"姿勢識別器初始化失敗: {e}")
            print("將繼續處理但不進行姿勢分析")
    
    # 單張圖片處理流程
    if args.image:
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
            
            # 3. 姿勢分析（如果啟用）
            pose_result = None
            if pose_recognizer and total_count > 0:
                print("正在進行姿勢分析...")
                # 讀取原始圖片用於姿勢分析
                original_image = cv2.imread(args.image)
                pose_result = pose_recognizer.analyze_poses(
                    image=original_image,
                    total_persons=total_count,
                    force_analyze=True
                )
                
                if pose_result:
                    pose_text = pose_recognizer.format_pose_result(pose_result)
                    print(f"姿勢分析結果: {pose_text}")
            
            # 4. 保存結果
            print("正在保存結果...")
            cv2.imwrite(track_output, processed_image)
            cv2.imwrite(heat_output, heatmap_image)
            
            # 創建人數記錄文件
            with open(count_file, 'w', encoding='utf-8') as f:
                f.write("時間(秒),人數")
                if pose_result:
                    f.write(",坐姿人數,站立人數,跑步人數")
                f.write("\n")
                
                # 記錄數據
                f.write(f"0.0,{total_count}")
                if pose_result:
                    f.write(f",{pose_result['sitting']},{pose_result['standing']},{pose_result['running']}")
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
                    "points": [(float(x), float(y)) for (x, y) in person_coordinates],
                    "LLM_result": {
                        "坐": pose_result['sitting'] if pose_result else 0,
                        "站": pose_result['standing'] if pose_result else 0,
                        "跑": pose_result['running'] if pose_result else 0
                    }
                }
            }

            with open(json_output, 'w', encoding='utf-8') as jf:
                json.dump(results, jf, ensure_ascii=False, indent=4)

            print(f"JSON結果已保存至: {json_output}")


            # 輸出結果信息
            print(f"\n處理完成！")
            print(f"檢測到總人數: {total_count}")
            if pose_result:
                pose_text = pose_recognizer.format_pose_result(pose_result)
                print(f"姿勢分析: {pose_text}")
            print(f"追蹤圖片已保存至: {track_output}")
            print(f"熱力圖已保存至: {heat_output}")
            print(f"統計數據已保存至: {count_file}")
            
        except Exception as e:
            print(f"處理過程中發生錯誤: {e}")
            return
    
    # 視頻處理（留作未來擴展）
    # 視頻處理
    elif args.video:
        print(f"正在處理影片: {args.video}")
        
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("無法打開影片文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 每秒處理一幀
        process_interval = int(fps)
        if process_interval == 0:
            process_interval = 1
        
        output_prefix = args.output_prefix or os.path.splitext(os.path.basename(args.video))[0]
        track_output_path = f"{output_prefix}_track_out.mp4"
        heat_output_path = f"{output_prefix}_heat_out.mp4"
        json_output_path = f"{output_prefix}_result.json"
        
        # 初始化 VideoWriter
        # 輸出影片為 1 FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        track_writer = cv2.VideoWriter(track_output_path, fourcc, 1, (width, height))
        heat_writer = cv2.VideoWriter(heat_output_path, fourcc, 1, (args.heat_width, args.heat_height))
        
        results_list = []
        # all_person_coordinates = [] # 不再累積熱力圖
        
        frame_count = 0
        processed_seconds = 0
        
        # 獲取空間配置的四個角點
        space_points = heatmap_gen.get_space_points(args.config)
        project_face_coords = [
            (float(space_points['topleft'][0]), float(space_points['topleft'][1])),
            (float(space_points['topright'][0]), float(space_points['topright'][1])),
            (float(space_points['bottomright'][0]), float(space_points['bottomright'][1])),
            (float(space_points['bottomleft'][0]), float(space_points['bottomleft'][1]))
        ]

        print(f"開始處理影片，總幀數: {total_frames}, FPS: {fps}, 預計處理秒數: {total_frames // process_interval}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每秒處理一次
            if frame_count % process_interval == 0:
                print(f"正在處理第 {processed_seconds} 秒...")
                
                try:
                    # 1. 人物檢測
                    total_count, person_coordinates, processed_image = detector.detect_persons_full_image(
                        image_source=frame,
                        space_config=args.config
                    )
                    
                    # 累積座標用於熱力圖 (已移除累積)
                    # all_person_coordinates.extend(person_coordinates)
                    
                    # 2. 生成熱力圖 (使用當前幀的座標)
                    heatmap_image, transformed_coordinates = heatmap_gen.generate_complete_heatmap(
                        person_coordinates=person_coordinates,
                        space_config=args.config,
                        show_markers=True,
                        show_count=True
                    )
                    
                    # 3. 姿勢分析（如果啟用）
                    pose_result = None
                    if pose_recognizer and total_count > 0:
                        # 使用原始幀進行姿勢分析
                        pose_result = pose_recognizer.analyze_poses(
                            image=frame,
                            total_persons=total_count,
                            force_analyze=True
                        )
                    
                    # 4. 寫入影片
                    track_writer.write(processed_image)
                    heat_writer.write(heatmap_image)
                    
                    # 5. 記錄數據
                    result_entry = {
                        "time": processed_seconds,
                        "num_of_people": total_count,
                        "project_face_coords": project_face_coords,
                        "points": [(float(x), float(y)) for (x, y) in person_coordinates], # 當前幀的點
                        "LLM_result": {
                            "坐": pose_result['sitting'] if pose_result else 0,
                            "站": pose_result['standing'] if pose_result else 0,
                            "跑": pose_result['running'] if pose_result else 0
                        }
                    }
                    results_list.append(result_entry)
                    
                    processed_seconds += 1
                    
                except Exception as e:
                    print(f"處理第 {processed_seconds} 秒時發生錯誤: {e}")
            
            frame_count += 1
            
        cap.release()
        track_writer.release()
        heat_writer.release()
        
        # 保存 JSON 結果
        final_json = {"results": results_list}
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
            
        print(f"\n影片處理完成！")
        print(f"追蹤影片已保存至: {track_output_path}")
        print(f"熱力圖影片已保存至: {heat_output_path}")
        print(f"JSON結果已保存至: {json_output_path}")
        return

if __name__ == "__main__":
    main()