import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional, Union
from config import POINTS_CONFIGS, PREDICTION_THRESHOLDS_CONFIG

class PersonDetector:
    def __init__(self, model_path: str = 'yolov9e.pt', device: str = None):
        """
        初始化人物檢測器
        
        Args:
            model_path: YOLO模型文件路徑
            device: 設備類型 ('cuda' 或 'cpu')，如果為None則自動選擇
        """
        # 檢查設備
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用設備: {self.device}")
        
        # 加載YOLO模型
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to(self.device)
    
    def get_space_points(self, space_config: str) -> Dict[str, List[int]]:
        """
        根據空間配置名稱獲取點位參數
        
        Args:
            space_config: 空間配置名稱 ('2', 'test', '245', '246', '377')
            
        Returns:
            包含四個點位的字典
        """
        if space_config not in POINTS_CONFIGS:
            raise ValueError(f"不支援的空間配置: {space_config}")
        return POINTS_CONFIGS[space_config]
    
    def detect_persons_in_region(self, 
                                frame: np.ndarray, 
                                conf: float = 0.25, 
                                iou: float = 0.5) -> Tuple[np.ndarray, List[List[int]]]:
        """
        在單一區域中檢測人物（不使用追蹤）
        
        Args:
            frame: 輸入圖片
            conf: 檢測置信度閾值
            iou: IOU閾值
            
        Returns:
            tuple: (處理後的圖片, 人物中心點座標列表)
        """
        boxes = []
        
        # 使用YOLO進行檢測（不使用追蹤）
        results = self.model(frame, classes=[0], device=self.device, conf=conf, iou=iou)
        
        boxes_ = results[0].boxes.xyxy.cpu().numpy()
        
        # 在影像上繪製每個偵測框
        for i, box in enumerate(boxes_):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 計算中心點
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            boxes.append([center_x, center_y])
            
            # 標註人物編號
            # cv2.putText(frame, f"Person {i+1}", (x1, y1 - 10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, boxes
    
    def detect_persons_full_image(self, 
                                 image_source: Union[str, np.ndarray], 
                                 space_config: str) -> Tuple[int, List[List[int]], np.ndarray]:
        """
        對完整圖片進行人物檢測（分割2x2處理）
        
        Args:
            image_source: 圖片路徑 或 圖片數組
            space_config: 空間配置名稱
            
        Returns:
            tuple: (總人數, 所有人物座標點列表, 處理後的圖片)
        """
        # 讀取圖片
        if isinstance(image_source, str):
            frame = cv2.imread(image_source)
            if frame is None:
                raise ValueError(f"無法讀取圖片: {image_source}")
        else:
            frame = image_source.copy()
        
        frame_height, frame_width = frame.shape[:2]
        
        # 分割 2x2
        height_mid = frame_height // 2
        width_mid = frame_width // 2
        
        top_left = frame[:height_mid, :width_mid]
        top_right = frame[:height_mid, width_mid:]
        bottom_left = frame[height_mid:, :width_mid]
        bottom_right = frame[height_mid:, width_mid:]
        
        # 各區域偵測
        top_left, boxes_tl = self.detect_persons_in_region(
            top_left, conf=PREDICTION_THRESHOLDS_CONFIG[space_config]["tf"]["conf"], iou=PREDICTION_THRESHOLDS_CONFIG[space_config]["tf"]["iou"])
        top_right, boxes_tr = self.detect_persons_in_region(
            top_right, conf=PREDICTION_THRESHOLDS_CONFIG[space_config]["tr"]["conf"], iou=PREDICTION_THRESHOLDS_CONFIG[space_config]["tr"]["iou"])
        bottom_left, boxes_bl = self.detect_persons_in_region(
            bottom_left, conf=PREDICTION_THRESHOLDS_CONFIG[space_config]["bf"]["conf"], iou=PREDICTION_THRESHOLDS_CONFIG[space_config]["bf"]["iou"])
        bottom_right, boxes_br = self.detect_persons_in_region(
            bottom_right, conf=PREDICTION_THRESHOLDS_CONFIG[space_config]["br"]["conf"], iou=PREDICTION_THRESHOLDS_CONFIG[space_config]["br"]["iou"])

        # 調整座標到完整圖片坐標系
        boxes_tr = [[x + width_mid, y] for x, y in boxes_tr]
        boxes_bl = [[x, y + height_mid] for x, y in boxes_bl]
        boxes_br = [[x + width_mid, y + height_mid] for x, y in boxes_br]
        
        # 合併所有檢測結果
        all_boxes = boxes_tl + boxes_tr + boxes_bl + boxes_br
        
        # 合併畫面
        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        processed_frame = np.vstack((top, bottom))
        
        # 繪製分割線
        # cv2.line(processed_frame, (0, height_mid), (frame_width, height_mid), (0, 0, 255), 2)
        # cv2.line(processed_frame, (width_mid, 0), (width_mid, frame_height), (0, 0, 255), 2)
        
        # 獲取並繪製空間點位
        points = self.get_space_points(space_config)
        src_pts_int = np.int32([
            points['topleft'],
            points['topright'],
            points['bottomright'],
            points['bottomleft']
        ])
        
        # 標註點與多邊形
        for point in src_pts_int:
            cv2.circle(processed_frame, point, 8, (255, 0, 0), -1)
        points_array = src_pts_int.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(processed_frame, [points_array], True, (0, 255, 0), thickness=4)
        
        return len(all_boxes), all_boxes, processed_frame

def main():
    """
    測試函數
    """
    # 初始化檢測器
    detector = PersonDetector(model_path='yolov9e.pt')
    
    # 測試圖片檢測
    try:
        total_count, person_coordinates, processed_image = detector.detect_persons_full_image(
            image_path='test.png',
            space_config='test'
        )
        
        print(f"檢測到總人數: {total_count}")
        print(f"人物座標點: {person_coordinates}")
        
        # 保存處理後的圖片
        cv2.imwrite('person_detection_result.png', processed_image)
        print("檢測結果已保存至: person_detection_result.png")
        
    except Exception as e:
        print(f"檢測過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
