import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from config import POINTS_CONFIGS

class HeatmapGenerator:
    def __init__(self, width: int = 1000, height: int = 1000):
        """
        初始化熱力圖生成器
        
        Args:
            width: 熱力圖寬度
            height: 熱力圖高度
        """
        self.width = width
        self.height = height
    
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
    
    def get_perspective_transform_matrix(self, space_config: str) -> np.ndarray:
        """
        根據空間配置計算透視變換矩陣
        
        Args:
            space_config: 空間配置名稱
            
        Returns:
            透視變換矩陣
        """
        points = self.get_space_points(space_config)
        
        # 設定透視變換的源點
        src_pts = np.float32([
            points['topleft'], 
            points['topright'], 
            points['bottomright'], 
            points['bottomleft']
        ])
        
        # 設定目標的矩形平面
        # dst_pts = np.float32([
        #     [0, 0], 
        #     [self.width, 0], 
        #     [self.width, self.height], 
        #     [0, self.height]
        # ])
        dst_pts = np.float32([
            [0, 0], 
            [500, 0], 
            [500, 500], 
            [0, 500]
        ])
        
        # 計算透視變換矩陣
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M
    
    def create_heatmap_base(self, person_coordinates: List[List[int]], 
                           transform_matrix: np.ndarray,
                           heat_radius: int = 30) -> np.ndarray:
        """
        創建基礎熱力圖
        
        Args:
            person_coordinates: 人物座標點列表 [[x1, y1], [x2, y2], ...]
            transform_matrix: 透視變換矩陣
            heat_radius: 熱力點半徑
            
        Returns:
            基礎熱力圖 (灰度)
        """
        # 創建熱力圖
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 在變換後的矩形平面上繪製點並累積熱力
        for coord in person_coordinates:
            center_x, center_y = coord
            points_array = np.array([[center_x, center_y]], dtype='float32')
            points_array = np.array([points_array])
            transformed_points = cv2.perspectiveTransform(points_array, transform_matrix)
            
            x, y = int(transformed_points[0][0][0]), int(transformed_points[0][0][1])
            # if 0 <= x < self.width and 0 <= y < self.height:
            cv2.circle(heatmap, (x, y), heat_radius, 1, -1)
        
        return heatmap
    
    def process_heatmap(self, base_heatmap: np.ndarray, 
                       blur_kernel_size: int = 201) -> np.ndarray:
        """
        處理熱力圖（高斯模糊和歸一化）
        
        Args:
            base_heatmap: 基礎熱力圖
            blur_kernel_size: 高斯模糊核大小
            
        Returns:
            處理後的熱力圖 (0-1範圍)
        """
        # 高斯模糊
        heatmap = cv2.GaussianBlur(base_heatmap, (blur_kernel_size, blur_kernel_size), 0)
        
        # 歸一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap
    
    def colorize_heatmap(self, normalized_heatmap: np.ndarray, 
                        colormap: int = cv2.COLORMAP_JET,
                        background_color: Tuple[int, int, int] = (255, 0, 0),
                        alpha: float = 0.7) -> np.ndarray:
        """
        為熱力圖添加顏色
        
        Args:
            normalized_heatmap: 歸一化的熱力圖 (0-1範圍)
            colormap: OpenCV顏色映射
            background_color: 背景顏色 (B, G, R)
            alpha: 熱力圖透明度
            
        Returns:
            彩色熱力圖
        """
        # 轉換為8位圖像
        heat_gray = np.uint8(255 * normalized_heatmap)
        heatmap_colored = cv2.applyColorMap(heat_gray, colormap)
        
        # 創建背景
        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background[:] = background_color
        
        # 合併背景和熱力圖
        final_heatmap = cv2.addWeighted(background, 1 - alpha, heatmap_colored, alpha, 0)
        
        return final_heatmap
    
    def add_person_markers(self, heatmap_image: np.ndarray,
                          person_coordinates: List[List[int]],
                          transform_matrix: np.ndarray,
                          marker_color: Tuple[int, int, int] = (255, 255, 255),
                          marker_radius: int = 6) -> np.ndarray:
        """
        在熱力圖上添加人物位置標記
        
        Args:
            heatmap_image: 熱力圖圖像
            person_coordinates: 人物座標點列表
            transform_matrix: 透視變換矩陣
            marker_color: 標記顏色 (B, G, R)
            marker_radius: 標記半徑
            
        Returns:
            添加標記後的熱力圖
        """
        result_image = heatmap_image.copy()
        transformed_person_coordinates = []
        
        if len(person_coordinates) > 0:
            for coord in person_coordinates:
                center_x, center_y = coord
                points_array = np.array([[center_x, center_y]], dtype='float32')
                points_array = np.array([points_array])
                transformed_points = cv2.perspectiveTransform(points_array, transform_matrix)
                
                x, y = int(transformed_points[0][0][0]), int(transformed_points[0][0][1])
                transformed_person_coordinates.append((x, y))
                cv2.circle(result_image, (x, y), marker_radius, marker_color, -1)
        
        return result_image, transformed_person_coordinates
    
    def add_text_overlay(self, heatmap_image: np.ndarray,
                        text: str,
                        position: Tuple[int, int] = (10, 30),
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 2) -> np.ndarray:
        """
        在熱力圖上添加文字
        
        Args:
            heatmap_image: 熱力圖圖像
            text: 要添加的文字
            position: 文字位置 (x, y)
            font_scale: 字體大小
            color: 文字顏色 (B, G, R)
            thickness: 文字粗細
            
        Returns:
            添加文字後的熱力圖
        """
        result_image = heatmap_image.copy()
        cv2.putText(result_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return result_image
    
    def generate_complete_heatmap(self, person_coordinates: List[List[int]],
                                 space_config: str,
                                 show_markers: bool = True,
                                 show_count: bool = True,
                                 heat_radius: int = 30,
                                 blur_kernel_size: int = 201) -> np.ndarray:
        """
        生成完整的熱力圖
        
        Args:
            person_coordinates: 人物座標點列表
            space_config: 空間配置名稱
            show_markers: 是否顯示人物位置標記
            show_count: 是否顯示人數統計
            heat_radius: 熱力點半徑
            blur_kernel_size: 高斯模糊核大小
            
        Returns:
            完整的熱力圖
        """
        # 獲取透視變換矩陣
        transform_matrix = self.get_perspective_transform_matrix(space_config)
        
        # 創建基礎熱力圖
        base_heatmap = self.create_heatmap_base(person_coordinates, transform_matrix, heat_radius)
        
        # 處理熱力圖
        processed_heatmap = self.process_heatmap(base_heatmap, blur_kernel_size)
        
        # 添加顏色
        colored_heatmap = self.colorize_heatmap(processed_heatmap)
        
        # 添加人物標記
        if show_markers:
            colored_heatmap, transformed_coordinates = self.add_person_markers(
                colored_heatmap, 
                person_coordinates, 
                transform_matrix, 
                marker_color=(0, 0, 255),  # 紅色標記
                marker_radius=8  # 較大的標記半徑
            )
        
        # 添加人數統計
        if show_count:
            person_count = len(person_coordinates)
            count_text = f"number of persons: {person_count}"
            colored_heatmap = self.add_text_overlay(colored_heatmap, count_text)
        
        return colored_heatmap, transformed_coordinates
    
    def save_heatmap(self, heatmap_image: np.ndarray, output_path: str) -> None:
        """
        保存熱力圖到文件
        
        Args:
            heatmap_image: 熱力圖圖像
            output_path: 輸出文件路徑
        """
        cv2.imwrite(output_path, heatmap_image)
        print(f"熱力圖已保存至: {output_path}")

def main():
    """
    測試函數
    """
    # 初始化熱力圖生成器
    heatmap_gen = HeatmapGenerator(width=500, height=500)
    
    # 測試數據：一些人物座標點
    test_coordinates = [
        [100, 200],
        [300, 400],
        [500, 600],
        [700, 300],
        [800, 800]
    ]
    
    try:
        # 生成熱力圖
        heatmap = heatmap_gen.generate_complete_heatmap(
            person_coordinates=test_coordinates,
            space_config='test',
            show_markers=True,
            show_count=True
        )
        
        # 保存熱力圖
        heatmap_gen.save_heatmap(heatmap, 'test_heatmap_output.png')
        
        print(f"成功生成熱力圖，包含 {len(test_coordinates)} 個人物座標點")
        
    except Exception as e:
        print(f"生成熱力圖過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()