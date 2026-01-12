import os
import base64
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import io
import google.generativeai as genai

class PoseRecognizer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        初始化姿勢識別器
        
        Args:
            api_key: Google Gemini API金鑰，如果為None會從環境變數或API_KEY.txt讀取
            model_name: 使用的Gemini模型名稱
        """
        self.model_name = model_name
        self.last_analysis_time = 0
        self.analysis_interval = 30  # 30秒分析一次 (一分鐘兩次)
        
        # 設定API金鑰
        if api_key is None:
            api_key = self._load_api_key()
        
        if not api_key:
            raise ValueError("無法獲取Google Gemini API金鑰，請提供api_key參數或設定環境變數GEMINI_API_KEY")
        
        # 配置Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Gemini姿勢識別器已初始化，使用模型: {model_name}")
    
    def _load_api_key(self) -> Optional[str]:
        """
        從環境變數或API_KEY.txt文件載入API金鑰
        
        Returns:
            API金鑰字串，如果無法找到則返回None
        """
        # 首先嘗試從環境變數獲取
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # 嘗試從API_KEY.txt文件讀取
        api_key_file = 'API_KEY.txt'
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    return api_key
            except Exception as e:
                print(f"讀取API_KEY.txt文件時發生錯誤: {e}")
        
        # 嘗試從上級目錄的API_KEY.txt文件讀取
        parent_api_key_file = '../API_KEY.txt'
        if os.path.exists(parent_api_key_file):
            try:
                with open(parent_api_key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    return api_key
            except Exception as e:
                print(f"讀取上級目錄API_KEY.txt文件時發生錯誤: {e}")
        
        return None
    
    def _should_analyze_now(self) -> bool:
        """
        檢查是否應該進行姿勢分析（根據頻率限制）
        
        Returns:
            True如果應該分析，False如果還未到分析時間
        """
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis_time = current_time
            return True
        return False
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        將OpenCV圖像編碼為base64字串
        
        Args:
            image: OpenCV圖像 (BGR格式)
            
        Returns:
            base64編碼的圖像字串
        """
        # 轉換BGR到RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 轉換為PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # 壓縮圖像以減少API調用成本
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # 編碼為base64
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_data
    
    def _create_pose_analysis_prompt(self, total_persons: int) -> str:
        """
        創建姿勢分析的提示詞
        
        Args:
            total_persons: 圖片中總人數
            
        Returns:
            分析提示詞
        """
        prompt = f"""
請分析這張圖片中的人物姿勢。圖片中總共有 {total_persons} 個人。

請統計每種姿勢的人數，並以以下格式回答：
坐: X人
站: Y人
跑: Z人

分析要求：
1. 仔細觀察每個人的身體姿勢
2. "坐"：包括坐在椅子上、地上、或任何坐姿
3. "站"：包括站立、走路等直立姿勢
4. "跑"：包括跑步、快速移動的姿勢
5. 如果某種姿勢沒有人，請標記為0人
6. 請確保總人數等於 {total_persons} 人

只需要回答人數統計，不需要其他說明。
"""
        return prompt
    
    def analyze_poses(self, image: np.ndarray, total_persons: int, 
                     force_analyze: bool = False) -> Optional[Dict[str, int]]:
        """
        分析圖片中人物的姿勢
        
        Args:
            image: 輸入圖片 (OpenCV格式)
            total_persons: 圖片中總人數
            force_analyze: 是否強制分析（忽略時間限制）
            
        Returns:
            姿勢統計字典 {'sitting': int, 'standing': int, 'running': int}
            如果未到分析時間且force_analyze為False，則返回None
        """
        # 檢查是否需要分析
        if not force_analyze and not self._should_analyze_now():
            return None
        
        try:
            # 編碼圖片
            image_base64 = self._encode_image_to_base64(image)
            
            # 創建提示詞
            prompt = self._create_pose_analysis_prompt(total_persons)
            
            # 準備圖片數據
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_base64
            }
            
            # 調用Gemini API
            response = self.model.generate_content([prompt, image_part])
            
            # 解析回應
            result = self._parse_pose_response(response.text, total_persons)
            
            print(f"姿勢分析完成: {result}")
            return result
            
        except Exception as e:
            print(f"姿勢分析過程中發生錯誤: {e}")
            return None
    
    def _parse_pose_response(self, response_text: str, expected_total: int) -> Dict[str, int]:
        """
        解析Gemini API的回應文字
        
        Args:
            response_text: API回應文字
            expected_total: 預期總人數
            
        Returns:
            姿勢統計字典
        """
        # 初始化結果
        result = {
            'sitting': 0,
            'standing': 0,  
            'running': 0
        }
        
        try:
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if '坐:' in line or '坐：' in line:
                    # 提取坐的人數
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        result['sitting'] = int(match.group(1))
                
                elif '站:' in line or '站：' in line:
                    # 提取站的人數
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        result['standing'] = int(match.group(1))
                
                elif '跑:' in line or '跑：' in line:
                    # 提取跑的人數
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        result['running'] = int(match.group(1))
            
            # 驗證總數是否正確
            total_analyzed = sum(result.values())
            if total_analyzed != expected_total:
                print(f"警告: 分析總人數({total_analyzed})與預期人數({expected_total})不符")
                
        except Exception as e:
            print(f"解析回應時發生錯誤: {e}")
        
        return result
    
    def analyze_from_file(self, image_path: str, total_persons: int,
                         force_analyze: bool = False) -> Optional[Dict[str, int]]:
        """
        從文件讀取圖片並分析姿勢
        
        Args:
            image_path: 圖片文件路徑
            total_persons: 圖片中總人數
            force_analyze: 是否強制分析
            
        Returns:
            姿勢統計字典，如果失敗則返回None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法讀取圖片: {image_path}")
            
            return self.analyze_poses(image, total_persons, force_analyze)
            
        except Exception as e:
            print(f"從文件分析姿勢時發生錯誤: {e}")
            return None
    
    def set_analysis_interval(self, seconds: int) -> None:
        """
        設定分析間隔時間
        
        Args:
            seconds: 間隔秒數
        """
        self.analysis_interval = seconds
        print(f"分析間隔已設定為 {seconds} 秒")
    
    def format_pose_result(self, pose_result: Dict[str, int]) -> str:
        """
        格式化姿勢分析結果為字串
        
        Args:
            pose_result: 姿勢統計字典
            
        Returns:
            格式化的字串
        """
        if not pose_result:
            return "無姿勢分析結果"
        
        return f"坐: {pose_result['sitting']}人, 站: {pose_result['standing']}人, 跑: {pose_result['running']}人"

def main():
    """
    測試函數
    """
    try:
        # 初始化姿勢識別器
        recognizer = PoseRecognizer()
        
        # 設定為測試模式（更短的間隔）
        recognizer.set_analysis_interval(5)  # 5秒間隔用於測試
        
        # 測試圖片分析
        test_image_path = 'test.png'
        if os.path.exists(test_image_path):
            result = recognizer.analyze_from_file(
                image_path=test_image_path,
                total_persons=3,  # 假設有3個人
                force_analyze=True  # 強制分析以進行測試
            )
            
            if result:
                formatted_result = recognizer.format_pose_result(result)
                print(f"姿勢分析結果: {formatted_result}")
            else:
                print("姿勢分析失敗")
        else:
            print(f"測試圖片 {test_image_path} 不存在")
            
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()