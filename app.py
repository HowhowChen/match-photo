import cv2
import os
import numpy as np
from dotenv import load_dotenv

def find_and_save_matching(template_path, image_path, image_name, threshold=0.9):
    # 讀取模板圖片和目標圖片
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 獲取模板的高度和寬度
    template_height, template_width = template.shape[:2]

    # 使用 TM_CCOEFF_NORMED 方法進行模板匹配
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    # 建立 result 資料夾（如果不存在）
    os.makedirs("results", exist_ok=True)

    for loc in zip(*locations[::-1]):
        top_left = loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        
        # 在目標圖片上畫出匹配結果的矩形框
        matched_image = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), 2)
        
        # 取得另存檔案的名稱
        save_filename = f"result/{image_name}"
        
        # 儲存匹配的圖片
        cv2.imwrite(save_filename, matched_image)

if __name__ == "__main__":
    load_dotenv()
    template_path = os.environ.get("TEMPLATE_PATH") # 模板圖片的路徑
    image_path = os.environ.get("IMAGE_PATH") # 目標圖片路徑
    images = os.listdir(image_path) 
    for image_name in images:
        find_and_save_matching(template_path, os.path.join(image_path, image_name), image_name)
