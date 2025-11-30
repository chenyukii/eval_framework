# parser.py
import json
import re
from typing import List, Dict, Union, Tuple, Optional
from ..metrics import normalize_task_name

class DataParser:
    """数据解析器，负责将原始字符串解析为各任务所需的数据结构"""

    def __init__(self):
        self.box_pattern = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')
        self.quad_pattern = re.compile(r'<quad><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)></quad>')
        self.number_pattern = re.compile(r'^\d+$')

    def parse_gt(self, task: str, gt_str: str) -> Union[List, str, int, float, None]:
        return self._parse_by_task(task, gt_str)

    def parse_model_output(self, task: str, output_str: str) -> Union[List, str, int, float, None]:
        return self._parse_by_task(task, output_str)

    def _parse_by_task(self, task: str, value_str: str) -> Union[List, str, int, float, None]:
        task = normalize_task_name(task) or task
        if not value_str:
            return None
        try:
            if task in ["图片检索"]:
                return [int(num.strip()) for num in value_str.split(',')]
            elif task in ["图片分类"]:
                return [cls.strip() for cls in value_str.split(';')]
            elif task in ["VQA1"]:
                return value_str.strip()
            elif task in ["VQA2", "计数"]:
                if self.number_pattern.match(value_str.strip()):
                    return int(value_str.strip())
                return None
            elif task in ["VQA3", "视觉定位"]:
                boxes = self.box_pattern.findall(value_str)
                return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
            elif task in ["水平区域检测"]:
                boxes = self.box_pattern.findall(value_str)
                box_list = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
                stripped = value_str.strip()
                count_match = re.match(r"\d+", stripped)
                count = int(count_match.group()) if count_match else 0
                return count, box_list
            elif task in ["旋转区域检测"]:
                quads = self.quad_pattern.findall(value_str)
                quad_list = [
                    (int(x1), int(y1), int(x2), int(y2),
                     int(x3), int(y3), int(x4), int(y4))
                    for x1, y1, x2, y2, x3, y3, x4, y4 in quads
                ]
                stripped = value_str.strip()
                count_match = re.match(r"\d+", stripped)
                count = int(count_match.group()) if count_match else 0
                return count, quad_list
            elif task in ["水平区域分类", "旋转区域分类"]:
                return value_str.strip()
            elif task in ["简洁图片描述", "详细图片描述", "区域描述"]:
                return value_str.strip()
            elif task in ["预留扩展任务"]:
                return value_str.strip()
            else:
                return value_str.strip()
        except Exception as e:
            print(f"解析任务 {task} 数据 {value_str} 失败: {str(e)}")
            return None
