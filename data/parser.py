# parser.py
import json
import re
from typing import List, Dict, Union, Tuple, Optional


class DataParser:
    """数据解析器，负责将原始字符串解析为各任务所需的数据结构"""

    def __init__(self):
        # 修正边界框解析正则表达式，支持数字前后的尖括号
        self.box_pattern = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')
        self.quad_pattern = re.compile(r'<quad><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)></quad>')
        self.number_pattern = re.compile(r'^\d+$')

    def parse_gt(self, task: str, gt_str: str) -> Union[List, str, int, float, None]:
        """解析标注数据的gt字段"""
        return self._parse_by_task(task, gt_str)

    def parse_model_output(self, task: str, output_str: str) -> Union[List, str, int, float, None]:
        """解析模型输出的model_output字段"""
        return self._parse_by_task(task, output_str)

    def _parse_by_task(self, task: str, value_str: str) -> Union[List, str, int, float, None]:
        """根据任务类型解析字符串"""
        if not value_str:
            return None

        try:
            if task in ["图片检索"]:
                # 解析为整数列表 [1,2,3]
                return [int(num.strip()) for num in value_str.split(',')]

            elif task in ["图片分类"]:
                # 解析为类别列表 ["car", "truck"]
                return [cls.strip() for cls in value_str.split(';')]

            elif task in ["VQA1"]:
                # 保持字符串 "Yes"/"No"
                return value_str.strip()

            elif task in ["VQA2", "计数"]:
                # 解析为整数
                if self.number_pattern.match(value_str.strip()):
                    return int(value_str.strip())
                return None

            elif task in ["VQA3", "视觉定位"]:
                # 解析为水平边界框列表 [(x1,y1,x2,y2), ...]
                boxes = self.box_pattern.findall(value_str)
                return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]


            elif task in ["水平区域检测"]:

                # 解析为 (数量, 边界框列表)
                # 直接使用正则提取所有边界框，避免分割拼接问题

                boxes = self.box_pattern.findall(value_str)
                box_list = [
                    (int(x1), int(y1), int(x2), int(y2))
                    for x1, y1, x2, y2 in boxes
                ]

                # 从字符串开头提取连续数字作为数量

                stripped = value_str.strip()
                count_match = re.match(r"\d+", stripped)
                count = int(count_match.group()) if count_match else 0
                return count, box_list



            elif task in ["旋转区域检测"]:

                # 解析为 (数量, 旋转边界框列表)

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
                # 保持类别字符串
                return value_str.strip()

            elif task in ["简洁图片描述", "详细图片描述", "区域描述"]:
                # 保持文本字符串
                return value_str.strip()

            # 新增像素级预留任务的解析支持
            elif task in ["预留扩展任务"]:
                # 预留语义分割、变化检测等任务的基础解析逻辑
                # 保持原始字符串格式，待后续扩展时细化
                return value_str.strip()

            else:
                # 未知任务返回原始字符串
                return value_str.strip()

        except Exception as e:
            print(f"解析任务 {task} 数据 {value_str} 失败: {str(e)}")
            return None