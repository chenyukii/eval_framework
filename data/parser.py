# parser.py
import json
import re
from typing import List, Dict, Union, Tuple, Optional
from ..metrics import normalize_task_name

Number = Union[int, float]
Point = Tuple[Number, Number]
Polygon = List[Point]


class DataParser:
    """数据解析器：将原始字符串解析为各任务所需的数据结构"""

    def __init__(self):
        # 检测类已有正则
        self.box_pattern = re.compile(r'<box><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)></box>')
        self.quad_pattern = re.compile(
            r'<quad><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)>'
            r'<(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)></quad>'
        )
        # 分割类多边形/变化检测点列
        self.poly_block = re.compile(r'<poly>(.*?)</poly>', re.IGNORECASE)
        self.dot_block = re.compile(r'<dot>(.*?)</dot>', re.IGNORECASE)
        self.num_pattern = re.compile(r'-?\d+(?:\.\d+)?')
        self.number_pattern = re.compile(r'^\d+$')

    def parse_gt(self, task: str, gt_str: str) -> Union[List, str, int, float, None]:
        return self._parse_by_task(task, gt_str)

    def parse_model_output(self, task: str, output_str: str) -> Union[List, str, int, float, None]:
        return self._parse_by_task(task, output_str)

    # ----------------- 内部工具 -----------------

    def _parse_polygons(self, value_str: str, kind: str) -> Optional[List[Polygon]]:
        """解析 <poly>/<dot>... 标签为多边形列表"""
        block_pat = self.poly_block if kind == "poly" else self.dot_block
        polys: List[Polygon] = []
        try:
            for block in block_pat.findall(value_str):
                nums = [float(n) for n in self.num_pattern.findall(block)]
                if len(nums) < 2 or len(nums) % 2 != 0:
                    return None  # 坐标数不足或奇数，判无效
                poly: Polygon = []
                for i in range(0, len(nums), 2):
                    poly.append((nums[i], nums[i + 1]))
                polys.append(poly)
            # 无标签直接返回空列表；由上游决定是否跳过
            return polys
        except Exception:
            return None

    def _parse_by_task(self, task: str, value_str: str) -> Union[List, str, int, float, None]:
        task = normalize_task_name(task) or task
        if value_str is None:
            return None
        value_str = value_str.strip()
        if not value_str:
            return None

        try:
            # 图像/分类/VQA/检测保留原逻辑
            # if task in ["图片检索"]:
            #     return [int(num.strip()) for num in value_str.split(',')]
            if task in ["图片检索"]:
                # 支持逗号/分号分隔，过滤空串和非数字
                raw_parts = re.split(r"[;,]", value_str)
                nums = []
                for part in raw_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if not part.isdigit():
                        return None  # 非数字视为格式错误
                    nums.append(int(part))
                return nums
            elif task in ["图片分类"]:
                return [cls.strip() for cls in value_str.split(';') if cls.strip()]
            elif task in ["VQA1"]:
                return value_str.strip().lower()
            elif task in ["VQA2", "计数"]:
                if self.number_pattern.match(value_str):
                    return int(value_str)
                return None
            elif task in ["VQA3", "视觉定位"]:
                boxes = self.box_pattern.findall(value_str)
                return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in boxes]
            elif task in ["水平区域检测"]:
                boxes = self.box_pattern.findall(value_str)
                box_list = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in boxes]
                stripped = value_str
                count_match = re.match(r"\d+", stripped)
                count = int(count_match.group()) if count_match else len(box_list)
                return count, box_list
            elif task in ["旋转区域检测"]:
                quads = self.quad_pattern.findall(value_str)
                quad_list = [
                    (float(x1), float(y1), float(x2), float(y2),
                     float(x3), float(y3), float(x4), float(y4))
                    for x1, y1, x2, y2, x3, y3, x4, y4 in quads
                ]
                stripped = value_str
                count_match = re.match(r"\d+", stripped)
                count = int(count_match.group()) if count_match else len(quad_list)
                return count, quad_list
            elif task in ["水平区域分类", "旋转区域分类"]:
                return value_str
            elif task in ["简洁图片描述", "详细图片描述", "区域描述"]:
                return value_str
            # -------- 像素级任务 --------
            elif task in ["像素分类", "细粒度识别"]:
                return value_str
            elif task in ["语义分割"]:
                return self._parse_polygons(value_str, "poly")
            elif task in ["实例分割"]:
                return self._parse_polygons(value_str, "poly")
            elif task in ["变化检测"]:
                return self._parse_polygons(value_str, "dot")
            elif task in ["预留扩展任务"]:
                return value_str
            else:
                return value_str
        except Exception as e:
            print(f"解析任务 {task} 数据 {value_str} 失败: {str(e)}")
            return None
