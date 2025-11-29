import os
import json
import base64
from typing import List, Dict, Tuple, Optional
from .parser import DataParser


class DataValidator:
    """数据验证器，负责校验标注数据和模型输出的有效性"""

    def __init__(self, error_log_path: str = "./error_log.txt",
                 invalid_sample_log_path: str = "./invalid_sample_log.txt"):
        self.parser = DataParser()
        self.supported_tasks = {
            "图片检索", "图片分类", "VQA1", "VQA2", "VQA3", "计数",
            "简洁图片描述", "详细图片描述",
            "水平区域分类", "旋转区域分类",
            "水平区域检测", "旋转区域检测",
            "视觉定位", "区域描述",
            "预留扩展任务",  # 像素级任务预留
        }

        self.error_log_path = error_log_path
        self.invalid_sample_log_path = invalid_sample_log_path

        # 初始化日志文件
        self._init_log_files()

    def _init_log_files(self):
        """初始化日志文件"""
        # 错误日志：包含任务类型
        if not os.path.exists(self.error_log_path):
            dir_name = os.path.dirname(self.error_log_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(self.error_log_path, 'w', encoding='utf-8') as f:
                f.write("时间戳\t样本源\t任务类型\t错误类型\t错误描述\n")

        # 无效样本日志：只需要记录 source 和原因即可
        if not os.path.exists(self.invalid_sample_log_path):
            dir_name = os.path.dirname(self.invalid_sample_log_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(self.invalid_sample_log_path, 'w', encoding='utf-8') as f:
                f.write("时间戳\t样本源\t错误类型\t错误描述\n")

    def _log_error(self, source: str, task: str, error_type: str, error_desc: str):
        """记录错误日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}\t{source}\t{task}\t{error_type}\t{error_desc}\n")

    def _log_invalid_sample(self, source: str, error_type: str, error_desc: str):
        """记录无效样本日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.invalid_sample_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}\t{source}\t{error_type}\t{error_desc}\n")

    def validate_annotation_sample(self, sample: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        验证标注样本的有效性

        Args:
            sample: 标注样本字典

        Returns:
            (是否有效, 错误信息)
        """
        required_fields = ["prompt", "frames", "gt", "task", "source"]

        # 检查必填字段
        for field in required_fields:
            if field not in sample or not sample[field]:
                error_msg = f"缺少必填字段 {field}"
                self._log_invalid_sample(sample.get('source', 'unknown'), "字段缺失", error_msg)
                return False, error_msg

        # 检查任务类型是否支持
        task = sample['task']
        if task not in self.supported_tasks:
            error_msg = f"不支持的任务类型 {task}"
            self._log_invalid_sample(sample['source'], "任务不支持", error_msg)
            return False, error_msg

        # 检查frames的base64编码（开发测试阶段可跳过）
        # 注意：根据需求文档，测试代码应跳过base64检测
        # try:
        #     if isinstance(sample['frames'], list):
        #         for frame in sample['frames']:
        #             base64.b64decode(frame)
        #     else:
        #         base64.b64decode(sample['frames'])
        # except Exception as e:
        #     error_msg = f"frames字段base64解码失败: {str(e)}"
        #     self._log_invalid_sample(sample['source'], "base64解码失败", error_msg)
        #     return False, error_msg

        # 检查gt格式是否可解析
        parsed_gt = self.parser.parse_gt(task, sample['gt'])
        if parsed_gt is None:
            error_msg = f"gt字段格式错误: {sample['gt']}"
            self._log_invalid_sample(sample['source'], "gt格式错误", error_msg)
            return False, error_msg

        return True, None

    def validate_model_output_sample(self, sample: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        验证模型输出样本的有效性

        Args:
            sample: 模型输出样本字典

        Returns:
            (是否有效, 错误信息)
        """
        required_fields = ["sample_id", "task", "model_output", "source"]

        # 检查必填字段
        for field in required_fields:
            if field not in sample or not sample[field]:
                error_msg = f"缺少必填字段 {field}"
                self._log_error(sample.get('source', 'unknown'), sample.get('task', 'unknown'),
                                "字段缺失", error_msg)
                return False, error_msg

        # 检查任务类型是否支持
        task = sample['task']
        if task not in self.supported_tasks:
            error_msg = f"不支持的任务类型 {task}"
            self._log_error(sample['source'], task, "任务不支持", error_msg)
            return False, error_msg

        # 检查model_output是否为空
        if not sample['model_output'].strip():
            error_msg = "模型输出为空"
            self._log_error(sample['source'], task, "输出为空", error_msg)
            return False, error_msg

        # 检查model_output格式是否可解析
        parsed_output = self.parser.parse_model_output(task, sample['model_output'])
        if parsed_output is None:
            error_msg = f"模型输出格式错误: {sample['model_output']}"
            self._log_error(sample['source'], task, "输出格式错误", error_msg)
            return False, error_msg

        return True, None

    def batch_validate_annotations(self, annotations: List[Dict[str, str]]) -> Tuple[
        List[Dict[str, str]], List[Dict[str, str]]]:
        """批量验证标注样本，返回有效样本和无效样本"""
        valid = []
        invalid = []
        for sample in annotations:
            is_valid, _ = self.validate_annotation_sample(sample)
            if is_valid:
                valid.append(sample)
            else:
                invalid.append(sample)
        print(f"标注样本验证完成 - 有效: {len(valid)}, 无效: {len(invalid)}")
        return valid, invalid

    def batch_validate_model_outputs(self, outputs: List[Dict[str, str]]) -> Tuple[
        List[Dict[str, str]], List[Dict[str, str]]]:
        """批量验证模型输出样本，返回有效样本和无效样本"""
        valid = []
        invalid = []
        for sample in outputs:
            is_valid, _ = self.validate_model_output_sample(sample)
            if is_valid:
                valid.append(sample)
            else:
                invalid.append(sample)
        print(f"模型输出验证完成 - 有效: {len(valid)}, 无效: {len(invalid)}")
        return valid, invalid