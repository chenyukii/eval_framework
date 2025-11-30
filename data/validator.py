import os
import json
import base64
from typing import List, Dict, Tuple, Optional
from .parser import DataParser
from ..metrics import normalize_task_name, is_supported_task
from ..utils.logger import get_logger, log_struct

logger = get_logger(__name__)


class DataValidator:
    """数据验证器，负责校验标注数据和模型输出的有效性，并输出结构化日志。"""

    def __init__(self, error_log_path: str = "./error_log.txt",
                 invalid_sample_log_path: str = "./invalid_sample_log.txt"):
        self.parser = DataParser()
        self.supported_tasks = {
            "图片检索", "图片分类", "VQA1", "VQA2", "VQA3", "计数",
            "简洁图片描述", "详细图片描述",
            "水平区域分类", "旋转区域分类",
            "水平区域检测", "旋转区域检测",
            "视觉定位", "区域描述",
            "预留扩展任务",
        }

        self.error_log_path = error_log_path
        self.invalid_sample_log_path = invalid_sample_log_path
        self._init_log_files()

    def _init_log_files(self):
        if not os.path.exists(self.error_log_path):
            dir_name = os.path.dirname(self.error_log_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(self.error_log_path, 'w', encoding='utf-8') as f:
                f.write("时间戳\t样本源\t任务类型\t错误类型\t错误描述\n")

        if not os.path.exists(self.invalid_sample_log_path):
            dir_name = os.path.dirname(self.invalid_sample_log_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(self.invalid_sample_log_path, 'w', encoding='utf-8') as f:
                f.write("时间戳\t样本源\t错误类型\t错误描述\n")

    def _log_error(self, source: str, task: str, error_type: str, error_desc: str):
        """原有文本日志；保留兼容，同时写结构化日志。"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}\t{source}\t{task}\t{error_type}\t{error_desc}\n")
        log_struct(
            stage="validate",
            error_code=error_type,
            message=error_desc,
            level="ERROR",
            logger=logger,
            source=source,
            task=task,
        )

    def _log_invalid_sample(self, source: str, error_type: str, error_desc: str):
        """原有文本日志；保留兼容，同时写结构化日志。"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.invalid_sample_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}\t{source}\t{error_type}\t{error_desc}\n")
        log_struct(
            stage="validate",
            error_code=error_type,
            message=error_desc,
            level="WARNING",
            logger=logger,
            source=source,
        )

    def validate_annotation_sample(self, sample: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        required_fields = ["prompt", "frames", "gt", "task", "source"]
        for field in required_fields:
            if field not in sample or not sample[field]:
                error_msg = f"缺少必填字段 {field}"
                self._log_invalid_sample(sample.get('source', 'unknown'), "字段缺失", error_msg)
                return False, error_msg

        task_norm = normalize_task_name(sample['task'])
        if not task_norm or not is_supported_task(task_norm):
            error_msg = f"不支持的任务类型 {sample['task']}"
            self._log_invalid_sample(sample['source'], "任务不支持", error_msg)
            return False, error_msg
        sample['task'] = task_norm

        parsed_gt = self.parser.parse_gt(task_norm, sample['gt'])
        if parsed_gt is None:
            error_msg = f"gt字段格式错误: {sample['gt']}"
            self._log_invalid_sample(sample['source'], "gt格式错误", error_msg)
            return False, error_msg

        return True, None

    def validate_model_output_sample(self, sample: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        required_fields = ["sample_id", "task", "model_output", "source"]
        for field in required_fields:
            if field not in sample or not sample[field]:
                error_msg = f"缺少必填字段 {field}"
                self._log_error(sample.get('source', 'unknown'), sample.get('task', 'unknown'),
                                "字段缺失", error_msg)
                return False, error_msg

        task_norm = normalize_task_name(sample['task'])
        if not task_norm or not is_supported_task(task_norm):
            error_msg = f"不支持的任务类型 {sample['task']}"
            self._log_error(sample['source'], sample.get('task', 'unknown'), "任务不支持", error_msg)
            return False, error_msg
        sample['task'] = task_norm

        if not sample['model_output'].strip():
            error_msg = "模型输出为空"
            self._log_error(sample['source'], task_norm, "输出为空", error_msg)
            return False, error_msg

        parsed_output = self.parser.parse_model_output(task_norm, sample['model_output'])
        if parsed_output is None:
            error_msg = f"模型输出格式错误: {sample['model_output']}"
            self._log_error(sample['source'], task_norm, "输出格式错误", error_msg)
            return False, error_msg

        return True, None

    def batch_validate_annotations(self, annotations: List[Dict[str, str]]) -> Tuple[
        List[Dict[str, str]], List[Dict[str, str]]]:
        valid = []
        invalid = []
        for sample in annotations:
            is_valid, _ = self.validate_annotation_sample(sample)
            if is_valid:
                valid.append(sample)
            else:
                invalid.append(sample)
        logger.info("标注样本验证完成 - 有效: %d, 无效: %d", len(valid), len(invalid))
        return valid, invalid

    def batch_validate_model_outputs(self, outputs: List[Dict[str, str]]) -> Tuple[
        List[Dict[str, str]], List[Dict[str, str]]]:
        valid = []
        invalid = []
        for sample in outputs:
            is_valid, _ = self.validate_model_output_sample(sample)
            if is_valid:
                valid.append(sample)
            else:
                invalid.append(sample)
        logger.info("模型输出验证完成 - 有效: %d, 无效: %d", len(valid), len(invalid))
        return valid, invalid
