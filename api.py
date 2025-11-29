from typing import Dict, Any, List
from metrics.base import MetricBase
from data.loader import DataLoader


class EvaluationAPI:
    """评估系统API接口，提供单样本和批量评估功能"""

    def __init__(self, config_path: str = "./config"):
        """
        初始化评估API

        Args:
            config_path: 配置文件目录路径
        """
        self.data_loader = DataLoader()
        self.config_path = config_path
        self.task_metric_map = self._load_task_metric_map()  # 任务-指标映射

    def _load_task_metric_map(self) -> Dict[str, str]:
        """加载任务与指标类的映射关系（从配置文件）"""
        # 暂不实现具体逻辑，只定义接口
        raise NotImplementedError("需实现任务-指标映射加载逻辑")

    def get_metric_calculator(self, task_name: str, calc_aux: bool = False) -> MetricBase:
        """
        获取指定任务的指标计算器

        Args:
            task_name: 任务名称
            calc_aux: 是否计算辅助指标

        Returns:
            指标计算器实例
        """
        # 暂不实现具体逻辑，只定义接口
        raise NotImplementedError("需实现指标计算器获取逻辑")

    def evaluate_single_sample(self, task_name: str, gt: str, model_output: str, calc_aux: bool = False) -> Dict[
        str, float]:
        """
        评估单个样本

        Args:
            task_name: 任务名称
            gt: 真实标签
            model_output: 模型输出
            calc_aux: 是否计算辅助指标

        Returns:
            指标字典
        """
        # 暂不实现具体逻辑，只定义接口
        raise NotImplementedError("需实现单样本评估逻辑")

    def evaluate_batch_from_files(self, anno_path: str, model_result_path: str, output_dir: str, calc_aux: bool = False,
                                  vis: bool = False) -> Dict[str, Any]:
        """
        从文件批量评估

        Args:
            anno_path: 标注文件或目录路径
            model_result_path: 模型输出文件或目录路径
            output_dir: 结果输出目录
            calc_aux: 是否计算辅助指标
            vis: 是否生成可视化图表

        Returns:
            整体评估结果
        """
        # 暂不实现具体逻辑，只定义接口
        raise NotImplementedError("需实现批量文件评估逻辑")