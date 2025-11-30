# metrics/__init__.py
from __future__ import annotations

from typing import List

from .base import MetricCollection, BaseMetric
from .classification import build_classification_metrics
from .detection import build_detection_metrics
from .retrieval import build_retrieval_metrics
from .caption import build_caption_metrics


# 在顶部定义
TASK_ALIASES = {
    "区域检测HBB": "水平区域检测",
    "quyujianceHBB": "水平区域检测",
    "区域检测OBB": "旋转区域检测",
    "quyujianceOBB": "旋转区域检测",
    "目标计数": "计数",
    "count": "计数",
    # 可继续补充其他历史写法
}

def normalize_task_name(task: str | None) -> str | None:
    if task is None:
        return None
    task = task.strip()
    return TASK_ALIASES.get(task, task)


# 各任务类别划分（根据需求文档）
CLASSIFICATION_TASKS = {
    "图片分类",
    "水平区域分类",
    "旋转区域分类",
    "VQA1",
    "VQA2",
    "计数",
}

DETECTION_TASKS = {
    "水平区域检测",
    "旋转区域检测",
    "VQA3",
    "视觉定位",
}

RETRIEVAL_TASKS = {
    "图片检索",
}

CAPTION_TASKS = {
    "简洁图片描述",
    "详细图片描述",
    "区域描述",
}


SUPPORTED_TASKS = (
    CLASSIFICATION_TASKS
    | DETECTION_TASKS
    | RETRIEVAL_TASKS
    | CAPTION_TASKS
)


# def is_supported_task(task: str) -> bool:
#     """判断任务是否在当前评估框架支持范围内"""
#     return task in SUPPORTED_TASKS

#新增的行：标准化任务名称
def is_supported_task(task: str) -> bool:
    norm = normalize_task_name(task)
    return norm in SUPPORTED_TASKS if norm else False


def build_task_metrics(
    task: str,
    calc_aux_metric: bool = True,
) -> MetricCollection:
    """
    根据任务类型构建对应的 MetricCollection。

    统一入口：
        - 分类 / VQA1 / VQA2 / 计数       -> classification.build_classification_metrics
        - 检测 / VQA3 / 视觉定位         -> detection.build_detection_metrics
        - 图片检索                      -> retrieval.build_retrieval_metrics
        - 简洁/详细图片描述 & 区域描述  -> caption.build_caption_metrics

    Args:
        task: 标注中的 task 字段（中文名称）
        calc_aux_metric: 是否计算辅助指标（目前只对检索任务生效，
                         其他任务的辅助指标由各自 builder 内部决定）

    Returns:
        MetricCollection 实例

    Raises:
        ValueError: 不支持的任务类型
    """
    #新增的行：标准化任务名称
    task = normalize_task_name(task) or task
    if task in CLASSIFICATION_TASKS:
        return build_classification_metrics(task)

    if task in DETECTION_TASKS:
        return build_detection_metrics(task)

    if task in RETRIEVAL_TASKS:
        # 检索任务支持通过 calc_aux_metric 控制是否要算 Recall / F1
        return build_retrieval_metrics(calc_aux_metric=calc_aux_metric)

    if task in CAPTION_TASKS:
        return build_caption_metrics(task)

    raise ValueError(f"不支持的任务类型：{task}")
