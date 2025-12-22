# evaluator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import build_task_metrics
from .data.parser import DataParser  # 如果你的路径是 data/parser.py，这里改成相对 import


class EvaluationCore:
    """
    评估核心类：

    只负责：
        - 根据 task 构建合适的 MetricCollection
        - 遍历一组样本，调用 metrics.update(...)
        - 返回最终的指标结果字典

    不负责：
        - 读文件 / 写文件
        - 日志记录
        - 配对逻辑（sample_id/source 匹配）
        - 文本 / 图像等字段的格式验证

    这些职责由 DataLoader / DataValidator / DataParser 等模块完成，
    这里只假设你已经拿到了“能被 DataParser 正确解析”的样本对。
    """

    def __init__(self) -> None:
        # DataParser 用于把 raw gt / raw model_output 解析成结构化形式
        self.parser = DataParser()

    # ========= 已解析样本评估 =========

    def evaluate_parsed_samples(
        self,
        task: str,
        pairs: Sequence[Dict[str, Any]],
        *,
        calc_aux_metric: bool = True,
        sample_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        对一组“已配对 + 已解析”的样本做评估。

        Args:
            task:
                任务名称（例如 "图片分类"、"VQA2"、"水平区域检测"、"图片检索"、
                "简洁图片描述" 等），必须是 metrics.build_task_metrics 支持的任务。
            pairs:
                样本列表，每个元素是一个 dict，至少包含：
                    - "gt":   解析后的 ground truth（DataParser 的输出）
                    - "pred": 解析后的模型输出（DataParser 的输出或你自行构造）
                推荐额外字段（非必须，但有利于日志）：
                    - "sample_id": 样本 ID
                    - "source":    数据源（文件名 / 图片名）
            calc_aux_metric:
                是否计算辅助指标（目前只对“图片检索”任务起作用；
                其他任务的辅助指标由各自 build_xxx_metrics 决定）。
            sample_details:
                若传入空列表，将按输入顺序记录每条样本的单独指标，用于后续定位
                哪些样本表现差。形如：
                [
                  {"sample_id": "...", "source": "...", "metrics": {...}},
                  ...
                ]
                顺序与 pairs 顺序一致。

        Returns:
            指标结果字典，例如：
                {
                    "accuracy": 0.92,
                    "macro_f1": 0.88,
                    ...
                }
            数值是 0~1 之间的小数，保留两位小数（由 MetricCollection.compute_all 统一处理）。
        """
        if not pairs:
            # 没有样本，直接返回空指标字典
            return {}

        # 1. 根据任务构建指标集合（分类 / 检测 / 检索 / 描述 / 像素级等）
        metric_collection = build_task_metrics(task, calc_aux_metric=calc_aux_metric)

        # 2. 遍历每条样本，调用 metrics.update(...)
        for pair in pairs:
            if "gt" not in pair or "pred" not in pair:
                # 如果传进来的结构不符合要求，直接抛异常，
                # 让上层在开发阶段就能发现问题。
                raise KeyError(
                    "每个 pair 至少需要包含 'gt' 和 'pred' 字段；"
                    f"当前 pair: {pair}"
                )

            gt = pair.get("gt")
            pred = pair.get("pred")
            source = pair.get("source")  # 不是必须，但检测类指标会当作 image_id

            # 统一接口：所有 Metric 的 update 都是 (gt, pred, task, source, **kwargs)
            metric_collection.update(
                gt=gt,
                pred=pred,
                task=task,
                source=source,
            )

            # 如果需要样本级指标，单独构建一次同任务的 MetricCollection，仅喂这一条
            if sample_details is not None:
                per_sample_metrics = build_task_metrics(
                    task,
                    calc_aux_metric=calc_aux_metric,
                )
                per_sample_metrics.update(
                    gt=gt,
                    pred=pred,
                    task=task,
                    source=source,
                )
                sample_details.append(
                    {
                        "sample_id": pair.get("sample_id"),
                        "source": source,
                        "metrics": per_sample_metrics.compute_all(),
                    }
                )

        # 3. 计算所有指标结果（0~1 范围，保留两位小数）
        return metric_collection.compute_all()

    # ========= 原始样本对评估（一口气吃完） =========

    def evaluate_raw_paired_samples(
        self,
        task: str,
        paired_samples: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
        *,
        calc_aux_metric: bool = True,
        sample_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        对一组“原始样本对”做评估（一次性吃完的版本）：

        输入：
            paired_samples: [(anno_sample, model_sample), ...]
                anno_sample: 来自标注文件，至少包含 gt/task/source
                model_sample: 来自模型输出，至少包含 model_output/task/source

        内部流程：
            1. 用 DataParser 把 gt / model_output 解析成各任务需要的结构
            2. 组装成 evaluate_parsed_samples 需要的格式
            3. 调用 evaluate_parsed_samples 得到指标结果
        """
        if not paired_samples:
            return {}

        parsed_pairs: List[Dict[str, Any]] = []

        for anno_sample, model_sample in paired_samples:
            # 任务名优先用标注里的，其次用模型输出里的，最后退回函数参数 task
            sample_task = (
                anno_sample.get("task")
                or model_sample.get("task")
                or task
            )

            raw_gt = anno_sample.get("gt", "")
            raw_output = model_sample.get("model_output", "")

            parsed_gt = self.parser.parse_gt(sample_task, raw_gt)
            if parsed_gt is None:
                raise ValueError(
                    f"解析 gt 失败 (task={sample_task}, source={anno_sample.get('source')}): {raw_gt}"
                )

            parsed_pred = self.parser.parse_model_output(sample_task, raw_output)
            if parsed_pred is None:
                raise ValueError(
                    f"解析 model_output 失败 (task={sample_task}, source={model_sample.get('source')}): {raw_output}"
                )

            parsed_pairs.append(
                {
                    "gt": parsed_gt,
                    "pred": parsed_pred,
                    # source 优先用标注里的，没有就用模型输出里的
                    "source": anno_sample.get("source") or model_sample.get("source"),
                    "sample_id": anno_sample.get("sample_id") or model_sample.get("sample_id"),
                    "task": sample_task,
                }
            )

        # 复用已解析样本评估接口
        return self.evaluate_parsed_samples(
            task=task,
            pairs=parsed_pairs,
            calc_aux_metric=calc_aux_metric,
            sample_details=sample_details,
        )

    # ========= 下面是“流式 / 可复用 metrics”接口 =========

    def build_metrics(
        self,
        task: str,
        *,
        calc_aux_metric: bool = True,
    ):
        """
        构建一份可跨文件 / 跨批次复用的 MetricCollection。

        适合这样的使用场景：
            core = EvaluationCore()
            metrics = core.build_metrics(task)
            for batch in many_batches:
                core.update_metrics_with_raw_pairs(task, metrics, batch)
            result = metrics.compute_all()
        """
        return build_task_metrics(task, calc_aux_metric=calc_aux_metric)

    def update_metrics_with_raw_pairs(
        self,
        task: str,
        metric_collection,
        paired_samples: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> None:
        """
        关键：对给定的 MetricCollection 做“流式更新”。

        - 不自己 new metrics（由外部传进来）
        - 不构造 parsed_pairs 大列表
        - 只负责：遍历每个 (anno_sample, model_sample)，解析后调用 metrics.update(...)
        """
        if not paired_samples:
            return

        for anno_sample, model_sample in paired_samples:
            sample_task = (
                anno_sample.get("task")
                or model_sample.get("task")
                or task
            )

            # 保险起见，如果样本里 task 和当前评估 task 不一致，直接跳过
            if sample_task != task:
                # 这里也可以选择 raise，这里先保守处理为跳过
                # 你以后可以根据需要改成报错
                continue

            raw_gt = anno_sample.get("gt", "")
            raw_output = model_sample.get("model_output", "")

            parsed_gt = self.parser.parse_gt(sample_task, raw_gt)
            if parsed_gt is None:
                # DataValidator 前面一般已经过滤了非法样本，这里再防御性跳过一次
                continue

            parsed_pred = self.parser.parse_model_output(sample_task, raw_output)
            if parsed_pred is None:
                continue

            source = anno_sample.get("source") or model_sample.get("source")

            metric_collection.update(
                gt=parsed_gt,
                pred=parsed_pred,
                task=task,
                source=source,
            )
