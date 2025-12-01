# metrics/detection.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMetric, MetricCollection
from ..utils.iou import bbox_iou, quad_iou

BBox = Tuple[float, float, float, float]
Quad = Tuple[float, float, float, float, float, float, float, float]


def _extract_count_and_boxes(task: str, value: Any) -> Tuple[int, List[Any]]:
    """
    从解析后的检测类结果中提取 (count, boxes)

    适用任务:
        - 水平区域检测、旋转区域检测：预期格式 (count, box_list)，也容忍直接给 box_list
        - VQA3：预期 box_list，也兼容 (count, box_list) 写法

    兜底逻辑:
        - 无 count 时，用 len(box_list) 作为数量
        - 无效输入时返回 (0, [])
    """
    if value is None:
        return 0, []

    if task in {"水平区域检测", "旋转区域检测", "VQA3"}:
        # 优先处理 (count, box_list) 形式
        if isinstance(value, tuple) and len(value) == 2:
            raw_count, boxes = value
            boxes = boxes if isinstance(boxes, list) else []
            try:
                count = int(raw_count)
            except Exception:
                count = len(boxes)
            return count, boxes
        # 直接给了列表：用列表长度兜底
        if isinstance(value, list):
            return len(value), value

    return 0, []


class DetectionAPMetric(BaseMetric):
    """
    通用检测 AP 指标（单阈值，单类），支持：
        - 水平区域检测（水平框）
        - 旋转区域检测（旋转框）
        - VQA3（水平框）
        - 视觉定位（水平框）

    说明:
        - 当前模型输出没有置信度，内部假设 score 固定为 1
        - 因此 PR 曲线只会有一个点，AP 退化为该点的面积
        - 如果未来有 score，可在计算前对 score 排序，得到标准 PR 曲线
    """

    def __init__(
        self,
        name: str,
        iou_threshold: float,
        box_mode: str = "bbox",  # "bbox" 或 "quad"
        is_aux: bool = False,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.box_mode = box_mode

        if box_mode == "bbox":
            self._iou_fn = bbox_iou
        elif box_mode == "quad":
            self._iou_fn = quad_iou
        else:
            raise ValueError(f"未知 box_mode: {box_mode}")

        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        # image_id -> gt 框列表
        self.gts: Dict[str, List[Any]] = {}
        # image_id -> 预测框列表（单类，无分类字段）
        self.preds: Dict[str, List[Any]] = {}

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        按任务提取框列表并累积到当前 image_id
            - 检测类：从 (count, boxes) 取出 boxes
            - VQA3/视觉定位：直接使用 boxes 列表
        """
        if source is None:
            # 没有 image_id 无法匹配，直接跳过
            return

        if task in {"水平区域检测", "旋转区域检测"}:
            _, gt_boxes = _extract_count_and_boxes(task, gt)
            _, pred_boxes = _extract_count_and_boxes(task, pred)
        elif task in {"VQA3", "视觉定位"}:
            gt_boxes = gt if isinstance(gt, list) else (gt or [])
            pred_boxes = pred if isinstance(pred, list) else (pred or [])
        else:
            return  # 非检测任务直接忽略

        gt_boxes = gt_boxes or []
        pred_boxes = pred_boxes or []

        self.gts.setdefault(source, []).extend(gt_boxes)
        self.preds.setdefault(source, []).extend(pred_boxes)

    def compute(self) -> float:
        """
        在单一 IoU 阈值下计算 AP（score=1 的退化版）：
            1) 统计正样本数 npos
            2) 展开所有预测（score 固定为 1）
            3) 贪心匹配：每个预测找 IoU 最高且未匹配的 gt，IoU>=阈值记 TP，否则 FP
            4) 累积 TP/FP 得到单点 PR，再积分得 AP
        """
        npos = sum(len(boxes) for boxes in self.gts.values())

        detections: List[Tuple[str, Any, float]] = []
        for image_id, boxes in self.preds.items():
            for b in boxes:
                detections.append((image_id, b, 1.0))  # 目前无 score，固定 1.0

        if npos == 0 or not detections:
            return 0.0

        # 如有 score，可在此排序：detections.sort(key=lambda x: x[2], reverse=True)

        gt_matched: Dict[str, List[bool]] = {
            img_id: [False] * len(boxes) for img_id, boxes in self.gts.items()
        }

        tp: List[int] = []
        fp: List[int] = []

        for image_id, pred_box, score in detections:
            gt_boxes = self.gts.get(image_id, [])
            if image_id not in gt_matched:
                gt_matched[image_id] = [False] * len(gt_boxes)

            best_iou = 0.0
            best_gt_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if gt_matched[image_id][idx]:
                    continue
                iou = self._iou_fn(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                gt_matched[image_id][best_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # 累积 TP/FP，得到单点 PR
        tp_cum: List[int] = []
        fp_cum: List[int] = []
        c_tp = 0
        c_fp = 0
        for t, f in zip(tp, fp):
            c_tp += t
            c_fp += f
            tp_cum.append(c_tp)
            fp_cum.append(c_fp)

        recalls: List[float] = []
        precisions: List[float] = []
        for t_c, f_c in zip(tp_cum, fp_cum):
            recall = t_c / npos
            precision = t_c / max(t_c + f_c, 1e-8)
            recalls.append(recall)
            precisions.append(precision)

        return self._compute_ap(recalls, precisions)

    @staticmethod
    def _compute_ap(recalls: List[float], precisions: List[float]) -> float:
        """
        VOC/COCO 风格 AP 计算：后向包络 + 积分
        """
        if not recalls:
            return 0.0

        # 在首尾补 (0,0) 和 (1,0) 便于积分
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]

        # 后向包络，确保 precision 单调不增
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # 逐段累积面积
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        return ap


class DetectionMAPMetric(DetectionAPMetric):
    """
    多阈值 mAP（COCO 风格），复用 DetectionAPMetric 的数据累积方式
    - iou_thresholds: 阈值列表，如 [0.5,0.55,...,0.95]
    - compute: 对每个阈值计算 AP，取平均
    """

    def __init__(
        self,
        name: str,
        iou_thresholds: List[float],
        box_mode: str = "bbox",
        is_aux: bool = False,
    ) -> None:
        self.iou_thresholds = iou_thresholds
        super().__init__(name=name, iou_threshold=0.5, box_mode=box_mode, is_aux=is_aux)

    def compute(self) -> float:
        if not self.iou_thresholds:
            return 0.0
        aps = []
        for thr in self.iou_thresholds:
            # 临时替换阈值计算单点 AP
            self.iou_threshold = thr
            aps.append(super().compute())
        # 恢复主阈值避免副作用（非必要，但保持安全）
        self.iou_threshold = self.iou_thresholds[0]
        return sum(aps) / len(aps)


class DetectionCountAccuracy(BaseMetric):
    """
    数量准确率：预测数量 == gt 数量 计为正确
    适用：水平/旋转区域检测、VQA3
    """

    def __init__(self, name: str = "count_acc", is_aux: bool = True) -> None:
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if task not in {"水平区域检测", "旋转区域检测", "VQA3"}:
            return
        gt_count, _ = _extract_count_and_boxes(task, gt)
        pred_count, _ = _extract_count_and_boxes(task, pred)

        # 两边都 0 且无框信息时跳过（避免把“全空”计入统计）
        if gt_count == 0 and pred_count == 0:
            return

        self.total += 1
        if gt_count == pred_count:
            self.correct += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class DetectionCountMAE(BaseMetric):
    """
    数量 MAE：|pred_count - gt_count| 的平均值
    适用：水平/旋转区域检测、VQA3
    """

    def __init__(self, name: str = "count_mae", is_aux: bool = True) -> None:
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.errors: List[float] = []

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if task not in {"水平区域检测", "旋转区域检测", "VQA3"}:
            return
        gt_count, _ = _extract_count_and_boxes(task, gt)
        pred_count, _ = _extract_count_and_boxes(task, pred)

        # 两边都 0 且无框信息时跳过
        if gt_count == 0 and pred_count == 0:
            return

        self.errors.append(abs(pred_count - gt_count))

    def compute(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)


def build_detection_metrics(task: str) -> MetricCollection:
    """
    构建检测类指标集合：
        - 水平 / 旋转区域检测：
            核心：AP@0.5
            辅助：mAP(0.5:0.95) + AP@0.75 + count_acc + count_mae
        - VQA3：
            核心：AP@0.5
            辅助：mAP(0.5:0.95) + AP@0.75 + count_acc + count_mae
        - 视觉定位：
            核心：AP@0.5
            辅助：AP@0.25 + mAP(0.5:0.95)
    """
    metrics: List[BaseMetric] = []
    coco_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5:0.05:0.95

    if task == "水平区域检测":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionMAPMetric(
                name="map",
                iou_thresholds=coco_thresholds,
                box_mode="bbox",
                is_aux=True,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="bbox",
                is_aux=True,
            )
        )
        metrics.append(DetectionCountAccuracy(name="count_acc", is_aux=True))
        metrics.append(DetectionCountMAE(name="count_mae", is_aux=True))

    elif task == "旋转区域检测":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="quad",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionMAPMetric(
                name="map",
                iou_thresholds=coco_thresholds,
                box_mode="quad",
                is_aux=True,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="quad",
                is_aux=True,
            )
        )
        metrics.append(DetectionCountAccuracy(name="count_acc", is_aux=True))
        metrics.append(DetectionCountMAE(name="count_mae", is_aux=True))

    elif task == "VQA3":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionMAPMetric(
                name="map",
                iou_thresholds=coco_thresholds,
                box_mode="bbox",
                is_aux=True,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_75",
                iou_threshold=0.75,
                box_mode="bbox",
                is_aux=True,
            )
        )
        metrics.append(DetectionCountAccuracy(name="count_acc", is_aux=True))
        metrics.append(DetectionCountMAE(name="count_mae", is_aux=True))

    elif task == "视觉定位":
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_5",
                iou_threshold=0.5,
                box_mode="bbox",
                is_aux=False,
            )
        )
        metrics.append(
            DetectionAPMetric(
                name="ap_iou_0_25",
                iou_threshold=0.25,
                box_mode="bbox",
                is_aux=True,
            )
        )
        metrics.append(
            DetectionMAPMetric(
                name="map",
                iou_thresholds=coco_thresholds,
                box_mode="bbox",
                is_aux=True,
            )
        )

    # 非检测任务不会走到这里
    return MetricCollection(metrics)
