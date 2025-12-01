# metrics/segmentation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BaseMetric, MetricCollection
from .classification import build_classification_metrics
from ..utils.iou import polygon_iou, polys_union_area

Polygon = Sequence[Tuple[float, float]]


def _to_polygon_list(value: Any) -> List[Polygon]:
    """将解析后的多边形结果规整为列表"""
    if value is None:
        return []
    if isinstance(value, list):
        # 确保元素都是序列
        result: List[Polygon] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                result.append(tuple(item))
        return result
    return []


class UnionIoUMetric(BaseMetric):
    """
    样本级并集 IoU（用于语义分割/变化检测）
    计算每个样本 gt 与 pred 多边形集合的并集 IoU，再对样本求平均
    """

    def reset(self) -> None:
        self.ious: List[float] = []

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return  # 双空不计入
        union_area = polys_union_area(list(gt_polys) + list(pred_polys))
        if union_area <= 0.0:
            return
        inter_area = (
            polys_union_area(gt_polys)
            + polys_union_area(pred_polys)
            - union_area
        )
        iou = inter_area / union_area if union_area > 0 else 0.0
        self.ious.append(max(iou, 0.0))

    def compute(self) -> float:
        if not self.ious:
            return 0.0
        return sum(self.ious) / len(self.ious)


class DiceMetric(BaseMetric):
    """样本级 Dice 系数（2*inter/(sum areas))"""

    def reset(self) -> None:
        self.dices: List[float] = []

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        area_gt = polys_union_area(gt_polys)
        area_pred = polys_union_area(pred_polys)
        if area_gt <= 0.0 and area_pred <= 0.0:
            return
        inter_area = area_gt + area_pred - polys_union_area(gt_polys + pred_polys)
        denom = area_gt + area_pred
        dice = 2 * inter_area / denom if denom > 0 else 0.0
        self.dices.append(max(dice, 0.0))

    def compute(self) -> float:
        if not self.dices:
            return 0.0
        return sum(self.dices) / len(self.dices)


class PolyAPMetric(BaseMetric):
    """
    多边形检测/实例分割 mAP（退化单类，score=1）
    - 支持多阈值（默认 COCO 0.5:0.05:0.95），compute 返回均值
    - 可用于输出单点 AP（thresholds=[0.5] 等）
    """

    def __init__(
        self,
        name: str,
        iou_thresholds: List[float],
        is_aux: bool = False,
    ) -> None:
        self.iou_thresholds = iou_thresholds
        super().__init__(name=name, is_aux=is_aux)

    def reset(self) -> None:
        self.gts: Dict[str, List[Polygon]] = {}
        self.preds: Dict[str, List[Polygon]] = {}

    def update(
        self,
        *,
        gt: Any,
        pred: Any,
        task: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if source is None:
            return
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        self.gts.setdefault(source, []).extend(gt_polys)
        self.preds.setdefault(source, []).extend(pred_polys)

    def _compute_ap_single(self, thr: float) -> float:
        npos = sum(len(v) for v in self.gts.values())
        detections: List[Tuple[str, Polygon]] = []
        for img_id, polys in self.preds.items():
            for poly in polys:
                detections.append((img_id, poly))

        if npos == 0 or not detections:
            return 0.0

        gt_matched: Dict[str, List[bool]] = {img: [False] * len(polys) for img, polys in self.gts.items()}
        tp: List[int] = []
        fp: List[int] = []

        # score 固定为 1，无需排序
        for image_id, pred_poly in detections:
            gt_polys = self.gts.get(image_id, [])
            if image_id not in gt_matched:
                gt_matched[image_id] = [False] * len(gt_polys)

            best_iou = 0.0
            best_idx = -1
            for idx, gt_poly in enumerate(gt_polys):
                if gt_matched[image_id][idx]:
                    continue
                iou = polygon_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= thr and best_idx >= 0:
                gt_matched[image_id][best_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # 累积 PR 曲线（单点，但仍按 VOC 风格积分）
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

        return self._voc_ap(recalls, precisions)

    @staticmethod
    def _voc_ap(recalls: List[float], precisions: List[float]) -> float:
        """VOC/COCO 风格 AP：后向包络 + 积分"""
        if not recalls:
            return 0.0
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    def compute(self) -> float:
        if not self.iou_thresholds:
            return 0.0
        aps = [self._compute_ap_single(thr) for thr in self.iou_thresholds]
        return sum(aps) / len(aps)


class CountAccuracy(BaseMetric):
    """数量准确率：len(pred) == len(gt)"""

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
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        self.total += 1
        if len(gt_polys) == len(pred_polys):
            self.correct += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class CountMAE(BaseMetric):
    """数量 MAE：|len(pred) - len(gt)| 的平均"""

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
        gt_polys = _to_polygon_list(gt)
        pred_polys = _to_polygon_list(pred)
        if not gt_polys and not pred_polys:
            return
        self.errors.append(abs(len(pred_polys) - len(gt_polys)))

    def compute(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)


def build_segmentation_metrics(task: str) -> MetricCollection:
    """
    像素级任务指标集合：
    - 语义分割/变化检测：miou(核心) + dice/map/ap@0.5/ap@0.75
    - 实例分割：map(核心) + ap@0.5/ap@0.75/count_acc/count_mae
    - 像素分类/细粒度识别：复用分类指标
    """
    task = task or ""
    # 分类型像素任务复用
    if task in {"像素分类", "细粒度识别"}:
        return build_classification_metrics("图片分类")  # 复用同一指标组合

    metrics: List[BaseMetric] = []
    coco_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5:0.05:0.95

    if task in {"语义分割", "变化检测"}:
        metrics.append(UnionIoUMetric(name="miou", is_aux=False))
        metrics.append(DiceMetric(name="dice", is_aux=True))
        metrics.append(PolyAPMetric(name="map", iou_thresholds=coco_thresholds, is_aux=True))
        metrics.append(PolyAPMetric(name="ap_iou_0_5", iou_thresholds=[0.5], is_aux=True))
        metrics.append(PolyAPMetric(name="ap_iou_0_75", iou_thresholds=[0.75], is_aux=True))
    elif task in {"实例分割"}:
        metrics.append(PolyAPMetric(name="map", iou_thresholds=coco_thresholds, is_aux=False))
        metrics.append(PolyAPMetric(name="ap_iou_0_5", iou_thresholds=[0.5], is_aux=True))
        metrics.append(PolyAPMetric(name="ap_iou_0_75", iou_thresholds=[0.75], is_aux=True))
        metrics.append(CountAccuracy(name="count_acc", is_aux=True))
        metrics.append(CountMAE(name="count_mae", is_aux=True))

    return MetricCollection(metrics)
