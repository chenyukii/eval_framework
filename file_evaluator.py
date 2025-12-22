# file_evaluator.py
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple, Optional

from .data.loader import DataLoader
from .evaluator import EvaluationCore
from .utils.logger import get_logger, log_struct

logger = get_logger(__name__)


class FileEvaluationRunner:
    """
    文件级评估封装：
    输入：标注文件路径 + 模型输出文件路径 + task
    内部：
        - DataLoader 负责：读取文件 + 验证样本 + 配对样本
        - EvaluationCore 负责：解析 gt/model_output + 指标计算
    输出：指标结果 + 一些样本统计信息 + （可选）每条样本指标
    """

    def __init__(
        self,
        error_log_path: str = "./error_log.txt",
        invalid_sample_log_path: str = "./invalid_sample_log.txt",
    ) -> None:
        # DataLoader 内部会自带 DataParser & DataValidator
        self.loader = DataLoader(
            error_log_path=error_log_path,
            invalid_sample_log_path=invalid_sample_log_path,
        )
        # EvaluationCore 内部持有一个 DataParser + 调用各类 metrics
        self.core = EvaluationCore()

    def run_for_files(
        self,
        task: str,
        annotation_file: str,
        model_output_file: str,
        *,
        calc_aux_metric: bool = True,
        collect_sample_details: bool = False,
    ) -> Dict[str, Any]:
        """
        对单个标注文件和单个模型输出文件做评估。

        collect_sample_details:
            - False（默认）：仅返回整体指标，性能最佳
            - True：额外返回每条配对样本的指标明细，便于定位差样本
        """
        # 1. 加载 + 验证 标注文件
        anno_valid, anno_invalid = self.loader.load_and_validate_annotation_file(
            annotation_file
        )

        # 2. 加载 + 验证 模型输出文件
        model_valid, model_invalid = self.loader.load_and_validate_model_output_file(
            model_output_file
        )

        # 3. 配对样本（只用“有效”的样本做配对）
        paired_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = \
            self.loader.pair_samples(anno_valid, model_valid)

        # 4. 调用 EvaluationCore，走“原始样本对 -> 解析 -> 指标”这一条链
        sample_details: Optional[List[Dict[str, Any]]] = [] if collect_sample_details else None
        metrics = self.core.evaluate_raw_paired_samples(
            task=task,
            paired_samples=paired_samples,
            calc_aux_metric=calc_aux_metric,
            sample_details=sample_details,
        )

        # 5. 顺便把一些统计信息也返回，方便后续展示
        stats = {
            "num_anno_valid": len(anno_valid),
            "num_anno_invalid": len(anno_invalid),
            "num_model_valid": len(model_valid),
            "num_model_invalid": len(model_invalid),
            "num_paired": len(paired_samples),
        }

        logger.info(
            "单文件评估完成 | task=%s | 指标=%s | 统计=%s",
            task,
            metrics,
            stats,
        )
        if anno_invalid or model_invalid:
            log_struct(
                stage="evaluate",
                error_code="INVALID_SAMPLES_PRESENT",
                message="存在无效样本，已跳过。",
                level="WARNING",
                logger=logger,
                anno_invalid=len(anno_invalid),
                model_invalid=len(model_invalid),
            )

        result: Dict[str, Any] = {
            "metrics": metrics,
            "stats": stats,
        }
        if sample_details is not None:
            result["details"] = sample_details  # 与输入配对顺序一致

        return result


def run_evaluation_for_file(
    task: str,
    annotation_file: str,
    model_output_file: str,
    *,
    calc_aux_metric: bool = True,
    collect_sample_details: bool = False,
) -> Dict[str, Any]:
    """
    单文件便捷函数：大多数情况下外部只用这一行就够了。
    """
    runner = FileEvaluationRunner()
    return runner.run_for_files(
        task=task,
        annotation_file=annotation_file,
        model_output_file=model_output_file,
        calc_aux_metric=calc_aux_metric,
        collect_sample_details=collect_sample_details,
    )


# ======================================================================
#                   目录级评估（流式，内存友好）
# ======================================================================


class DirEvaluationRunner:
    """
    目录级评估：
    - anno_dir 下可以有多个标注文件（默认 *.txt）
    - model_dir 下放对应的模型输出文件（默认“同名匹配”）
    - 使用同一份 metrics 对整个目录 streaming 更新，避免一次性吃爆内存
    """

    def __init__(
        self,
        error_log_path: str = "./error_log_dir.txt",
        invalid_sample_log_path: str = "./invalid_sample_log_dir.txt",
    ) -> None:
        self.loader = DataLoader(
            error_log_path=error_log_path,
            invalid_sample_log_path=invalid_sample_log_path,
        )
        self.core = EvaluationCore()

    def _find_model_file(self, annotation_file: str, model_dir: str) -> str | None:
        """
        默认策略：同名匹配
        """
        base = os.path.basename(annotation_file)
        candidate = os.path.join(model_dir, base)
        return candidate if os.path.exists(candidate) else None

    def run_for_dir(
        self,
        task: str,
        anno_dir: str,
        model_dir: str,
        pattern: str = "*.txt",
        *,
        recursive: bool = False,
        calc_aux_metric: bool = True,
        collect_sample_details: bool = False,
    ) -> Dict[str, Any]:
        """
        对一个目录下的多对标注 / 模型输出文件做“整体评估”。
        collect_sample_details=True 时，会收集跨文件的所有配对样本指标（可能较大）。
        """
        # 1. 列出所有标注文档
        if recursive:
            glob_pattern = os.path.join(anno_dir, "**", pattern)
            anno_files = glob.glob(glob_pattern, recursive=True)
        else:
            glob_pattern = os.path.join(anno_dir, pattern)
            anno_files = glob.glob(glob_pattern)

        anno_files = sorted(anno_files)

        # 2. 构建全局 metrics（贯穿整个目录评估）
        metrics = self.core.build_metrics(task, calc_aux_metric=calc_aux_metric)

        # 3. 全局统计信息
        stats: Dict[str, Any] = {
            "num_anno_valid": 0,
            "num_anno_invalid": 0,
            "num_model_valid": 0,
            "num_model_invalid": 0,
            "num_paired": 0,
            "num_files_total": len(anno_files),
            "num_files_evaluated": 0,
            "num_files_missing_model": 0,
        }

        # 4. 逐文件 streaming 更新
        all_sample_details: Optional[List[Dict[str, Any]]] = [] if collect_sample_details else None
        for anno_file in anno_files:
            model_file = self._find_model_file(anno_file, model_dir)
            if model_file is None:
                stats["num_files_missing_model"] += 1
                logger.warning("[警告] 找不到对应模型输出文件，跳过：%s", anno_file)
                log_struct(
                    stage="match",
                    error_code="MODEL_FILE_MISSING",
                    message="目录模式缺少对应模型输出文件",
                    level="WARNING",
                    logger=logger,
                    anno_file=anno_file,
                )
                continue

            logger.info("[INFO] 评估文件对：%s  vs  %s", anno_file, model_file)

            anno_valid, anno_invalid = self.loader.load_and_validate_annotation_file(
                anno_file
            )
            model_valid, model_invalid = self.loader.load_and_validate_model_output_file(
                model_file
            )
            paired_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = \
                self.loader.pair_samples(anno_valid, model_valid)

            stats["num_anno_valid"] += len(anno_valid)
            stats["num_anno_invalid"] += len(anno_invalid)
            stats["num_model_valid"] += len(model_valid)
            stats["num_model_invalid"] += len(model_invalid)
            stats["num_paired"] += len(paired_samples)
            stats["num_files_evaluated"] += 1

            # 核心：对同一份 metrics 做流式更新（跨文件累积）
            self.core.update_metrics_with_raw_pairs(
                task=task,
                metric_collection=metrics,
                paired_samples=paired_samples,
            )

            # 如需明细，再额外跑一次（仅该文件的配对样本），避免污染全局 metrics
            if all_sample_details is not None and paired_samples:
                file_details: List[Dict[str, Any]] = []
                self.core.evaluate_raw_paired_samples(
                    task=task,
                    paired_samples=paired_samples,
                    calc_aux_metric=calc_aux_metric,
                    sample_details=file_details,
                )
                all_sample_details.extend(file_details)

            # 用完就释放 per-file 列表引用，便于 GC
            del anno_valid, anno_invalid, model_valid, model_invalid, paired_samples

        # 5. 所有文件处理完之后 compute 一次，全局指标
        metric_result = metrics.compute_all()

        logger.info(
            "目录评估完成 | task=%s | 指标=%s | 统计=%s",
            task,
            metric_result,
            stats,
        )
        if stats["num_files_missing_model"] > 0:
            log_struct(
                stage="evaluate",
                error_code="MISSING_MODEL_FILES",
                message="部分标注文件缺少对应模型输出文件",
                level="WARNING",
                logger=logger,
                missing_files=stats["num_files_missing_model"],
            )

        result: Dict[str, Any] = {
            "metrics": metric_result,
            "stats": stats,
        }
        if all_sample_details is not None:
            result["details"] = all_sample_details  # 顺序为遍历文件顺序内的配对顺序

        return result


def run_evaluation_for_dir(
    task: str,
    anno_dir: str,
    model_dir: str,
    *,
    pattern: str = "*.txt",
    recursive: bool = False,
    calc_aux_metric: bool = True,
    collect_sample_details: bool = False,
) -> Dict[str, Any]:
    """
    目录级便捷函数：
    Example:
        result = run_evaluation_for_dir(
            task="VQA2",
            anno_dir="data/anno",
            model_dir="data/model",
            collect_sample_details=True,
        )
    """
    runner = DirEvaluationRunner()
    return runner.run_for_dir(
        task=task,
        anno_dir=anno_dir,
        model_dir=model_dir,
        pattern=pattern,
        recursive=recursive,
        calc_aux_metric=calc_aux_metric,
        collect_sample_details=collect_sample_details,
    )
