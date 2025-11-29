# eval_main.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from .file_evaluator import (
    run_evaluation_for_file,
    run_evaluation_for_dir,
)
from .metrics import SUPPORTED_TASKS, is_supported_task


def load_config(path: str) -> Dict[str, Any]:
    """
    加载配置文件（支持 JSON / YAML）。

    - .json       -> 用 json.load
    - .yml/.yaml  -> 用 yaml.safe_load（需要已经安装 pyyaml）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")

    ext = os.path.splitext(path)[1].lower()

    with open(path, "r", encoding="utf-8") as f:
        if ext in (".json", ""):
            cfg = json.load(f)
        elif ext in (".yml", ".yaml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "需要先安装 pyyaml 才能加载 YAML 配置文件：\n"
                    "    pip install pyyaml"
                ) from e
            cfg = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式：{ext}（只支持 .json / .yml / .yaml）")

    if not isinstance(cfg, dict):
        raise ValueError("配置文件的顶层结构必须是一个对象（JSON/YAML 字典）")

    return cfg


def parse_args() -> argparse.Namespace:
    """
    命令行参数定义：

    三种使用方式（二选一）：

    A）单文件评估 + 命令行参数
        --task / -t
        --annotation_file / -a
        --model_output_file / -m

    B）目录评估 + 命令行参数
        --task / -t
        --annotation_dir
        --model_output_dir
        [--pattern] [--recursive]

    C）配置文件
        --config config.json
        （其他参数可选，用来覆盖配置里的内容）
    """
    parser = argparse.ArgumentParser(
        description="多模态遥感大模型多任务评估脚本",
    )

    # 配置文件
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="配置文件路径（JSON 或 YAML），可在其中指定 task/输入文件/目录等参数",
    )

    # 任务名称（不再强制 required，可以在配置里给）
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=False,
        default=None,
        help=(
            "任务名称，例如："
            "图片分类、水平区域分类、旋转区域分类、"
            "VQA1、VQA2、计数、"
            "水平区域检测、旋转区域检测、VQA3、视觉定位、"
            "图片检索、简洁图片描述、详细图片描述、区域描述"
        ),
    )

    # ===== 单文件模式 =====
    parser.add_argument(
        "--annotation_file",
        "-a",
        type=str,
        default=None,
        help="【单文件模式】标注文件路径（每行一个 JSON 样本）",
    )

    parser.add_argument(
        "--model_output_file",
        "-m",
        type=str,
        default=None,
        help="【单文件模式】模型输出文件路径（每行一个 JSON，或 JSON 数组）",
    )

    # ===== 目录模式 =====
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="【目录模式】标注文件所在目录",
    )

    parser.add_argument(
        "--model_output_dir",
        type=str,
        default=None,
        help="【目录模式】模型输出文件所在目录（按同名文件匹配）",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=None,  # 让配置文件有机会覆盖默认值
        help="【目录模式】标注文件匹配模式（默认：*.txt，可在配置中设置）",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="【目录模式】是否递归遍历子目录（命令行指定时会覆盖配置）",
    )

    # ===== 通用选项 =====
    parser.add_argument(
        "--no_aux_metric",
        action="store_true",
        help="不计算辅助指标（目前只影响图片检索任务的 Recall/F1，命令行会覆盖配置）",
    )

    parser.add_argument(
        "--output_json",
        "-o",
        type=str,
        default=None,
        help="将评估结果（metrics + stats）保存到指定 JSON 文件路径（命令行会覆盖配置）",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 0. 先加载配置文件（如果有）
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"错误：加载配置文件失败：\n{e}")
            return 1

    # 1. 合并 task
    task = args.task or config.get("task")
    if not task:
        print("错误：未指定任务类型。请通过 --task 或配置文件中的 task 字段进行指定。")
        return 1

    # 2. 检查 task 是否支持
    if not is_supported_task(task):
        print(f"错误：不支持的任务类型：{task}")
        print("当前支持的任务类型包括：")
        print("  " + "、".join(sorted(SUPPORTED_TASKS)))
        return 1

    # 3. 合并文件/目录参数（命令行优先级高于配置）
    anno_file = args.annotation_file or config.get("annotation_file")
    model_file = args.model_output_file or config.get("model_output_file")

    anno_dir = args.annotation_dir or config.get("annotation_dir")
    model_dir = args.model_output_dir or config.get("model_output_dir")

    # pattern & recursive
    if args.pattern is not None:
        pattern = args.pattern
    else:
        pattern = config.get("pattern", "*.txt")

    recursive = bool(config.get("recursive", False))
    if args.recursive:
        recursive = True

    # calc_aux_metric：默认 True，可在配置里设置为 False，命令行 --no_aux_metric 再强制关掉
    calc_aux_metric = bool(config.get("calc_aux_metric", True))
    if args.no_aux_metric:
        calc_aux_metric = False

    # output_json：命令行优先
    output_json = args.output_json or config.get("output_json")

    # 4. 判断是单文件模式还是目录模式
    use_file_mode = anno_file is not None and model_file is not None
    use_dir_mode = anno_dir is not None and model_dir is not None

    if use_file_mode and use_dir_mode:
        print("错误：不能同时指定文件模式和目录模式。")
        print("请二选一：")
        print("  单文件模式：annotation_file + model_output_file（命令行或配置）")
        print("  目录模式  ：annotation_dir + model_output_dir（命令行或配置）")
        return 1

    if not use_file_mode and not use_dir_mode:
        print("错误：既没有提供文件路径，也没有提供目录路径。")
        print("请至少选择一种模式：")
        print("  单文件模式：annotation_file + model_output_file")
        print("  目录模式  ：annotation_dir + model_output_dir")
        return 1

    # ========== 单文件模式 ==========
    if use_file_mode:
        if not os.path.exists(anno_file):
            print(f"错误：标注文件不存在：{anno_file}")
            return 1

        if not os.path.exists(model_file):
            print(f"错误：模型输出文件不存在：{model_file}")
            return 1

        print("========== 评估配置（单文件模式） ==========")
        if args.config:
            print(f"配置文件      : {args.config}")
        print(f"任务类型      : {task}")
        print(f"标注文件      : {anno_file}")
        print(f"模型输出文件  : {model_file}")
        print(f"计算辅助指标  : {calc_aux_metric}")
        print("==========================================")

        result: Dict[str, Any] = run_evaluation_for_file(
            task=task,
            annotation_file=anno_file,
            model_output_file=model_file,
            calc_aux_metric=calc_aux_metric,
        )

    # ========== 目录模式 ==========
    else:
        if not os.path.isdir(anno_dir):
            print(f"错误：标注目录不存在或不是目录：{anno_dir}")
            return 1

        if not os.path.isdir(model_dir):
            print(f"错误：模型输出目录不存在或不是目录：{model_dir}")
            return 1

        print("========== 评估配置（目录模式） ==========")
        if args.config:
            print(f"配置文件      : {args.config}")
        print(f"任务类型      : {task}")
        print(f"标注目录      : {anno_dir}")
        print(f"模型输出目录  : {model_dir}")
        print(f"文件匹配模式  : {pattern}")
        print(f"递归子目录    : {recursive}")
        print(f"计算辅助指标  : {calc_aux_metric}")
        print("==========================================")

        result: Dict[str, Any] = run_evaluation_for_dir(
            task=task,
            anno_dir=anno_dir,
            model_dir=model_dir,
            pattern=pattern,
            recursive=recursive,
            calc_aux_metric=calc_aux_metric,
        )

    metrics: Dict[str, float] = result.get("metrics", {})
    stats: Dict[str, Any] = result.get("stats", {})

    # 5. 打印指标结果
    print("\n========== 评估结果（metrics） ==========")
    if not metrics:
        print("没有可用指标（可能是没有有效样本/配对样本）。")
    else:
        for name, value in metrics.items():
            print(f"{name:20s}: {value:.4f}")
    print("==========================================")

    # 6. 打印样本/文件统计信息
    print("\n========== 样本 / 文件统计（stats） ==========")
    if not stats:
        print("没有统计信息。")
    else:
        for name, value in stats.items():
            print(f"{name:20s}: {value}")
    print("============================================")

    # 7. 如果需要，将结果输出到 JSON 文件
    if output_json:
        out_path = output_json
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n已将评估结果保存到：{out_path}")
        except Exception as e:
            print(f"\n警告：写入 JSON 文件失败（{out_path}）：{e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
