"""
批量跑 config 目录下的多个评估配置，并汇总指标。

用法：
    python run_all_configs.py                   # 扫描 config 下的 cfg*.json
    python run_all_configs.py --pattern cfg_    # 指定文件名前缀
    python run_all_configs.py --config_dir path # 指定配置目录
说明：
    - 每个配置会调用 eval_main 运行（模块方式）。
    - 若配置里未设置 output_json，会自动在当前目录生成 <cfg_stem>_result.json。
    - 汇总表只收集 output_json 中的 metrics 字段。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

DEFAULT_PATTERN = "cfg_"  # 只跑以这个前缀开头的 json


def run_config(cfg_path: Path) -> Tuple[int, Path | None]:
    """执行单个配置，返回 (return_code, result_json_path)。"""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] 读取配置失败 {cfg_path}: {e}")
        return 1, None

    out_path = cfg.get("output_json")
    if not out_path:
        out_path = f"{cfg_path.stem}_result.json"
        cfg["output_json"] = out_path
        tmp_cfg_path = cfg_path.with_suffix(".tmp.json")
        tmp_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg_path_to_use = tmp_cfg_path
    else:
        cfg_path_to_use = cfg_path

    # 以模块方式运行，避免相对导入错误
    cmd = [sys.executable, "-m", "eval_framework.eval_main", "--config", str(cfg_path_to_use)]
    print(f"\n[INFO] 运行：{' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if cfg_path_to_use != cfg_path and cfg_path_to_use.exists():
        cfg_path_to_use.unlink(missing_ok=True)  # 清理临时配置
    return proc.returncode, Path(out_path) if out_path else None


def collect_metrics(result_path: Path) -> Dict[str, Any]:
    if not result_path.exists():
        print(f"[WARN] 未找到结果文件：{result_path}")
        return {}
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return data.get("metrics", {})
    except Exception as e:
        print(f"[WARN] 读取结果失败 {result_path}: {e}")
        return {}


def main() -> int:
        parser = argparse.ArgumentParser(description="批量运行配置并汇总指标")
        parser.add_argument("--config_dir", default="config", help="配置文件所在目录")
        parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="只跑此前缀的 .json 文件")
        args = parser.parse_args()

        cfg_dir = Path(args.config_dir)
        if not cfg_dir.is_dir():
            print(f"[ERROR] 配置目录不存在：{cfg_dir}")
            return 1

        cfg_files = sorted(p for p in cfg_dir.glob("*.json") if p.name.startswith(args.pattern))
        if not cfg_files:
            print(f"[WARN] 未找到匹配的配置文件（前缀 {args.pattern}）")
            return 0

        summary: List[Tuple[str, Dict[str, Any]]] = []
        for cfg in cfg_files:
            code, result_path = run_config(cfg)
            if code != 0:
                print(f"[ERROR] 运行失败：{cfg} (code={code})")
                continue
            metrics = collect_metrics(result_path) if result_path else {}
            summary.append((cfg.name, metrics))

        print("\n======= 汇总（按配置文件） =======")
        for name, metrics in summary:
            print(f"\n{name}")
            if not metrics:
                print("  无指标或读取失败")
            else:
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
        print("================================")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
