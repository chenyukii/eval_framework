# quick_check_config_eval.py
import json
import os
import subprocess
import sys


def write_demo_vqa2_files():
    anno_path = "tmp_cfg_vqa2_anno.txt"
    model_path = "tmp_cfg_vqa2_output.txt"

    anno_samples = [
        {
            "prompt": "How many cars are there?",
            "frames": ["dummy_frame_1"],
            "gt": "3",
            "task": "VQA2",
            "source": "img1",
        },
        {
            "prompt": "How many buildings are visible?",
            "frames": ["dummy_frame_2"],
            "gt": "5",
            "task": "VQA2",
            "source": "img2",
        },
        {
            "prompt": "How many rivers are there?",
            "frames": ["dummy_frame_3"],
            "gt": "0",
            "task": "VQA2",
            "source": "img3",
        },
    ]

    model_samples = [
        {
            "sample_id": "img1",
            "task": "VQA2",
            "model_output": "3",  # 正确
            "source": "img1",
        },
        {
            "sample_id": "img2",
            "task": "VQA2",
            "model_output": "4",  # 差 1
            "source": "img2",
        },
        {
            "sample_id": "img3",
            "task": "VQA2",
            "model_output": "2",  # 差 2
            "source": "img3",
        },
    ]

    with open(anno_path, "w", encoding="utf-8") as f:
        for s in anno_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(model_path, "w", encoding="utf-8") as f:
        for s in model_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return anno_path, model_path


def write_demo_config(anno_path: str, model_path: str) -> str:
    cfg = {
        "task": "VQA2",
        "annotation_file": anno_path,
        "model_output_file": model_path,
        "calc_aux_metric": True,
        "output_json": "tmp_cfg_vqa2_result.json",
    }
    cfg_path = "eval_config_demo.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg_path


def main():
    anno_path, model_path = write_demo_vqa2_files()
    cfg_path = write_demo_config(anno_path, model_path)

    cmd = [sys.executable, "eval_main.py", "--config", cfg_path]
    print("运行命令：", " ".join(cmd))

    # 让子进程直接输出到当前控制台，避免编码问题
    proc = subprocess.run(cmd)

    assert proc.returncode == 0, "eval_main.py 返回码非 0，配置化测试失败"

    # 检查输出 JSON
    result_path = "tmp_cfg_vqa2_result.json"
    assert os.path.exists(result_path), "未找到结果 JSON 文件"

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    assert "metrics" in result and "stats" in result, "结果 JSON 结构不正确"
    print("配置化 CLI 测试通过 ✅")


if __name__ == "__main__":
    main()
