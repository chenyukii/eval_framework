# quick_check_file_evaluation.py
import json
import os

from file_evaluator import run_evaluation_for_file


def write_demo_vqa2_files() -> tuple[str, str]:
    """
    自动在当前目录生成两个临时文件：

        - tmp_vqa2_anno.txt         （标注）
        - tmp_vqa2_output.txt       （模型输出）

    格式都符合 DataLoader.load_*_file 的要求：
        - 每行一个 JSON
        - 字段齐全
    """
    anno_path = "tmp_vqa2_anno.txt"
    model_path = "tmp_vqa2_output.txt"

    # 造 3 条标注样本（和我们之前 VQA2 的例子保持一致）
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

    # 对应的模型输出样本
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

    # 写标注文件（每行一个 JSON）
    with open(anno_path, "w", encoding="utf-8") as f:
        for sample in anno_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 写模型输出文件（每行一个 JSON）
    with open(model_path, "w", encoding="utf-8") as f:
        for sample in model_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return anno_path, model_path


def main():
    anno_path, model_path = write_demo_vqa2_files()

    result = run_evaluation_for_file(
        task="VQA2",
        annotation_file=anno_path,
        model_output_file=model_path,
        calc_aux_metric=True,
    )

    print("评估结果:", result)

    metrics = result["metrics"]
    stats = result["stats"]

    # 简单检查：有 accuracy / abs_error 这些 key
    assert "accuracy" in metrics
    # 如果你在 VQA2 指标里把绝对误差叫 "abs_error" 或 "mae"，可以二选一检查一下：
    assert any(k in metrics for k in ["abs_error", "mae"])

    print("统计信息:", stats)
    print("文件级评估测试通过 ✅")

    # 测完你可以选择把临时文件删掉
    # os.remove(anno_path)
    # os.remove(model_path)


if __name__ == "__main__":
    main()
