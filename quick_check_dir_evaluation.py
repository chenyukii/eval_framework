# quick_check_dir_evaluation.py
import json
import os

from file_evaluator import run_evaluation_for_dir


def write_demo_vqa2_dirs() -> tuple[str, str]:
    """
    自动构造一个小的目录结构用于测试目录评估：

        ./tmp_dir_vqa2_anno/
            split1.txt
            split2.txt

        ./tmp_dir_vqa2_model/
            split1.txt
            split2.txt

    每个文件里 3 条 VQA2 样本（和之前 single-file 测试一致/类似）。
    """
    anno_dir = "tmp_dir_vqa2_anno"
    model_dir = "tmp_dir_vqa2_model"

    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --------- split1: 和之前的例子差不多 ---------
    anno_samples_1 = [
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

    model_samples_1 = [
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

    # --------- split2: 再造一批，数字随便改 ---------
    anno_samples_2 = [
        {
            "prompt": "How many trees are there?",
            "frames": ["dummy_frame_4"],
            "gt": "10",
            "task": "VQA2",
            "source": "img4",
        },
        {
            "prompt": "How many ships are visible?",
            "frames": ["dummy_frame_5"],
            "gt": "2",
            "task": "VQA2",
            "source": "img5",
        },
        {
            "prompt": "How many bridges are there?",
            "frames": ["dummy_frame_6"],
            "gt": "1",
            "task": "VQA2",
            "source": "img6",
        },
    ]

    model_samples_2 = [
        {
            "sample_id": "img4",
            "task": "VQA2",
            "model_output": "9",  # 差 1
            "source": "img4",
        },
        {
            "sample_id": "img5",
            "task": "VQA2",
            "model_output": "0",  # 差 2
            "source": "img5",
        },
        {
            "sample_id": "img6",
            "task": "VQA2",
            "model_output": "4",  # 差 3
            "source": "img6",
        },
    ]

    # 写 split1
    with open(os.path.join(anno_dir, "split1.txt"), "w", encoding="utf-8") as f:
        for s in anno_samples_1:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(os.path.join(model_dir, "split1.txt"), "w", encoding="utf-8") as f:
        for s in model_samples_1:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 写 split2
    with open(os.path.join(anno_dir, "split2.txt"), "w", encoding="utf-8") as f:
        for s in anno_samples_2:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(os.path.join(model_dir, "split2.txt"), "w", encoding="utf-8") as f:
        for s in model_samples_2:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return anno_dir, model_dir


def main():
    anno_dir, model_dir = write_demo_vqa2_dirs()

    result = run_evaluation_for_dir(
        task="VQA2",
        anno_dir=anno_dir,
        model_dir=model_dir,
        pattern="*.txt",
        recursive=False,
        calc_aux_metric=True,
    )

    print("目录评估结果:", result)

    metrics = result["metrics"]
    stats = result["stats"]

    # 简单 sanity check：有 accuracy 指标
    assert "accuracy" in metrics
    # 至少有一些样本被统计到
    assert stats["num_anno_valid"] > 0
    assert stats["num_model_valid"] > 0
    assert stats["num_paired"] > 0
    assert stats["num_files_total"] == 2
    assert stats["num_files_evaluated"] == 2
    assert stats["num_files_missing_model"] == 0

    print("目录评估流式测试通过 ✅")


if __name__ == "__main__":
    main()
