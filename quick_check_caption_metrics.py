from metrics.caption import calculate_caption_metrics


def run_case(task: str):
    # 三条 gt caption（英文，符合需求文档的“遥感场景”风格）
    gt_list = [
        "a small harbor with several boats docked near the coastline",
        "a dense urban area with high rise buildings and wide roads",
        "farmland fields divided into rectangular plots with a river nearby",
    ]

    # case 1：预测 = gt（理想情况）
    pred_perfect = list(gt_list)

    # case 2：预测比较差（内容完全不相关）
    pred_bad = [
        "a red car parked on a street in the city center",
        "a group of people playing football on a green field",
        "a close up photo of a cat sitting on a sofa",
    ]

    metrics_perfect = calculate_caption_metrics(task, gt_list, pred_perfect)
    metrics_bad = calculate_caption_metrics(task, gt_list, pred_bad)

    print(f"=== 任务：{task} ===")
    print("完美匹配：", metrics_perfect)
    print("错误描述：", metrics_bad)

    # 理论上，完美匹配的指标应该明显更高
    assert metrics_perfect["cider"] > metrics_bad["cider"]
    assert metrics_perfect["rouge_l"] > metrics_bad["rouge_l"]
    assert metrics_perfect["bleu_4"] > metrics_bad["bleu_4"]
    assert metrics_perfect["meteor"] > metrics_bad["meteor"]
    print("caption 指标 sanity check 通过 ✅")


if __name__ == "__main__":
    # 随便选一个描述任务来测（这三个任务用的是同一套指标）
    run_case("简洁图片描述")
