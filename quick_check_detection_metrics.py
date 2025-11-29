from metrics.detection import build_detection_metrics


def test_horizontal_detection_perfect():
    """
    场景 1：水平区域检测，所有框预测完全正确，AP 应该为 1.0
    """
    metrics = build_detection_metrics("水平区域检测")

    # 样本 1：1 个 gt，1 个完全匹配的预测框
    gt1 = (1, [(0.0, 0.0, 2.0, 2.0)])
    pred1 = (1, [(0.0, 0.0, 2.0, 2.0)])
    metrics.update(gt=gt1, pred=pred1, task="水平区域检测", source="img1")

    # 样本 2：2 个 gt，2 个完全匹配的预测框
    gt2 = (2, [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)])
    pred2 = (2, [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)])
    metrics.update(gt=gt2, pred=pred2, task="水平区域检测", source="img2")

    summary = metrics.compute_all()
    print("水平区域检测 AP 指标:", summary)

    assert abs(summary["ap_iou_0_5"] - 1.0) < 1e-6
    assert abs(summary["ap_iou_0_75"] - 1.0) < 1e-6

    print("水平区域检测 AP 测试通过 ✅")


def test_rotated_detection_perfect():
    """
    场景 2：旋转区域检测，gt 和预测的四点框完全一致，AP 也应为 1.0
    """
    metrics = build_detection_metrics("旋转区域检测")

    # 用一个未旋转的方框来测试旋转 IoU 逻辑
    q = (0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0)
    gt = (1, [q])
    pred = (1, [q])

    metrics.update(gt=gt, pred=pred, task="旋转区域检测", source="img_rot")

    summary = metrics.compute_all()
    print("旋转区域检测 AP 指标:", summary)

    assert abs(summary["ap_iou_0_5"] - 1.0) < 1e-6
    assert abs(summary["ap_iou_0_75"] - 1.0) < 1e-6

    print("旋转区域检测 AP 测试通过 ✅")


def test_visual_loc_mixed():
    """
    场景 3：视觉定位，1 个正确框 + 1 个严重偏离的错误框，
    AP@0.5 应该介于 0 和 1 之间。
    """
    metrics = build_detection_metrics("视觉定位")

    # 样本 1：预测完全正确
    gt1 = [(0.0, 0.0, 2.0, 2.0)]
    pred1 = [(0.0, 0.0, 2.0, 2.0)]
    metrics.update(gt=gt1, pred=pred1, task="视觉定位", source="img_loc_1")

    # 样本 2：预测框与 gt 几乎不重叠（IoU 很小）
    gt2 = [(0.0, 0.0, 2.0, 2.0)]
    pred2 = [(3.0, 3.0, 5.0, 5.0)]
    metrics.update(gt=gt2, pred=pred2, task="视觉定位", source="img_loc_2")

    summary = metrics.compute_all()
    print("视觉定位 AP 指标:", summary)

    assert 0.0 < summary["ap_iou_0_5"] < 1.0
    print("视觉定位 AP 测试通过 ✅")


if __name__ == "__main__":
    test_horizontal_detection_perfect()
    test_rotated_detection_perfect()
    test_visual_loc_mixed()
