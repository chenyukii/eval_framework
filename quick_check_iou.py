from utils.iou import bbox_iou, quad_iou


def test_bbox_iou():
    """
    测试 1：轴对齐 bbox IoU

    案例：
        box1 = (0,0,2,2)
        box2 = (1,1,3,3)

        交集区域 = 1x1 = 1
        每个面积 = 2x2 = 4
        并集 = 4 + 4 - 1 = 7
        IoU = 1 / 7 ≈ 0.142857
    """
    box1 = (0, 0, 2, 2)
    box2 = (1, 1, 3, 3)
    iou = bbox_iou(box1, box2)
    print("bbox IoU:", iou)
    assert abs(iou - (1.0 / 7.0)) < 1e-6
    print("bbox IoU 测试通过 ✅")


def test_quad_iou_same():
    """
    测试 2：完全相同的旋转矩形，IoU 应为 1

    我们构造一个未旋转的正方形：
        q = (0,0,2,0,2,2,0,2)
    """
    q = (0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0)
    iou = quad_iou(q, q)
    print("quad IoU (same):", iou)
    assert abs(iou - 1.0) < 1e-6
    print("相同旋转框 IoU 测试通过 ✅")


def test_quad_iou_shift():
    """
    测试 3：两个水平矩形平移一部分，四点表示的 IoU 是否合理。

    q1: (0,0)-(2,0)-(2,2)-(0,2)
    q2: (1,0)-(3,0)-(3,2)-(1,2)

    理论上：
        交集 = 1x2 = 2
        面积 = 4, 4
        并集 = 4+4-2 = 6
        IoU = 2/6 = 1/3 ≈ 0.3333
    """
    q1 = (0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0)
    q2 = (1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0)
    iou = quad_iou(q1, q2)
    print("quad IoU (shift):", iou)
    assert abs(iou - (1.0 / 3.0)) < 1e-6
    print("平移旋转框 IoU 测试通过 ✅")


def test_quad_iou_rotated():
    """
    测试 4：一个正方形 + 一个绕中心旋转 45 度的正方形，IoU 应该在 (0,1) 之间。

    这里我们不要求精确值，只要求在一个合理范围内：
        0.5 < IoU < 0.9
    """
    # 未旋转的正方形
    q1 = (0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0)

    # 绕中心 (1,1) 旋转 45 度后的四个顶点
    # 这里直接写死一个预先算好的例子即可
    q2 = (
        1.0, -0.4142135624,
        2.4142135624, 1.0,
        1.0, 2.4142135624,
        -0.4142135624, 1.0,
    )

    iou = quad_iou(q1, q2)
    print("quad IoU (rotated):", iou)
    assert 0.5 < iou < 0.9
    print("旋转正方形 IoU 测试通过 ✅")


if __name__ == "__main__":
    test_bbox_iou()
    test_quad_iou_same()
    test_quad_iou_shift()
    test_quad_iou_rotated()
