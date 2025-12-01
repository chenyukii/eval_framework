# utils/iou.py
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union as shapely_union

    _HAVE_SHAPELY = True
except Exception:
    _HAVE_SHAPELY = False

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]   # (x1, y1, x2, y2)
Quad = Tuple[float, float, float, float,
             float, float, float, float]   # (x1,y1,...,x4,y4)


def bbox_iou(box1: BBox, box2: BBox) -> float:
    """计算轴对齐边界框 IoU"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    area2 = max(x4 - x3, 0.0) * max(y4 - y3, 0.0)
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0

    return inter / union


def quad_to_polygon(quad: Quad) -> List[Point]:
    """8 维旋转框转为顶点列表"""
    x1, y1, x2, y2, x3, y3, x4, y4 = quad
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def _polygon_area_signed(points: Sequence[Point]) -> float:
    """多边形有符号面积，>0 逆时针，<0 顺时针"""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_area(points: Sequence[Point]) -> float:
    """多边形面积（绝对值），不要求闭合"""
    if _HAVE_SHAPELY:
        if len(points) < 3:
            return 0.0
        try:
            return float(ShapelyPolygon(points).area)
        except Exception:
            return 0.0
    return abs(_polygon_area_signed(points))


def _sutherland_hodgman_clip(
    subject_polygon: List[Point],
    clip_polygon: List[Point],
) -> List[Point]:
    """
    Sutherland-Hodgman 多边形裁剪算法
    用 clip_polygon 对 subject_polygon 裁剪，返回交集多边形顶点列表
    """
    if not subject_polygon or not clip_polygon:
        return []

    area = _polygon_area_signed(clip_polygon)
    is_ccw = area > 0  # >0 表示逆时针

    def inside(p: Point, cp1: Point, cp2: Point) -> bool:
        cross = (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0])
        return cross >= -1e-9 if is_ccw else cross <= 1e-9

    def intersection(s: Point, e: Point, cp1: Point, cp2: Point) -> Point:
        x1, y1 = s
        x2, y2 = e
        x3, y3 = cp1
        x4, y4 = cp2

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return e  # 近似平行，返回终点

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output_list = subject_polygon
    for i in range(len(clip_polygon)):
        input_list = output_list
        output_list = []
        if not input_list:
            break

        cp1 = clip_polygon[i]
        cp2 = clip_polygon[(i + 1) % len(clip_polygon)]
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output_list.append(intersection(s, e, cp1, cp2))
                output_list.append(e)
            elif inside(s, cp1, cp2):
                output_list.append(intersection(s, e, cp1, cp2))
            s = e

    return output_list


def _shapely_polygon_union(polys: List[Sequence[Point]]) -> float:
    """使用 shapely 计算并集面积"""
    shapes = []
    for pts in polys:
        if len(pts) < 3:
            continue
        try:
            shapes.append(ShapelyPolygon(pts))
        except Exception:
            continue
    if not shapes:
        return 0.0
    return float(shapely_union(shapes).area)


def _polygon_intersection_area(poly1: Sequence[Point], poly2: Sequence[Point]) -> float:
    """两多边形交集面积，优先 shapely，回退 Sutherland-Hodgman"""
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0
    if _HAVE_SHAPELY:
        try:
            inter = ShapelyPolygon(poly1).intersection(ShapelyPolygon(poly2))
            return float(inter.area)
        except Exception:
            pass
    inter_poly = _sutherland_hodgman_clip(list(poly1), list(poly2))
    return polygon_area(inter_poly)


def polygon_iou(poly1: Sequence[Point], poly2: Sequence[Point]) -> float:
    """多边形 IoU（任意顶点数，不要求凸，多边形应闭合有序）"""
    inter = _polygon_intersection_area(poly1, poly2)
    if inter <= 0.0:
        return 0.0
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def polys_union_area(polys: List[Sequence[Point]]) -> float:
    """多边形集合并集面积"""
    if _HAVE_SHAPELY:
        try:
            return _shapely_polygon_union(polys)
        except Exception:
            pass
    # 回退：逐个累积并集面积（近似）
    if not polys:
        return 0.0
    union = list(polys[0])
    area = polygon_area(union)
    for poly in polys[1:]:
        inter_poly = _sutherland_hodgman_clip(union, list(poly))
        inter_area = polygon_area(inter_poly)
        area = area + polygon_area(poly) - inter_area
        union = _sutherland_hodgman_clip(union, list(poly)) + list(poly)
    return area


def quad_iou(q1: Quad, q2: Quad) -> float:
    """旋转框 IoU：转多边形后求交并"""
    poly1 = quad_to_polygon(q1)
    poly2 = quad_to_polygon(q2)

    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    inter_poly = _sutherland_hodgman_clip(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union
