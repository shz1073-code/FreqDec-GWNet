import builtins
import math

import cv2
import numpy as np


def _is_valid_point(point_xy):
    if point_xy is None:
        return False
    if not isinstance(point_xy, (builtins.tuple, builtins.list)) or len(point_xy) != 2:
        return False
    x, y = point_xy
    return x is not None and y is not None


def sigmoid_to_uint8_mask(prob_map, threshold=0.5):
    return (prob_map >= threshold).astype(np.uint8)


def keep_largest_component(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    if binary_mask.sum() == 0:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))
    return (labels == largest_label).astype(np.uint8)


def skeletonize(binary_mask):
    mask = (binary_mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(mask, kernel)
        opened = cv2.dilate(eroded, kernel)
        residual = cv2.subtract(mask, opened)
        skeleton = cv2.bitwise_or(skeleton, residual)
        mask = eroded
        if cv2.countNonZero(mask) == 0:
            break

    return (skeleton > 0).astype(np.uint8)


def find_endpoints(centerline_mask):
    centerline_mask = (centerline_mask > 0).astype(np.uint8)
    if centerline_mask.sum() == 0:
        return []

    padded = np.pad(centerline_mask, 1, mode="constant")
    endpoints = []
    ys, xs = np.where(centerline_mask > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        patch = padded[y : y + 3, x : x + 3]
        neighbor_count = int(patch.sum()) - 1
        if neighbor_count == 1:
            endpoints.append((int(x), int(y)))

    return endpoints


def nearest_point(points, target_xy):
    if not points or not _is_valid_point(target_xy):
        return None
    tx, ty = target_xy
    best = None
    best_dist = float("inf")
    for x, y in points:
        dist = math.hypot(x - tx, y - ty)
        if dist < best_dist:
            best_dist = dist
            best = (x, y)
    return best


def _mask_points(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    return [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]


def _heat_peak_point(heatmap):
    if heatmap is None:
        return None
    if heatmap.size == 0:
        return None
    max_value = float(np.max(heatmap))
    if max_value <= 0.0:
        return None
    flat_idx = int(np.argmax(heatmap))
    h, w = heatmap.shape
    y, x = divmod(flat_idx, w)
    return (int(x), int(y))


def _farthest_point(points, anchor_xy):
    if not points:
        return None
    ax, ay = anchor_xy
    return max(points, key=lambda p: (math.hypot(p[0] - ax, p[1] - ay), -p[1], -p[0]))


def build_endpoint_candidates(centerline_mask, tip_heatmap=None, prev_tip=None):
    """
    更鲁棒的端点候选构造：
    - 优先使用严格骨架端点
    - 如果只有 0/1 个端点，则从整条中心线里回退构造两个“端点代理”
    - 如果端点过多，则压缩成最有代表性的两个候选
    """
    points = _mask_points(centerline_mask)
    if not points:
        return []

    strict_endpoints = find_endpoints(centerline_mask)
    heat_peak = _heat_peak_point(tip_heatmap)

    def choose_anchor(pool):
        if prev_tip is not None:
            candidate = nearest_point(pool, prev_tip)
            if candidate is not None:
                return candidate
        if heat_peak is not None:
            candidate = nearest_point(pool, heat_peak)
            if candidate is not None:
                return candidate
        return min(pool, key=lambda p: (p[1], p[0]))

    if len(strict_endpoints) >= 2:
        anchor = choose_anchor(strict_endpoints)
        opposite = _farthest_point(strict_endpoints, anchor)
        reduced = [anchor]
        if opposite is not None and opposite != anchor:
            reduced.append(opposite)
        return reduced

    if len(strict_endpoints) == 1:
        anchor = strict_endpoints[0]
        opposite = _farthest_point(points, anchor)
        reduced = [anchor]
        if opposite is not None and opposite != anchor:
            reduced.append(opposite)
        return reduced

    # 没有严格端点时，直接在整条中心线上构造一个“近端/远端”候选对。
    anchor = choose_anchor(points)
    opposite = _farthest_point(points, anchor)
    reduced = [anchor]
    if opposite is not None and opposite != anchor:
        reduced.append(opposite)
    return reduced


def _endpoint_heat_score(point_xy, heatmap, window_radius=5):
    if heatmap is None:
        return 0.0
    x, y = point_xy
    h, w = heatmap.shape
    x0 = max(0, x - window_radius)
    x1 = min(w, x + window_radius + 1)
    y0 = max(0, y - window_radius)
    y1 = min(h, y + window_radius + 1)
    return float(heatmap[y0:y1, x0:x1].max()) if x0 < x1 and y0 < y1 else 0.0


def _distance_score(point_a, point_b, scale=35.0):
    if not _is_valid_point(point_a) or not _is_valid_point(point_b):
        return 0.0
    distance = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
    return math.exp(-distance / max(scale, 1e-6))


def _direction_score(prev_tip, prev_velocity, candidate_point):
    if not _is_valid_point(prev_tip) or prev_velocity is None or not _is_valid_point(candidate_point):
        return 0.0
    vx, vy = prev_velocity
    velocity_norm = math.hypot(vx, vy)
    if velocity_norm < 1e-6:
        return 0.0
    cx = candidate_point[0] - prev_tip[0]
    cy = candidate_point[1] - prev_tip[1]
    candidate_norm = math.hypot(cx, cy)
    if candidate_norm < 1e-6:
        return 1.0
    cosine = (vx * cx + vy * cy) / (velocity_norm * candidate_norm)
    return max(0.0, (cosine + 1.0) * 0.5)


def select_tip(endpoints, tip_heatmap=None, prev_tip=None, distance_weight=0.35):
    if not endpoints:
        return None, []

    scored = []
    for endpoint in endpoints:
        score = _endpoint_heat_score(endpoint, tip_heatmap)
        if prev_tip is not None:
            distance = math.hypot(endpoint[0] - prev_tip[0], endpoint[1] - prev_tip[1])
            score += distance_weight * (1.0 / (1.0 + distance))
        scored.append((score, endpoint))

    scored.sort(key=lambda item: (-item[0], item[1][1], item[1][0]))
    return scored[0][1], scored


def stabilize_tip_with_history(
    localization,
    tip_heatmap=None,
    prev_tip=None,
    prev_velocity=None,
    heat_weight=1.0,
    distance_weight=0.35,
    direction_weight=0.20,
    distance_scale=35.0,
    max_jump_distance=90.0,
    min_heat_confidence=0.15,
    ema_alpha=0.60,
):
    """
    在已有 localization 结果上再做一层轻量时序稳定：
    - 优先使用当前帧端点候选
    - 用上一帧 tip 和运动方向做重打分
    - 若当前候选跳变过大且热图支持弱，则回退到上一帧附近的中心线点
    """
    endpoints = list(localization.get("endpoints", []))
    centerline_points = list(localization.get("ordered_centerline", []))
    if not centerline_points:
        centerline_points = _mask_points(localization["centerline_mask"])

    candidate_points = list(endpoints)
    if prev_tip is not None:
        anchor = nearest_point(centerline_points, prev_tip)
        if anchor is not None and anchor not in candidate_points:
            candidate_points.append(anchor)

    if not candidate_points:
        return {
            "tip": None,
            "tracking_tip": prev_tip,
            "velocity": prev_velocity if prev_velocity is not None else (0.0, 0.0),
            "score": 0.0,
            "candidates": [],
        }

    scored_candidates = []
    for candidate in candidate_points:
        heat_score = _endpoint_heat_score(candidate, tip_heatmap)
        temporal_score = _distance_score(candidate, prev_tip, scale=distance_scale)
        motion_score = _direction_score(prev_tip, prev_velocity, candidate)
        total_score = (
            heat_weight * heat_score
            + distance_weight * temporal_score
            + direction_weight * motion_score
        )
        scored_candidates.append(
            {
                "point": candidate,
                "heat_score": heat_score,
                "temporal_score": temporal_score,
                "motion_score": motion_score,
                "total_score": total_score,
            }
        )

    scored_candidates.sort(
        key=lambda item: (-item["total_score"], item["point"][1], item["point"][0])
    )
    best = scored_candidates[0]
    chosen_tip = best["point"]

    if prev_tip is not None:
        jump_distance = math.hypot(chosen_tip[0] - prev_tip[0], chosen_tip[1] - prev_tip[1])
        if jump_distance > max_jump_distance and best["heat_score"] < min_heat_confidence:
            fallback_tip = nearest_point(centerline_points, prev_tip)
            if fallback_tip is not None:
                chosen_tip = fallback_tip

    if prev_tip is None or chosen_tip is None:
        tracking_tip = chosen_tip
        velocity = (0.0, 0.0)
    else:
        tracking_tip = (
            int(round((1.0 - ema_alpha) * prev_tip[0] + ema_alpha * chosen_tip[0])),
            int(round((1.0 - ema_alpha) * prev_tip[1] + ema_alpha * chosen_tip[1])),
        )
        velocity = (
            float(tracking_tip[0] - prev_tip[0]),
            float(tracking_tip[1] - prev_tip[1]),
        )

    return {
        "tip": chosen_tip,
        "tracking_tip": tracking_tip,
        "velocity": velocity,
        "score": best["total_score"],
        "candidates": scored_candidates,
    }


def _neighbors_8(x, y, width, height):
    if x is None or y is None:
        return
    for ny in range(builtins.max(0, y - 1), builtins.min(height, y + 2)):
        for nx in range(builtins.max(0, x - 1), builtins.min(width, x + 2)):
            if nx == x and ny == y:
                continue
            yield nx, ny


def trace_centerline(centerline_mask, start_point=None):
    centerline_mask = (centerline_mask > 0).astype(np.uint8)
    ys, xs = np.where(centerline_mask > 0)
    if len(xs) == 0:
        return []

    points = {(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())}
    endpoints = find_endpoints(centerline_mask)
    if not _is_valid_point(start_point):
        start_point = endpoints[0] if endpoints else min(points, key=lambda p: (p[1], p[0]))
    else:
        start_point = nearest_point(points, start_point) or (
            endpoints[0] if endpoints else min(points, key=lambda p: (p[1], p[0]))
        )

    width = centerline_mask.shape[1]
    height = centerline_mask.shape[0]
    ordered = []
    visited = set()
    stack = [start_point]

    while stack:
        point = stack.pop()
        if not _is_valid_point(point):
            continue
        if point in visited or point not in points:
            continue
        visited.add(point)
        ordered.append(point)

        x, y = point
        neighbors = []
        for nx, ny in _neighbors_8(x, y, width, height):
            if (nx, ny) in points and (nx, ny) not in visited:
                neighbors.append((nx, ny))
        neighbors.sort(key=lambda p: (p[1], p[0]), reverse=True)
        stack.extend(neighbors)

    return ordered


def localize_from_maps(
    seg_prob,
    centerline_prob=None,
    tip_prob=None,
    seg_threshold=0.5,
    centerline_threshold=0.2,
    prev_tip=None,
):
    seg_binary = keep_largest_component(sigmoid_to_uint8_mask(seg_prob, threshold=seg_threshold))

    if centerline_prob is not None:
        centerline_seed = sigmoid_to_uint8_mask(centerline_prob, threshold=centerline_threshold)
        centerline_seed = keep_largest_component(centerline_seed * seg_binary)
        centerline_mask = skeletonize(centerline_seed if centerline_seed.sum() > 0 else seg_binary)
    else:
        centerline_mask = skeletonize(seg_binary)

    endpoints = build_endpoint_candidates(centerline_mask, tip_heatmap=tip_prob, prev_tip=prev_tip)
    selected_tip, endpoint_scores = select_tip(endpoints, tip_heatmap=tip_prob, prev_tip=prev_tip)
    if selected_tip is None and prev_tip is not None:
        selected_tip = nearest_point(_mask_points(centerline_mask), prev_tip)
    ordered_centerline = trace_centerline(centerline_mask, start_point=selected_tip)

    return {
        "seg_mask": seg_binary,
        "centerline_mask": centerline_mask,
        "ordered_centerline": ordered_centerline,
        "endpoints": endpoints,
        "selected_tip": selected_tip,
        "endpoint_scores": endpoint_scores,
    }


def draw_localization_overlay(
    image_u8,
    localization,
    gt_mask=None,
    show_tip=True,
    show_endpoints=True,
    show_gt_mask=True,
    show_seg_mask=True,
):
    canvas = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)

    seg_mask = localization["seg_mask"]
    centerline = localization["ordered_centerline"]
    endpoints = localization["endpoints"]
    selected_tip = localization["selected_tip"]

    if gt_mask is not None and show_gt_mask:
        gt_mask = (gt_mask > 0).astype(np.uint8)
        canvas[gt_mask > 0, 1] = np.maximum(canvas[gt_mask > 0, 1], 100)

    if show_seg_mask:
        canvas[seg_mask > 0, 0] = np.maximum(canvas[seg_mask > 0, 0], 80)

    for x, y in centerline:
        cv2.circle(canvas, (int(x), int(y)), 0, (0, 0, 255), 1)

    if show_endpoints:
        for x, y in endpoints:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), 1)

    if selected_tip is not None and show_tip:
        cv2.circle(canvas, (int(selected_tip[0]), int(selected_tip[1])), 5, (0, 255, 255), -1)

    return canvas
