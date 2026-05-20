import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _extract_outline(pil_rgba: Image.Image, color: str, thickness: int) -> Image.Image:
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]

    _, silhouette = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    kernel_close = np.ones((7, 7), np.uint8)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((3, 3), np.uint8)
    silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel_open)

    edges = cv2.Canny(silhouette, 50, 150)

    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    r, g, b = hex_to_rgb(color)
    result = np.zeros_like(arr)
    mask = edges > 0
    result[mask] = [r, g, b, 255]
    return Image.fromarray(result, "RGBA")


def _extract_halftone(
    pil_rgba: Image.Image,
    color: str,
    dot_spacing: int = 15,
    dot_radius_max: float = 0.45,
    dot_radius_min: float = 0.06,
    threshold: float = 0.05,
    blur_sigma: float = 1.0,
    base_resolution: int = 1000,
    supersample: int = 3,
) -> Image.Image:
    arr = np.array(pil_rgba)
    orig_w, orig_h = pil_rgba.size

    # ── 1. 피사체 bounding box로 크롭 — 프레임 내 비율과 무관하게 점 밀도 균일화
    full_alpha = arr[:, :, 3]
    ys, xs = np.where(full_alpha > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    cropped = arr[y0:y1, x0:x1]
    alpha = cropped[:, :, 3]
    h, w = alpha.shape

    # ── 2. 해상도 자동 보정 (피사체 실제 크기 기준)
    scale = min(w, h) / base_resolution
    sp = max(6, int(dot_spacing * scale))

    # ── 3. alpha_mask — 노이즈 제거 + 경계 축소
    alpha_mask = (alpha > 10).astype(np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_erode = np.ones((3, 3), np.uint8)
    alpha_mask = cv2.erode(alpha_mask, kernel_erode, iterations=2).astype(np.float32)

    if alpha_mask.max() == 0:
        return Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    # ── 4. luma 기반 density (어두울수록 큰 점)
    luma = (0.299 * cropped[:, :, 0].astype(np.float32) +
            0.587 * cropped[:, :, 1].astype(np.float32) +
            0.114 * cropped[:, :, 2].astype(np.float32))
    inverted = 1.0 - luma / 255.0

    # ── 5. blur 후 mask 적용
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=blur_sigma)
    density = blurred * alpha_mask

    # ── 6. 감마 보정으로 대비 강화
    density = np.power(density, 0.5)
    density = np.clip(density, 0.0, 1.0)

    # ── 7. 슈퍼샘플 캔버스에 점 그리기
    S = supersample
    r, g, b = hex_to_rgb(color)
    canvas = np.zeros((h * S, w * S, 4), dtype=np.uint8)

    for cy in range(sp // 2, h, sp):
        for cx in range(sp // 2, w, sp):
            d = density[cy, cx]
            if d < threshold:
                continue
            radius = int((dot_radius_min + (dot_radius_max - dot_radius_min) * d) * sp * S)
            if radius < 1:
                continue
            cv2.circle(canvas, (cx * S, cy * S), radius, (r, g, b, 255), -1, lineType=cv2.LINE_AA)

    # ── 8. 크롭 크기로 축소 후 원본 캔버스에 배치
    crop_result = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    result = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    result[y0:y1, x0:x1] = crop_result
    return Image.fromarray(result, "RGBA")


def apply_outlines(
    masked_frames: list[dict],
    person_color: str,
    background_color: str,
    thickness: int,
    style: str = "halftone",
    dot_spacing: int = 15,
    object_color: str = "#FF5A1F",
) -> list[dict]:
    print(f"[DEBUG] person_color={person_color}, object_color={object_color}")
    outlined = []
    for item in masked_frames:
        ref = item.get("person") or item.get("object") or item.get("background")
        w, h = ref.size if ref else (1, 1)

        if style == "halftone":
            person_outline = (
                _extract_halftone(item["person"], person_color, dot_spacing=dot_spacing)
                if item.get("person") else Image.new("RGBA", (w, h), (0, 0, 0, 0))
            )
            object_outline = (
                _extract_halftone(item["object"], object_color, dot_spacing=dot_spacing)
                if item.get("object") else Image.new("RGBA", (w, h), (0, 0, 0, 0))
            )

            person_outline.save(Path(tempfile.gettempdir()) / f"debug_person_outline_{len(outlined)}.png")
            object_outline.save(Path(tempfile.gettempdir()) / f"debug_object_outline_{len(outlined)}.png")
        else:
            person_outline = (
                _extract_outline(item["person"], person_color, thickness)
                if item.get("person") else Image.new("RGBA", (w, h), (0, 0, 0, 0))
            )
            object_outline = (
                _extract_outline(item["object"], object_color, thickness)
                if item.get("object") else Image.new("RGBA", (w, h), (0, 0, 0, 0))
            )

        bg_outline = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        outlined.append({**item, "person_outline": person_outline, "object_outline": object_outline, "bg_outline": bg_outline})
    return outlined