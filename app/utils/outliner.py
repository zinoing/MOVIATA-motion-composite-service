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
    dot_spacing: int = 15,    # 18 → 15 (더 촘촘하게)
    dot_radius_max: float = 0.45,  # 0.55 → 0.45 (최대 점 작게)
    dot_radius_min: float = 0.06,  # 0.08 → 0.06 (최소 점 작게)
    threshold: float = 0.05,
    blur_sigma: float = 1.0,
    base_resolution: int = 1000,
    supersample: int = 3,
) -> Image.Image:
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    orig_w, orig_h = pil_rgba.size

    # ── 1. 해상도 자동 보정
    scale = min(w, h) / base_resolution
    sp = max(6, int(dot_spacing * scale))

    # ── 2. alpha_mask — 노이즈 제거 + 경계 축소
    alpha_mask = (alpha > 10).astype(np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_erode = np.ones((3, 3), np.uint8)
    alpha_mask = cv2.erode(alpha_mask, kernel_erode, iterations=2).astype(np.float32)

    if alpha_mask.max() == 0:
        return Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))

    # ── 3. luma 기반 density (어두울수록 큰 점)
    luma = (0.299 * arr[:, :, 0].astype(np.float32) +
            0.587 * arr[:, :, 1].astype(np.float32) +
            0.114 * arr[:, :, 2].astype(np.float32))
    inverted = 1.0 - luma / 255.0

    # ── 4. blur 후 mask 적용 (blur 먼저, mask 나중)
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=blur_sigma)
    density = blurred * alpha_mask

    # ── 5. 감마 보정으로 대비 강화
    density = np.power(density, 0.5)
    density = np.clip(density, 0.0, 1.0)

    # ── 6. 슈퍼샘플 캔버스에 점 그리기
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

    # ── 7. 원본 크기로 축소
    result = cv2.resize(canvas, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
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