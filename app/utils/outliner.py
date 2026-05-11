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
    dot_radius_max: float = 0.58,  # 최대 점 크기 (간격 대비 비율)
    dot_radius_min: float = 0.10,  # 최소 점 크기
    threshold: float = 0.08,       # 이 밀도 이하 점 생략
    blur_sigma: float = 3.0,       # 경계 페이드용 blur (마스크에만 적용)
    base_resolution: int = 1000,   # 해상도 보정 기준
) -> Image.Image:
    """
    Alpha 마스크 기반 하프톤:
    - 마스크 내부 = 점 그림, 외부 = 점 없음
    - 경계부는 Gaussian blur로 자연스럽게 페이드
    - cv2.circle 사용 (PIL ellipse 십자가 버그 방지)
    - 해상도에 따라 dot_spacing 자동 보정
    """
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3].astype(np.float32)  # 0~255
    h, w = alpha.shape

    # ── 1. 해상도 자동 보정 ──────────────────────────────────────
    # 기준 해상도(1000px) 대비 현재 이미지 크기로 spacing 스케일
    # → 작은 이미지도 큰 이미지도 동일한 밀도처럼 보임
    scale = min(w, h) / base_resolution
    sp = max(6, int(dot_spacing * scale))  # 최소 6px 보장

    # ── 2. 마스크 alpha blur → 경계 자연스러운 페이드 ──────────
    # 마스크 내부 = 1.0, 경계부 = 0→1 fade, 외부 = 0.0
    alpha_norm = alpha / 255.0
    density = cv2.GaussianBlur(alpha_norm, (0, 0), sigmaX=blur_sigma)

    # ── 5. cv2로 점 그리기 (PIL ellipse 십자가 버그 방지) ────────
    r, g, b = hex_to_rgb(color)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for cy in range(sp // 2, h, sp):
        for cx in range(sp // 2, w, sp):
            d = density[cy, cx]

            # threshold 이하 → 점 생략
            if d < threshold:
                continue

            radius = int((dot_radius_min + (dot_radius_max - dot_radius_min) * d) * sp)

            if radius < 1:
                continue

            cv2.circle(canvas, (cx, cy), radius, (r, g, b, 255), -1, lineType=cv2.LINE_AA)

    return Image.fromarray(canvas, "RGBA")


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