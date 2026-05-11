import math
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


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
    dot_spacing: int = 15,        # 기준 해상도(1000px) 기준 간격
    dot_radius_max: float = 0.60, # 최대 점 크기 비율
    dot_radius_min: float = 0.05, # 최소 점 크기 비율
    threshold: float = 0.05,      # 이 밀도 이하는 점 안 그림
    base_resolution: int = 1000,  # 해상도 보정 기준값
) -> Image.Image:
    """
    SAM2 마스크 기반 역휘도 하프톤:
    1. 마스크 내부 픽셀의 실제 밝기(역휘도)로 점 크기 결정
    2. 마스크 외부는 투명 (점 없음)
    3. 이미지 해상도에 따라 dot_spacing 자동 보정
       → 어떤 해상도에서도 동일한 밀도로 보임
    """
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    # ✅ 1. 해상도 기반 dot_spacing 자동 보정
    # 기준 해상도(1000px) 대비 현재 이미지 크기 비율로 spacing 조절
    scale = min(w, h) / base_resolution
    adjusted_spacing = max(4, int(dot_spacing * scale))

    # ✅ 2. 원본 RGB → 그레이스케일 → 역휘도
    # 어두운 픽셀 → density 높음 → 점 큼
    # 밝은 픽셀 → density 낮음 → 점 작음
    rgb = arr[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    density = 1.0 - (gray / 255.0)  # 역휘도

    r, g, b = hex_to_rgb(color)
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result)

    # ✅ 3. 수직/수평 정사각형 격자로 점 배치
    for cy in range(adjusted_spacing // 2, h, adjusted_spacing):
        for cx in range(adjusted_spacing // 2, w, adjusted_spacing):

            # 마스크 밖 → 점 안 그림
            if alpha[cy, cx] < 10:
                continue

            d = density[cy, cx]

            # threshold 이하 밀도 → 점 안 그림 (너무 밝은 부분 제거)
            if d < threshold:
                continue

            radius = int(
                (dot_radius_min + (dot_radius_max - dot_radius_min) * d)
                * adjusted_spacing
            )
            if radius < 1:
                continue

            x0, y0 = cx - radius, cy - radius
            x1, y1 = cx + radius, cy + radius
            draw.ellipse([x0, y0, x1, y1], fill=(r, g, b, 255))

    return result


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