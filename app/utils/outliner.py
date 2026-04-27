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
    dot_spacing: int = 8,
    dot_radius_max: float = 0.48,
    dot_radius_min: float = 0.08,
) -> Image.Image:
    """
    마스크 영역을 할프톤 dot 패턴으로 채운다.
    중심부는 크고 촘촘하게, 경계부는 작고 희박하게.
    distance transform 기반으로 크기 결정.
    """
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    _, silhouette = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # distance transform: 마스크 내부 중심일수록 값이 큼
    dist = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)

    # 정규화 후 감마 적용 → 경계부 dot이 더 급격히 작아짐
    max_dist = dist.max()
    if max_dist > 0:
        dist_norm = (dist / max_dist) ** 0.5
    else:
        dist_norm = dist

    r, g, b = hex_to_rgb(color)
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result)

    cos_a = math.cos(math.radians(45))
    sin_a = math.sin(math.radians(45))
    n_range = int(math.sqrt(w * w + h * h) / dot_spacing) + 2
    cx0, cy0 = w / 2, h / 2

    for i in range(-n_range, n_range):
        for j in range(-n_range, n_range):
            cx = int(cx0 + (i * cos_a - j * sin_a) * dot_spacing)
            cy = int(cy0 + (i * sin_a + j * cos_a) * dot_spacing)

            if not (0 <= cx < w and 0 <= cy < h):
                continue

            d = dist_norm[cy, cx]
            if d < 0.02:
                continue

            radius = int((dot_radius_min + (dot_radius_max - dot_radius_min) * d) * dot_spacing)
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
    dot_spacing: int = 12,
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
