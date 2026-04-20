import cv2
import numpy as np
from PIL import Image


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _extract_outline(pil_rgba: Image.Image, color: str, thickness: int) -> Image.Image:
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]
    edges = cv2.Canny(alpha, 50, 150)
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    r, g, b = _hex_to_rgb(color)
    result = np.zeros_like(arr)
    mask = edges > 0
    result[mask] = [r, g, b, 255]
    return Image.fromarray(result, "RGBA")


def apply_outlines(
    masked_frames: list[dict],
    person_color: str,
    background_color: str,
    thickness: int,
) -> list[dict]:
    outlined = []
    for item in masked_frames:
        person_outline = _extract_outline(item["person"], person_color, thickness)
        bg_outline = _extract_outline(item["background"], background_color, thickness)
        outlined.append({**item, "person_outline": person_outline, "bg_outline": bg_outline})
    return outlined
