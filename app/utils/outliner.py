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
    dot_spacing: int = 15,        # grid cell size in pixels
    blur_sigma: float = 5.0,      # gaussian blur sigma — controls edge fade width
    dot_radius_max: float = 0.62, # max dot radius as fraction of dot_spacing
    dot_radius_min: float = 0.05, # min dot radius as fraction of dot_spacing
    gamma: float = 1.0,           # density gamma correction (1.0 = linear)
    threshold: float = 0.05,      # density below this value skips the dot
) -> Image.Image:
    """
    Nike-style halftone fill:
    - Gaussian blur generates a smooth density map for natural edge fade
    - Straight grid (horizontal/vertical)
    - Dot size scaled by density
    """
    arr = np.array(pil_rgba)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    # extract binary silhouette
    _, silhouette = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # blur silhouette to get density map — edges naturally fall toward 0
    blurred = cv2.GaussianBlur(silhouette.astype(np.float32), (0, 0), sigmaX=blur_sigma)
    density = blurred / 255.0  # 0.0 ~ 1.0

    if gamma != 1.0:
        density = np.power(density, gamma)

    r, g, b = hex_to_rgb(color)
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(result)

    # straight grid centered in each cell
    for cy in range(dot_spacing // 2, h, dot_spacing):
        for cx in range(dot_spacing // 2, w, dot_spacing):
            d = density[cy, cx]

            if d < threshold:
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