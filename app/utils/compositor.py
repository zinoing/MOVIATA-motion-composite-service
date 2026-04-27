import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from app.core.config import OUTPUTS_DIR
from app.utils.outliner import hex_to_rgb


def save_layers(outlined_frames: list[dict], job_id: str) -> list[dict]:
    out_dir = Path(OUTPUTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for i, item in enumerate(outlined_frames):
        person: Image.Image = item["person_outline"]
        obj: Image.Image | None = item.get("object_outline")

        w, h = person.size
        # ✅ 투명 캔버스 → person 먼저 → object 위에
        combined = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        combined = Image.alpha_composite(combined, person)
        if obj is not None:
            combined = Image.alpha_composite(combined, obj)

        bbox = combined.getbbox()
        if bbox is None:
            bbox = (0, 0, combined.width, combined.height)
        cropped = combined.crop(bbox)
        cropped.save(str(out_dir / f"{job_id}_layer_{i}.png"))
        layers.append({
            "index": i,
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2] - bbox[0],
            "h": bbox[3] - bbox[1],
            "frame_w": combined.width,
            "frame_h": combined.height,
        })
    return layers


def composite_frames(
    outlined_frames: list[dict],
    job_id: str,
    output_format: str = "gif",
    mode: str = "normal",
    canvas_color: str = "#ffffff",
) -> str:
    composited: list[Image.Image] = []
    for item in outlined_frames:
        person_outline: Image.Image = item["person_outline"]
        obj_outline: Image.Image | None = item.get("object_outline")
        w, h = person_outline.size

        # ✅ 투명 캔버스 → person(흰색) → object(주황색) 순서로 합성
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas = Image.alpha_composite(canvas, person_outline)
        if obj_outline is not None:
            canvas = Image.alpha_composite(canvas, obj_outline)
        composited.append(canvas)

    out_path = Path(OUTPUTS_DIR) / f"{job_id}.{output_format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in ("gif", "webp"):
        composited[0].save(
            out_path, save_all=True, append_images=composited[1:],
            loop=0, duration=100,
        )
    elif output_format == "mp4":
        frame_arr = cv2.cvtColor(np.array(composited[0].convert("RGB")), cv2.COLOR_RGB2BGR)
        h_px, w_px = frame_arr.shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w_px, h_px))
        for img in composited:
            writer.write(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        if len(composited) == 1:
            composited[0].save(str(out_path))
        else:
            frame_w, frame_h = composited[0].size
            padding = 40
            total_w = frame_w * len(composited) + padding * (len(composited) - 1)
            strip = Image.new("RGBA", (total_w, frame_h), (0, 0, 0, 0))
            for i, img in enumerate(composited):
                x = i * (frame_w + padding)
                strip.paste(img, (x, 0), mask=img)
            strip.save(str(out_path))

    return str(out_path)