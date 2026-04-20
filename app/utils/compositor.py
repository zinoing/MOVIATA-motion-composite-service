from pathlib import Path
from PIL import Image
from app.core.config import OUTPUTS_DIR


def composite_frames(outlined_frames: list[dict], job_id: str, output_format: str) -> str:
    if not outlined_frames:
        raise ValueError("No frames to composite")

    ref = outlined_frames[0]["person_outline"]
    canvas = Image.new("RGBA", ref.size, (0, 0, 0, 255))

    for item in outlined_frames:
        canvas = Image.alpha_composite(canvas, item["bg_outline"])
        canvas = Image.alpha_composite(canvas, item["person_outline"])

    output_path = OUTPUTS_DIR / f"{job_id}.{output_format}"

    if output_format == "png":
        canvas.convert("RGB").save(str(output_path), "PNG")
    else:
        # For MP4, save as PNG first — ffmpeg encoding handled separately
        png_path = OUTPUTS_DIR / f"{job_id}_composite.png"
        canvas.convert("RGB").save(str(png_path), "PNG")
        _export_mp4(png_path, output_path)

    return str(output_path)


def _export_mp4(image_path: Path, output_path: Path) -> None:
    import ffmpeg
    (
        ffmpeg
        .input(str(image_path), loop=1, t=3)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
