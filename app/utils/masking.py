from pathlib import Path
from PIL import Image
from rembg import remove


def apply_masks(frames: list[Path], job_id: str) -> list[dict]:
    results = []
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            input_data = f.read()

        # rembg removes background, returning RGBA where alpha=0 is background
        person_rgba = remove(input_data)
        person_img = Image.open(__import__("io").BytesIO(person_rgba)).convert("RGBA")

        # Background is the inverse: original pixels where person alpha is 0
        original = Image.open(frame_path).convert("RGBA")
        bg_img = original.copy()
        r, g, b, a = person_img.split()
        # Invert alpha: background where person was transparent
        from PIL import ImageChops
        inv_alpha = ImageChops.invert(a.convert("L"))
        bg_img.putalpha(inv_alpha)

        results.append({"frame_path": frame_path, "person": person_img, "background": bg_img})

    return results
