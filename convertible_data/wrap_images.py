import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def wrap_image(img: np.ndarray) -> np.ndarray:
    """
    Toroidally shift an image by half its width and height.
    Works for grayscale or RGB images.
    """
    h, w = img.shape[:2]
    shift_y = h // 2
    shift_x = w // 2

    wrapped = np.roll(img, shift_y, axis=0)
    wrapped = np.roll(wrapped, shift_x, axis=1)

    return wrapped

input_dir = Path("Textures/floor/raw")
output_dir = Path("convertible_data/data/floors_wrapped")
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in tqdm(list(input_dir.glob("*.png"))):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    wrapped_np = wrap_image(img_np)
    wrapped_img = Image.fromarray(wrapped_np)

    wrapped_img.save(output_dir / img_path.name)
