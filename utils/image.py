import re
import numpy as np
import os
from PIL import Image, ImageDraw

DATASET_PATH = 'dataset'
PATCH_SIZE = 50

# class that loads an image patch along with some metadata
class ImgPatch:
    def __init__(self, path):
        path_matches = re.match(
            r'.*[/\\](\d+_idx\d)_x(\d+)_y(\d+)_class(\d)\.png', path)
        assert path_matches is not None, f"Wrong file path format: {path}"
        (patient_id, x_pos, y_pos, is_IDC) = path_matches.groups()
        self.path = path
        self.patient_id = patient_id
        self.x_pos = int(x_pos)
        self.y_pos = int(y_pos)
        self.is_IDC = is_IDC == '1'
        self.img = Image.open(path)

# class that given a patient_id, assembles the full mount slide image from its patches
class Patient:
    def __init__(self, patient_id):
        patient_id = patient_id.replace("_idx5", "")
        patient_path = os.path.join(DATASET_PATH, patient_id)
        assert os.path.isdir(
            patient_path), f"Could not find '{patient_path}'. Make sure {patient_id} is a valid patient id"
        img_paths = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(
            patient_path) for filename in filenames]
        self.img_patches = [ImgPatch(img_path) for img_path in img_paths]
        self.patient_id = patient_id

    def assemble_img(self, highlight_areas=False):
        full_img_width = max(
            patch.x_pos for patch in self.img_patches) + PATCH_SIZE
        full_img_height = max(
            patch.y_pos for patch in self.img_patches) + PATCH_SIZE
        full_img = Image.new('RGB', (full_img_width, full_img_height))
        for patch in self.img_patches:
            if highlight_areas:
                draw = ImageDraw.Draw(patch.img, 'RGBA')
                red = (255, 0, 0)
                green = (0, 255, 0)
                highlight_color = red if patch.is_IDC else green
                draw.rectangle((5, 5, 45, 45), fill=highlight_color +
                               (30,), outline=highlight_color, width=2)
            full_img.paste(im=patch.img, box=(patch.x_pos, patch.y_pos))
        return full_img


def assemble_biopsy_image(biopsy_imgs, biopsy_metadata, biopsy_labels=None, border_thickness=2, border_offset=5):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    OPACITY = 50  # 0 (transparent) 255 (opac)

    biopsy_imgs = np.array(biopsy_imgs)
    biopsy_metadata = np.array(biopsy_metadata)

    x_coords = [meta[1] for meta in biopsy_metadata]
    y_coords = [meta[2] for meta in biopsy_metadata]

    full_width = np.max(x_coords) + biopsy_imgs.shape[2]
    full_height = np.max(y_coords) + biopsy_imgs.shape[1]

    full_image = Image.new(
        "RGBA", (full_width, full_height), color=(255, 255, 255))

    for i in range(len(biopsy_imgs)):
        image = biopsy_imgs[i]
        x_coord = x_coords[i]
        y_coord = y_coords[i]

        fragment = Image.fromarray(image.astype(np.uint8))

        # Marquem la classe de cada fragment (IDC-positiu o IDC-negatiu)
        if biopsy_labels is not None:
            label = biopsy_labels[i]
            highlight = GREEN if not label else RED
            draw = ImageDraw.Draw(fragment, "RGBA")
            draw.rectangle(
                [border_offset, border_offset, fragment.width -
                    border_offset - 1, fragment.height - border_offset - 1],
                outline=highlight,
                width=border_thickness,
                fill=highlight + (OPACITY,)
            )

        full_image.paste(fragment, (x_coord, y_coord))

    return full_image
