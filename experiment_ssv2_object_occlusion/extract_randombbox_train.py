"""Extract frozen DINOv2 patch features into one HDF5 file.

The input file is expected to have rows in the format `<clip>.webm <label>`.

The center frame of every clip is expected as a JPG named after the clip id
(e.g. clip "78687.webm" in the annotation file -> 
"<image_dir>/projects_extern_nhr_nhr_ni_nib00021_dir.project_data_public_ssv2_clips_78687.jpg").

Layout of the output file:

    /<split>/features  float16 [N, 1, P, D]   patch tokens (P = (224/14)^2 = 256)
    /<split>/labels    int16   [N]
    /<split>/names     str     [N]            clip id as written in the annotation file
    /<split>.attrs["num_written"]             how many rows are valid (for resuming)

Rows are written in annotation-file order, so an interrupted run can be resumed
by simply re-running the same command.

Example (unoccluded):
    python extract.py \
        --annotations train:data/train.txt val:data/val.txt \
        --image_dir $COLD/felixbm/ssv2_center_frame_occluding_objects/frames/ \
        --out data/extracted_features/ssv2-center-dinov2-unoccluded.h5 \
        --batch_size 32 \
        --num_workers 14

Example (occluded – blacks out SAM3 bounding boxes before DINOv2):
    python extract.py \
        --annotations train:data/train.txt val:data/val.txt \
        --image_dir $COLD/felixbm/ssv2_center_frame_occluding_objects/frames/ \
        --out data/extracted_features/ssv2-center-dinov2-bbox-occluded.h5 \
        --coco_json data/annotations_score0.2_nms0.15.coco.json \
        --occlusion \
        --batch_size 32 \
        --num_workers 14

"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
from pathlib import Path

import h5py
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CenterFrameDataset(Dataset):
    """SSv2 center frames listed by a standard SSv2 annotation file.

    The annotation file has one whitespace-separated `<clip>.webm <label>` per line.

    This is the place to add on-the-fly changes to the images (augmentations,
    occlusions, ...): either extend `self.transform` or edit `__getitem__`.
    """

    def __init__(self, annotation_file, image_dir, image_size=224, image_ext=".jpg"):
        self.image_dir = image_dir
        self.image_ext = image_ext

        self.samples = []
        with open(annotation_file) as f:
            for line in f:
                line = line.split()
                if not line:
                    continue
                self.samples.append((Path(line[0]).stem, int(line[1])))

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir,
                                      "projects_extern_nhr_nhr_ni_nib00021_dir.project_data_public_ssv2_clips_" 
                                      + name + self.image_ext)).convert("RGB")
        return self.transform(img), label


class CenterFrameRandomBboxDataset(CenterFrameDataset):
    """Like CenterFrameDataset, but places *shuffled* bounding boxes at random positions.

    For the **train** split we do NOT occlude the object itself.  Instead:
      1. The list of bbox-sets is shuffled so every image receives the bounding
         boxes that belong to a *different* image (by index after shuffling).
      2. Each bbox is placed at a uniformly-random (x, y) offset such that the
         box stays fully within the raw PIL image.

    This lets the model learn an appearance bias from the distractor boxes
    during training without ever seeing its *own* object blacked out, so the
    bias cannot be exploited at evaluation time (where the val split uses the
    original in-place occlusion).

    Args:
        coco_json: Path to the COCO-format annotation JSON produced by SAM3.
        annotation_file: Same SSv2 split file as used by the parent class.
        image_dir: Directory containing the center-frame JPGs.
        image_size: Target square size fed to DINOv2 (default 224).
        image_ext: Extension of the frame files (default ``.jpg``).
        seed: Random seed for the bbox shuffle (default 42).
    """

    FILE_NAME_PREFIX = "projects_extern_nhr_nhr_ni_nib00021_dir.project_data_public_ssv2_clips_"

    def __init__(self, coco_json, annotation_file, image_dir, image_size=224, image_ext=".jpg", seed=42):
        super().__init__(annotation_file, image_dir, image_size, image_ext)

        print(f"Loading COCO annotations from {coco_json} ...")
        with open(coco_json) as f:
            coco = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        bboxes_by_fname: dict[str, list] = {}
        for ann in coco["annotations"]:
            fname = id_to_filename[ann["image_id"]]
            bboxes_by_fname.setdefault(fname, []).append(ann["bbox"])

        print(f"  Loaded bboxes for {len(bboxes_by_fname)} images "
              f"({sum(len(v) for v in bboxes_by_fname.values())} total detections).")

        # Build a per-sample list of bbox lists aligned with self.samples.
        # Samples without detections get an empty list.
        ordered_bboxes = []
        for name, _ in self.samples:
            fname = self.FILE_NAME_PREFIX + name + self.image_ext
            ordered_bboxes.append(bboxes_by_fname.get(fname, []))

        # Shuffle the bbox lists so each sample receives another sample's boxes.
        import random
        rng = random.Random(seed)
        shuffled_indices = list(range(len(ordered_bboxes)))
        rng.shuffle(shuffled_indices)
        self._shuffled_bboxes = [ordered_bboxes[i] for i in shuffled_indices]
        self._rng = rng

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        file_name = self.FILE_NAME_PREFIX + name + self.image_ext

        img = Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")
        img_w, img_h = img.size

        bboxes = self._shuffled_bboxes[idx]
        if bboxes:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for x, y, w, h in bboxes:
                bw, bh = int(w), int(h)
                # Clamp box dimensions to image bounds.
                bw = min(bw, img_w)
                bh = min(bh, img_h)
                # Random top-left corner such that the box stays within the image.
                rx = self._rng.randint(0, img_w - bw)
                ry = self._rng.randint(0, img_h - bh)
                draw.rectangle([rx, ry, rx + bw, ry + bh], fill=(0, 0, 0))

        return self.transform(img), label


class CenterFrameOccludedDataset(CenterFrameDataset):
    """Like CenterFrameDataset, but blacks out SAM3 detected bounding boxes.

    Before the standard transform pipeline the bounding box regions from the
    COCO annotation file (SAM3 detections) are filled with black (0, 0, 0) on
    the raw PIL image.  All annotations whose ``image_id`` maps to this clip's
    file_name are applied.  If a clip has no detections it is passed through
    unchanged.

    Args:
        coco_json: Path to the COCO-format annotation JSON produced by SAM3
            (e.g. ``data/annotations_score0.2_nms0.15.coco.json``).
        annotation_file: Same SSv2 split file as used by the parent class.
        image_dir: Directory containing the center-frame JPGs.
        image_size: Target square size fed to DINOv2 (default 224).
        image_ext: Extension of the frame files (default ``.jpg``).
    """

    FILE_NAME_PREFIX = "projects_extern_nhr_nhr_ni_nib00021_dir.project_data_public_ssv2_clips_"

    def __init__(self, coco_json, annotation_file, image_dir, image_size=224, image_ext=".jpg"):
        super().__init__(annotation_file, image_dir, image_size, image_ext)

        # Build a mapping: file_name (str) -> list of [x, y, w, h] bboxes.
        # We parse the JSON once here so __getitem__ is free of I/O overhead.
        print(f"Loading COCO annotations from {coco_json} ...")
        with open(coco_json) as f:
            coco = json.load(f)

        # id -> file_name lookup
        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

        self._bboxes: dict[str, list] = {}  # file_name -> [[x,y,w,h], ...]
        for ann in coco["annotations"]:
            fname = id_to_filename[ann["image_id"]]
            self._bboxes.setdefault(fname, []).append(ann["bbox"])

        print(f"  Loaded bboxes for {len(self._bboxes)} images "
              f"({sum(len(v) for v in self._bboxes.values())} total detections).")

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        file_name = self.FILE_NAME_PREFIX + name + self.image_ext

        img = Image.open(os.path.join(self.image_dir, file_name)).convert("RGB")

        # Black out every SAM3-detected bounding box on the raw PIL image.
        bboxes = self._bboxes.get(file_name, [])
        if bboxes:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for x, y, w, h in bboxes:
                # COCO bbox: [x_min, y_min, width, height] -> clamp to image bounds
                x0, y0 = int(x), int(y)
                x1, y1 = int(x + w), int(y + h)
                draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

        return self.transform(img), label


@torch.inference_mode()
def extract_split(model, args, split, annotation_file, f_out, device):
    if args.occlusion:
        if split == "train":
            # Train: place *shuffled* bboxes from other images at random positions
            # (object itself is never occluded, only a random distractor box is drawn).
            dataset = CenterFrameRandomBboxDataset(args.coco_json, annotation_file, args.image_dir, args.image_size)
        else:
            # Val / test: black out the image's own SAM3 bboxes (original behaviour).
            # dataset = CenterFrameOccludedDataset(args.coco_json, annotation_file, args.image_dir, args.image_size) 
            dataset = CenterFrameRandomBboxDataset(args.coco_json, annotation_file, args.image_dir, args.image_size) ### This is for final run to make random occlusion on both train and val. ## REMOVE this at end.
    else:
        dataset = CenterFrameDataset(annotation_file, args.image_dir, args.image_size)
    num_samples = len(dataset)

    # one dummy forward to learn the token grid / feature dim
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    num_patches, dim = forward_features(model, dummy).shape[1:]

    group = f_out.require_group(split)
    if "features" not in group:
        group.create_dataset("features", shape=(num_samples, 1, num_patches, dim), dtype="float16")
        group.create_dataset("labels", data=[lab for _, lab in dataset.samples], dtype="int16")
        group.create_dataset("names", data=[n for n, _ in dataset.samples], dtype=h5py.string_dtype())
        group.attrs["num_written"] = 0

    start = int(group.attrs["num_written"])
    assert group["features"].shape == (num_samples, 1, num_patches, dim), \
        f"{split}: existing file does not match this configuration, delete it first"
    if start >= num_samples:
        print(f"[{split}] already complete ({num_samples} samples)")
        return

    loader = DataLoader(
        Subset(dataset, range(start, num_samples)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    offset = start
    for images, _ in tqdm(loader, desc=f"{split} ({start}/{num_samples} done)"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats = forward_features(model, images)
        feats = feats.float().unsqueeze(1).cpu().numpy()  # [B, 1, P, D]

        group["features"][offset:offset + len(feats)] = feats
        offset += len(feats)
        group.attrs["num_written"] = offset
        f_out.flush()

    print(f"[{split}] wrote {num_samples} samples of shape (1, {num_patches}, {dim})")


def forward_features(model, images):
    """Patch tokens only: drop the CLS / register prefix tokens."""
    return model.forward_features(images)[:, model.num_prefix_tokens:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", nargs="+", required=True,
                   help="one or more 'split:/path/to/annotation.txt', e.g. train:.../train.txt val:.../val.txt")
    p.add_argument("--image_dir", required=True, help="folder with the center frame of every clip as <clip_id>.jpg")
    p.add_argument("--out", required=True, help="output .h5 file")
    p.add_argument("--model", default="vit_large_patch14_reg4_dinov2.lvd142m", help="timm model name")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--occlusion", action="store_true",                                                                                                                                                                                                   
                    help="If set, black out SAM3 bounding boxes on each frame before DINOv2 feature extraction.")                                                                                                                                         
    p.add_argument("--coco_json", default=None,                                                                                                                                                                                                          
                    help="Path to the COCO-format JSON file with SAM3 detections. Required when --occlusion is set.")     
    args = p.parse_args()

    if args.occlusion and not args.coco_json:                                                                                                                                                                                                            
        p.error("--coco_json is required when --occlusion is set.")   

    splits = dict(a.split(":", 1) for a in args.annotations)
    device = torch.device(args.device); print(f"{device=}")

    model = timm.create_model(args.model, pretrained=True, num_classes=0, dynamic_img_size=True)
    model = model.to(device).eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "a") as f_out:
        f_out.attrs["config"] = json.dumps(vars(args))
        for split, annotation_file in splits.items():
            extract_split(model, args, split, annotation_file, f_out, device)


if __name__ == "__main__":
    main()



