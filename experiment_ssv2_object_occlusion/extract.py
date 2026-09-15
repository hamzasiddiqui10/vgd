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

Example:
    python extract.py \
        --annotations train:data/train.txt val:data/val.txt \
        --image_dir $COLD/felixbm/ssv2_center_frame_occluding_objects/frames/ \
        --out data/extracted_features/ssv2-center-dinov2-unoccluded.h5 \
        --batch_size 32 \
        --num_workers 14

"""

import argparse
import json
import os
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


@torch.inference_mode()
def extract_split(model, args, split, annotation_file, f_out, device):
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
    args = p.parse_args()

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
