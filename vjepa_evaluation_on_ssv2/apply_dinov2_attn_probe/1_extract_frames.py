"""

python 1_extract_frames.py \
  --labels_file ./data/myvideos_2/test.txt \
  --video_dir /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/raw/ \
  --output_dir ./data/myvideos_2/

"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm



def extract_center_frames(labels_file, video_dir, output_dir):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    # Output directory for images and new labels text file
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "frames_labels.txt"

    # Read video names and labels
    with open(labels_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    records = []

    for vid_name, class_id in tqdm(lines, desc="Extracting center frames"):
        vid_path = video_dir / vid_name

        if not vid_path.exists():
            print(f"\n[Warning] Video not found: {vid_path}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"\n[Warning] Could not open video: {vid_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"\n[Warning] Zero total frames reported for: {vid_name}")
            cap.release()
            continue

        # Target center frame
        center_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(
                f"\n[Warning] Failed to read frame {center_idx} for {vid_name}"
            )
            continue

        # Save frame as JPG
        img_name = f"{Path(vid_name).stem}.jpg"
        img_path = frames_dir / img_name
        cv2.imwrite(str(img_path), frame)

        # Store relative frame path and class ID for next stage
        rel_img_path = Path("frames") / img_name
        records.append(f"{rel_img_path} {class_id}\n")

    # Write output labels text file
    with open(out_manifest, "w") as f:
        f.writelines(records)

    print(f"\nFinished! Extracted {len(records)}/{len(lines)} center frames.")
    print(f"Frames directory: {frames_dir}")
    print(f"New label file:   {out_manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract center frames from short videos."
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="Path to input labels.txt (e.g., vidname.avi 49)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to directory containing raw video files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames/ and frames_labels.txt",
    )

    args = parser.parse_args()
    extract_center_frames(args.labels_file, args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()