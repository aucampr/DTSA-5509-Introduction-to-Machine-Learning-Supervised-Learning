import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

"""
Non-recursive (default), write YOLO-format labels with confidence:
python D:/repos/intro_to_ml/src/run_on_folder.py --src_dir "D:/repos/intro_to_ml/data/rdd2022/sample" --out_dir "D:/repos/intro_to_ml/runs/detect/folder_test_2025_10_09" --save_conf
Recursive (process subfolders) and save xyxy labels without confidences:
python src/run_on_folder.py --src_dir "D:/repos/intro_to_ml/data/rdd2022" --out_dir "D:/repos/intro_to_ml/runs/detect/all_val" --recursive --label_format xyxy
"""

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def is_image(p: Path):
    return p.suffix.lower() in IMAGE_EXTS

def find_images(root: Path, recursive: bool):
    if recursive:
        return [p for p in root.rglob('*') if is_image(p)]
    else:
        return [p for p in root.iterdir() if p.is_file() and is_image(p)]

def run_on_folder(yolov5_repo, weights, src_dir, out_dir,
                  recursive=True, save_conf=True, label_format="yolo",
                  img_size=640, conf_thres=0.25, iou_thres=0.45):
    src_dir = Path(src_dir).resolve()
    out_dir = Path(out_dir).resolve()
    labels_root = out_dir / "labels"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    print("Loading model...")
    model = torch.hub.load(str(yolov5_repo), 'custom', path=str(weights), source='local')
    model.conf = conf_thres
    model.iou = iou_thres

    imgs = find_images(src_dir, recursive)
    print(f"Found {len(imgs)} images under {src_dir} (recursive={recursive})")
    if len(imgs) == 0:
        return

    for img_path in tqdm(imgs, desc="Processing images"):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"Warning: failed to read image {img_path}, skipping.")
                continue
            h, w = img_bgr.shape[:2]
            img_rgb = img_bgr[:, :, ::-1]
            results = model(img_rgb, size=img_size)

            df = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax, confidence, class, name

            # prepare output paths preserving relative structure
            rel = img_path.relative_to(src_dir)
            out_image_path = out_dir / rel
            out_image_path.parent.mkdir(parents=True, exist_ok=True)

            # draw boxes on copy of BGR image
            img_draw = img_bgr.copy()
            for _, row in df.iterrows():
                x1 = int(max(0, row['xmin'])); y1 = int(max(0, row['ymin']))
                x2 = int(min(w-1, row['xmax'])); y2 = int(min(h-1, row['ymax']))
                conf = float(row['confidence']); cls_name = str(row['name'])
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_draw, label, (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # save annotated image
            ok = cv2.imwrite(str(out_image_path), img_draw)
            if not ok:
                print(f"Warning: failed to write image {out_image_path}")

            # prepare label output path
            out_txt = labels_root / rel.with_suffix('.txt')
            out_txt.parent.mkdir(parents=True, exist_ok=True)

            # write labels in chosen format
            lines = []
            if len(df) == 0:
                out_txt.write_text("")  # empty file expected by YOLO
            else:
                for _, row in df.iterrows():
                    cls_id = int(row['class'])
                    if label_format == "yolo":
                        x_center = ((row['xmin'] + row['xmax']) / 2.0) / w
                        y_center = ((row['ymin'] + row['ymax']) / 2.0) / h
                        bw = (row['xmax'] - row['xmin']) / w
                        bh = (row['ymax'] - row['ymin']) / h
                        if save_conf:
                            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f} {row['confidence']:.6f}")
                        else:
                            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
                    else:  # xyxy absolute
                        if save_conf:
                            lines.append(f"{cls_id} {int(row['xmin'])} {int(row['ymin'])} {int(row['xmax'])} {int(row['ymax'])} {row['confidence']:.6f}")
                        else:
                            lines.append(f"{cls_id} {int(row['xmin'])} {int(row['ymin'])} {int(row['xmax'])} {int(row['ymax'])}")
                out_txt.write_text("\n".join(lines))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Done. Results saved to:", out_dir)
    print("Labels saved to:", labels_root)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run YOLOv5 on a folder of images and save annotated images + label .txt files")
    p.add_argument("--yolov5_repo", default=r"D:/repos/intro_to_ml/yolov5")
    p.add_argument("--weights", default=r"D:/repos/intro_to_ml/runs/train/exp_yolov5l_bs32_ep150/weights/best.pt")
    p.add_argument("--src_dir", default=r"D:/repos/intro_to_ml/data/rdd2022/sample")
    p.add_argument("--out_dir", default=r"D:/repos/intro_to_ml/runs/detect/folder_test_2025_10_09")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    p.add_argument("--save_conf", action="store_true", help="Include confidence in output .txt")
    p.add_argument("--label_format", choices=["yolo","xyxy"], default="yolo")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres", type=float, default=0.45)
    args = p.parse_args()

    run_on_folder(args.yolov5_repo, args.weights, args.src_dir, args.out_dir,
                  recursive=args.recursive, save_conf=args.save_conf,
                  label_format=args.label_format, img_size=args.img_size,
                  conf_thres=args.conf_thres, iou_thres=args.iou_thres)