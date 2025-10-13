## Dataset layout (absolute paths)

- Train images: `/data/rdd2022/train/images`  
- Train labels (YOLO txt): `/data/rdd2022/train/labels`  
- Validation images: `/data/rdd2022/val/images`  
- Validation labels: `/data/rdd2022/val/labels`  
- Test images: `/data/rdd2022/test/images`  
- Test labels: `/data/rdd2022/test/labels`

> Note: YOLOv5 expects labels to be alongside images with the same filename (different extension).  
> E.g. `/data/rdd2022/train/images/0001.jpg` and `/data/rdd2022/train/labels/0001.txt`.

---

## Final class list

Place this in `data/classes.txt` (one class per line):
Longitudinal
Transverse
Crocodile
Other
Pothole

---

## data/data.yaml

Place this file at `data/data.yaml`:

```yaml
train: /data/rdd2022/train/images
val:   /data/rdd2022/val/images
test:  /data/rdd2022/test/images
nc: 5
names: ['Longitudinal', 'Transverse', 'Crocodile', 'Other', 'Pothole']
```

## Repository layout (recommended)

environment.yml / requirements.txt
data/
  classes.txt
  data.yaml
  cleaned/            # optional: cleaned copies of images+labels
  splits/             # optional: train.txt / val.txt lists
yolov5/               # cloned Ultralytics yolov5 repo
runs/                 # training outputs, checkpoints, predictions
scripts/              # conversion, cleaning, split, visualize, train, eval scripts
docs/
  removed_files.csv
README.md

## Quick setup (local)

Create / activate environment

```bash
python -m venv pytorch
cd pytorch\scripts
activate
cd..
python.exe -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-current.txt
```

Clone YOLOv5 (one-time)
```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

Download dataset
https://www.kaggle.com/datasets/ziedkelboussi/rdd2020-dataset


Sanity-check: run pretrained COCO YOLOv5 on one image
from yolov5 directory
```bash
python detect.py --weights yolov5s.pt --source ../data/rdd2022/train/images/China_Drone_000001.jpg --conf 0.25
```

Quick fine‑tune (short run)
sample command: train 10 epochs (adjust --img/--batch for your GPU)
```bash
python train.py --img 640 --batch 16 --epochs 10 --data ../data/rdd2022/data.yaml --weights yolov5s.pt --project runs/train --name baseline
```
wandb error - uninstall wandb
```bash
pip uninstall wandb
```
outputs saved to runs/detect/exp
Adapt --img and --batch to available GPU memory.