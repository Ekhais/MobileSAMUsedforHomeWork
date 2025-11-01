# (完整脚本开始)
"""
train_mobilesam_iam_win.py

功能更新：
- 若命令行指定的预训练权重不存在，脚本会尝试自动下载（优先使用 --pretrained-url；若未给出，则尝试使用 Hugging Face hf_hub_download，最后回退并报错提示用户）。
- 自动对 data_root/images 中的图片生成 masks 到 data_root/masks，但只为缺失的图片生成（已生成的不会重复）。
- 支持小规模试验的参数（--sample-limit, --mask-max-side, --mask-max-instances），便于你先用少量样本调试。
- 完成微调后在验证集上做评估并输出 mIoU、Dice、PixelAccuracy，同时保存每张验证图的预测 mask 到 results/ 并导出 per-image 指标 CSV。

使用示例（PowerShell）：
python train_mobilesam_iam_win.py \
  --repo "D:\project\sam\sam\MobileSAM" \
  --iam-root "D:\project\sam\sam\IAM_Handwriting" \
  --data-root "D:\project\sam\sam\data\iam" \
  --pretrained-path "" \
  --pretrained-url "" \
  --hf-repo "ChaoningZhang/MobileSAM" --hf-filename "mobile_sam_vit_t.pth" \
  --model-hint "vit_t" --epochs 12 --batch-size 1 --accum-steps 4 \
  --mask-sample-limit 500 --mask-max-side 512 --mask-max-instances 5 --sample-limit 100

注意：自动下载需要网络；hf_hub_download 需要 huggingface_hub 包可用（脚本会智能 fallback 到 requests 下载）。
"""
import json
import os
import sys
import argparse
import tarfile
import shutil
import csv
import inspect
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from PIL.Image import Resampling
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ---------------------------- Helpers ----------------------------
def add_repo_to_path(repo_path: str):
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"MobileSAM repo path not found: {repo_path}")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

def safe_makedirs(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def download_file(url: str, out_path: str):
    """Download with streaming write. Returns True on success."""
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"download_file failed: {e}")
        return False

# ---------------------------- Pretrained downloader ----------------
def ensure_pretrained(pretrained_path: str, pretrained_url: str = None, hf_repo: str = None, hf_filename: str = None):
    """
    Ensure pretrained_path exists. If not, try to download.

    Strategy:
      1) if pretrained_url provided -> download
      2) elif huggingface_hub available and hf_repo & hf_filename provided -> try hf_hub_download
      3) else -> raise RuntimeError and prompt user

    Returns: local path or raises RuntimeError
    """
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Pretrained file exists: {pretrained_path}")
        return pretrained_path

    # ensure parent dir
    if pretrained_path:
        safe_makedirs(os.path.dirname(pretrained_path))

    # 1) direct URL
    if pretrained_url:
        print(f"Attempting to download pretrained from URL: {pretrained_url}")
        ok = download_file(pretrained_url, pretrained_path)
        if ok:
            print("Downloaded pretrained to", pretrained_path)
            return pretrained_path
        else:
            print("Failed to download from --pretrained-url")

    # 2) try huggingface_hub
    if hf_repo and hf_filename:
        try:
            print(f"Trying to download from Hugging Face repo {hf_repo} file {hf_filename} ...")
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=hf_repo, filename=hf_filename, cache_dir=os.path.dirname(pretrained_path) or None)
            if pretrained_path and os.path.abspath(local) != os.path.abspath(pretrained_path):
                shutil.copy2(local, pretrained_path)
                local = pretrained_path
            print("Downloaded via hf_hub to", local)
            return local
        except Exception as e:
            print("hf_hub_download failed:", e)

    raise RuntimeError('Pretrained weights not found and automatic download failed. Provide --pretrained-url or correct --hf-repo/--hf-filename.')


def extract_totaltext_if_needed(tar_path: str, out_root: str):
    """
    从 total-text tar 包中提取 train/test 下的 img/ 和 ann/ 到 out_root 下的 train/img, train/ann, test/img, test/ann。
    如果目标目录已有文件则跳过已存在文件（不覆盖）。
    """
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Total-Text tar not found: {tar_path}")
    safe_makedirs(out_root)
    # We will extract only files whose path contains 'train/img', 'train/ann', 'test/img', 'test/ann'
    with tarfile.open(tar_path, 'r:*') as tar:
        members = tar.getmembers()
        # Build list of members to extract
        to_extract = []
        for m in members:
            # Some tar entries may have leading folder names, so we search substrings
            low = m.name.replace('\\','/').lower()
            if ('/train/img/' in low) or ('/train/images/' in low) or ('/test/img/' in low) or ('/test/images/' in low) \
               or ('/train/ann/' in low) or ('/train/annotations/' in low) or ('/test/ann/' in low) or ('/test/annotations/' in low):
                to_extract.append(m)
        if len(to_extract) == 0:
            # fallback: extract everything (conservative)
            print("Warning: didn't find expected train/img or train/ann paths in tar; extracting all files (may be large).")
            to_extract = members

        for m in tqdm(to_extract, desc="Extracting Total-Text"):
            try:
                # compute output path under out_root preserving subpath after last occurrence of 'train' or 'test'
                parts = m.name.replace('\\','/').split('/')
                # find index of 'train' or 'test' if present
                idx = None
                for i,p in enumerate(parts):
                    if p.lower() in ('train','test'):
                        idx = i
                        break
                if idx is not None:
                    subpath = os.path.join(*parts[idx:])  # e.g. train/img/img123.jpg or train/ann/img123.jpg.json
                else:
                    subpath = m.name  # fallback
                out_path = os.path.join(out_root, subpath)
                out_dir = os.path.dirname(out_path)
                safe_makedirs(out_dir)
                # only extract regular files
                if m.isdir():
                    continue
                if os.path.exists(out_path):
                    # don't overwrite
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                with open(out_path, 'wb') as of:
                    shutil.copyfileobj(f, of)
            except Exception as e:
                print(f"Failed to extract member {m.name}: {e}")

def find_image_for_ann(ann_fname: str, images_root: str):
    """
    ann_fname: full path to ann json, original name contains image full filename e.g. img1538.jpg.json
    images_root: folder where images are (may contain train/img and test/img subfolders)
    Returns image_path or None.
    Strategy: try exact basename without .json, then search images_root recursively for matching basename.
    """
    base = os.path.basename(ann_fname)
    # ann files could be named like 'img1538.jpg.json' or 'img1538.json'
    if base.lower().endswith('.json'):
        img_name = base[:-5]  # strip .json
    else:
        img_name = base
    # first try direct sibling locations (replace .json -> '')
    cand = os.path.join(os.path.dirname(ann_fname), img_name)
    if os.path.exists(cand):
        return cand
    # search images_root for file with same basename (ignoring extension)
    target_base = os.path.splitext(img_name)[0]
    for ext in ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG'):
        p = os.path.join(images_root, target_base + ext)
        if os.path.exists(p):
            return p
    # recursive search (slower)
    for p in glob(os.path.join(images_root, '**', '*.*'), recursive=True):
        if os.path.splitext(os.path.basename(p))[0].lower() == target_base.lower():
            return p
    return None

def annjson_to_mask_and_save(ann_json_path: str, image_path: str, out_mask_path: str, visualize=False):
    """
    读取 ann json（datasetninja 格式），把 objects[].points.exterior polygon 渲染为白色(255)填充 mask，
    如果 objects[].points.interior 存在则作为 hole（填 0）。
    image_path 用于取尺寸（优先），若不存在则使用 ann 中 size 字段。
    """
    # load ann
    with open(ann_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # determine canvas size (w,h)
    w = None; h = None
    # try from image
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as im:
                w, h = im.size
        except Exception:
            w = None; h = None
    # else try ann size
    if (w is None or h is None) and 'size' in data:
        try:
            h = int(data['size'].get('height', 0))
            w = int(data['size'].get('width', 0))
            if w == 0 or h == 0:
                w = None; h = None
        except Exception:
            w = None; h = None
    if w is None or h is None:
        raise RuntimeError(f"Cannot determine canvas size for ann {ann_json_path} (no image and ann.size missing)")

    mask_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    objects = data.get('objects', []) or []
    for obj in objects:
        pts = None
        pts_ext = obj.get('points', {}).get('exterior', None)
        pts_int = obj.get('points', {}).get('interior', None)
        # if exterior present and valid
        if pts_ext and len(pts_ext) >= 3:
            try:
                poly = [(int(round(p[0])), int(round(p[1]))) for p in pts_ext]
                # draw exterior as white
                draw.polygon(poly, outline=255, fill=255)
            except Exception:
                # skip invalid polygon
                continue
        # interior holes: draw fill=0
        if pts_int:
            # pts_int might be list of lists; ensure iteration
            if isinstance(pts_int[0], (list, tuple)):
                # if interior is list of points for single hole
                try:
                    polyh = [(int(round(p[0])), int(round(p[1]))) for p in pts_int]
                    draw.polygon(polyh, outline=0, fill=0)
                except Exception:
                    pass

    # save mask (white text on black background)
    mask_img.save(out_mask_path, optimize=True)

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.imshow(mask_img, cmap='gray')
        plt.title(os.path.basename(out_mask_path))
        plt.axis('off')
        plt.show()

def convert_totaltext_ann_folder_to_masks(ann_folder: str, images_root: str, masks_dir: str, sample_limit: int=None, visualize=False):
    """
    ann_folder: folder containing many ann json files (e.g. out_root/train/ann or out_root/test/ann)
    images_root: root folder where images live (e.g. out_root/train/img or a central images dir)
    masks_dir: output folder to save masks (one mask file per image, named <basename>.png)
    sample_limit: optional limit for debug
    Returns (num_processed, num_skipped, skipped_list)
    """
    safe_makedirs(masks_dir)
    ann_paths = sorted([p for p in glob(os.path.join(ann_folder, '*.json'))])
    if sample_limit is not None:
        ann_paths = ann_paths[:sample_limit]

    processed = 0
    skipped = 0
    skipped_list = []
    for ann in tqdm(ann_paths, desc=f"Converting anns in {os.path.basename(ann_folder)}"):
        try:
            # derive expected image name from ann filename (ann like img1538.jpg.json or img1538.json)
            base = os.path.basename(ann)
            if base.lower().endswith('.json'):
                img_name = base[:-5]  # removes .json
            else:
                img_name = base
            # image basename without extension
            base_no_ext = os.path.splitext(img_name)[0]
            out_mask_path = os.path.join(masks_dir, base_no_ext + '.png')
            if os.path.exists(out_mask_path):
                # already have mask
                skipped += 1
                continue
            # try find image file
            image_path = find_image_for_ann(ann, images_root)
            if image_path is None:
                # couldn't find corresponding image; skip but record
                skipped += 1
                skipped_list.append((ann, 'image_not_found'))
                continue
            # render and save
            annjson_to_mask_and_save(ann, image_path, out_mask_path, visualize=visualize)
            processed += 1
        except Exception as e:
            skipped += 1
            skipped_list.append((ann, str(e)))
    return processed, skipped, skipped_list

def prepare_totaltext_masks_from_tar(tar_path: str, out_root: str, split: str='train', sample_limit: int=None, visualize=False):
    """
    High-level helper:
      - extract tar (only relevant files) into out_root/{train,test}/...
      - convert ann jsons in out_root/{split}/ann -> masks saved to out_root/masks
    NOTE: out_root will contain extracted train/img and train/ann (if not already present).
    """
    extract_totaltext_if_needed(tar_path, out_root)

    # find extracted ann folder for given split
    possible_ann_dirs = [
        os.path.join(out_root, split, 'ann'),
        os.path.join(out_root, split, 'annotations'),
        os.path.join(out_root, split, 'gt'),
        os.path.join(out_root, f"{split}/ann"),
        os.path.join(out_root, f"{split}/annotations"),
    ]
    ann_dir = None
    for d in possible_ann_dirs:
        if os.path.isdir(d):
            ann_dir = d
            break
    if ann_dir is None:
        # try to find any folder under out_root that contains many .json and includes 'ann' in name
        for d in glob(os.path.join(out_root, '**'), recursive=True):
            if os.path.isdir(d):
                js = glob(os.path.join(d, '*.json'))
                if len(js) > 10 and ('ann' in d.lower() or 'gt' in d.lower() or 'json' in d.lower()):
                    ann_dir = d
                    break
    if ann_dir is None:
        raise RuntimeError("Could not locate ann folder after extraction. Check tar contents.")

    # images dir for that split (may be under out_root/{split}/img or images)
    possible_img_dirs = [
        os.path.join(out_root, split, 'img'),
        os.path.join(out_root, split, 'images'),
        os.path.join(out_root, 'images'),
        os.path.join(out_root, split),
    ]
    img_dir = None
    for d in possible_img_dirs:
        if os.path.isdir(d):
            # ensure contains images
            imgs = [p for p in glob(os.path.join(d, '*')) if p.lower().endswith(('.jpg','.png','.jpeg'))]
            if len(imgs) > 0:
                img_dir = d
                break
    if img_dir is None:
        # fallback: find folder under out_root containing many images
        for d in glob(os.path.join(out_root, '**'), recursive=True):
            if os.path.isdir(d):
                imgs = [p for p in glob(os.path.join(d, '*')) if p.lower().endswith(('.jpg','.png','.jpeg'))]
                if len(imgs) > 50:
                    img_dir = d
                    break
    if img_dir is None:
        raise RuntimeError("Could not locate images folder after extraction. Check tar contents.")

    # final masks output dir (placed under out_root/masks to be consistent)
    masks_out = os.path.join(out_root, split,'masks')
    safe_makedirs(masks_out)

    processed, skipped, skipped_list = convert_totaltext_ann_folder_to_masks(ann_dir, img_dir, masks_out, sample_limit=sample_limit, visualize=visualize)
    print(f"TotalText -> masks: processed={processed}, skipped={skipped}")
    if len(skipped_list) > 0:
        print("Some anns skipped (sample):", skipped_list[:10])
    return processed, skipped, skipped_list

# ---------------------------- Data prep --------------------------
def extract_iam_if_needed(iam_root: str, data_images_dir: str):
    raw_dir = os.path.join(iam_root, 'raw')
    tar_path = os.path.join(raw_dir, 'IAM_Handwriting.tar.gz')
    if os.path.exists(tar_path):
        print(f"Found IAM tar archive: {tar_path} -> extracting images to {data_images_dir} ...")
        safe_makedirs(data_images_dir)
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                members = [m for m in tar.getmembers() if m.isfile() and (m.name.lower().endswith('.jpg') or m.name.lower().endswith('.png'))]
                for m in tqdm(members, desc='Extracting images'):
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    out_name = os.path.join(data_images_dir, os.path.basename(m.name))
                    if not os.path.exists(out_name):
                        with open(out_name, 'wb') as of:
                            shutil.copyfileobj(f, of)
        except Exception as e:
            print('Failed to extract tar:', e)
    else:
        print('No IAM tar.gz found at', tar_path)

def copy_sample_images(iam_root: str, data_images_dir: str):
    sample_img_dir = os.path.join(iam_root, 'sample', 'image')
    if os.path.isdir(sample_img_dir):
        for p in glob(os.path.join(sample_img_dir, '*')):
            if p.lower().endswith(('.png', '.jpg', '.jpeg')):
                dst = os.path.join(data_images_dir, os.path.basename(p))
                if not os.path.exists(dst):
                    shutil.copy2(p, dst)
        print('Copied sample images from', sample_img_dir)
    else:
        print('No sample/image folder found under', iam_root)

# ---------------------- SAM auto-mask generation -----------------
def generate_and_save_masks(images_dir: str, masks_dir: str, model, device='cuda', max_masks_per_image=5, max_side_for_mask_gen=512, sample_limit=None):
    """Generate union binary masks only for images that are missing masks.
    Parameters:
      sample_limit: if not None, only process first sample_limit images (useful for fast tests)
    """
    try:
        from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
        from mobile_sam.predictor import SamPredictor
    except Exception as e:
        raise ImportError(f"Failed to import SamAutomaticMaskGenerator/SamPredictor from repo: {e}")

    predictor = SamPredictor(model)
    generator = SamAutomaticMaskGenerator(model)

    safe_makedirs(masks_dir)
    img_paths = sorted([p for p in glob(os.path.join(images_dir, '*')) if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if sample_limit is not None:
        img_paths = img_paths[:sample_limit]
    print(f'Generating masks for {len(img_paths)} images into {masks_dir} (only missing masks will be created) ...')
    for p in tqdm(img_paths):
        base = os.path.splitext(os.path.basename(p))[0]
        out_mask = os.path.join(masks_dir, base + '.png')
        if os.path.exists(out_mask):
            continue
        try:
            img0 = Image.open(p).convert('RGB')
            w,h = img0.size
            scale = min(1.0, max_side_for_mask_gen / max(w,h))
            if scale < 1.0:
                img = img0.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            else:
                img = img0
            arr = np.array(img)
            try:
                predictor.set_image(arr)
            except Exception:
                pass
            masks = generator.generate(arr)
            union_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
            count = 0
            for m in masks:
                seg = m.get('segmentation')
                if seg is None:
                    continue
                if isinstance(seg, (list, tuple)):
                    from PIL import ImageDraw
                    mask_img = Image.new('L', (arr.shape[1], arr.shape[0]), 0)
                    draw = ImageDraw.Draw(mask_img)
                    try:
                        pts = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
                        draw.polygon(pts, outline=1, fill=1)
                        arr_mask = np.array(mask_img, dtype=np.uint8)
                    except Exception:
                        continue
                else:
                    arr_mask = np.array(seg, dtype=np.uint8)
                    if arr_mask.ndim == 3:
                        arr_mask = arr_mask[:, :, 0]
                union_mask = np.maximum(union_mask, (arr_mask > 0).astype(np.uint8))
                count += 1
                if max_masks_per_image > 0 and count >= max_masks_per_image:
                    break
            if scale < 1.0:
                union_mask = np.array(Image.fromarray((union_mask*255).astype('uint8')).resize((w,h), Image.NEAREST))//255
            Image.fromarray((union_mask*255).astype('uint8')).save(out_mask, optimize=True)
            del arr, union_mask
            torch.cuda.empty_cache()
        except Exception as e:
            print('Failed to generate mask for', p, 'error:', e)

# ---------------------------- Dataset ----------------------------
class SimpleMaskDataset(Dataset):
    def __init__(self, data_root, split='train', img_exts=('.png', '.jpg', '.jpeg'), transforms=None, max_side=640, sample_limit=None):
        self.img_dir = os.path.join(data_root, 'img')
        self.mask_dir = os.path.join(data_root, 'masks')
        assert os.path.isdir(self.img_dir), f"images dir missing: {self.img_dir}"
        all_imgs = []
        for ext in img_exts:
            all_imgs += glob(os.path.join(self.img_dir, f'*{ext}'))
        all_imgs.sort()
        if sample_limit is not None:
            all_imgs = all_imgs[:sample_limit]
        n = len(all_imgs)
        split_idx = int(n * 0.9)
        sel = all_imgs[:split_idx] if split == 'train' else all_imgs[split_idx:]
        self.items = []
        for p in sel:
            base = os.path.splitext(os.path.basename(p))[0]
            mask_p = os.path.join(self.mask_dir, base + '.png')
            if os.path.exists(mask_p):
                self.items.append((p, mask_p))
            else:
                continue
        self.transforms = transforms
        self.max_side = max_side

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p_img, p_mask = self.items[idx]
        img = Image.open(p_img).convert('RGB')
        mask = Image.open(p_mask).convert('L')
        img = np.array(img)
        mask = np.array(mask)
        mask = (mask > 127).astype('float32')
        h, w = img.shape[:2]
        scale = self.max_side / max(h, w)
        if scale != 1.0:
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            img = np.array(Image.fromarray(img))#.resize((new_w, new_h), Resampling.BILINEAR))
            mask = np.array(Image.fromarray(mask))#.resize((new_w, new_h), Resampling.NEAREST))
        img_t = transforms.ToTensor()(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return {'image': img_t, 'mask': mask_t}

# ------------------------- Model helpers -------------------------
class SimpleConvAdapter(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = max(8, in_ch // 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_ch, kernel_size=1, bias=True)
        )
    def forward(self, x):
        return x + self.conv(x)

def load_mobilesam(pretrained_path=None, device='cuda', repo_path=None, model_hint='vit_t'):
    import sys, os, torch
    if repo_path is None:
        repo_path = os.environ.get('MOBILE_SAM_REPO', r'D:\project\sam\sam\MobileSAM')
    if os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    model = None
    try:
        from mobile_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b, build_sam_vit_t, build_sam  # type: ignore
        hint = (os.environ.get('MOBILE_SAM_MODEL_HINT', model_hint) or model_hint).lower()
        if hint == 'vit_h':
            model = build_sam_vit_h()
        elif hint == 'vit_l':
            model = build_sam_vit_l()
        elif hint == 'vit_b':
            model = build_sam_vit_b()
        elif hint == 'vit_t':
            model = build_sam_vit_t()
        else:
            model = build_sam()
    except Exception as e:
        try:
            import mobile_sam
            from mobile_sam import build_sam
            model = build_sam()
        except Exception as e2:
            raise ImportError(f"无法导入 MobileSAM 构造器: {e} / fallback error: {e2}")

    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(pretrained_path)
        ck = torch.load(pretrained_path, map_location='cpu')
        sd = ck.get('model_state_dict') if isinstance(ck, dict) and 'model_state_dict' in ck else (ck.get('state_dict') if isinstance(ck, dict) and 'state_dict' in ck else ck)
        def try_load(sdct, strict=True):
            try:
                model.load_state_dict(sdct, strict=strict)
                return True, None
            except Exception as e:
                return False, str(e)
        ok, err = try_load(sd, strict=True) if isinstance(sd, dict) else (False, 'checkpoint not dict')
        if not ok:
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                if new_k.startswith('model.'):
                    new_k = new_k[len('model.'):]
                new_sd[new_k] = v
            ok2, err2 = try_load(new_sd, strict=True)
            if not ok2:
                model_keys = set(model.state_dict().keys())
                intersect = {k: v for k, v in new_sd.items() if k in model_keys}
                if len(intersect) == 0:
                    raise RuntimeError(f'加载权重失败: {err} ; {err2}')
                model.load_state_dict(intersect, strict=False)

    model.to(device)
    return model
# ---- 替换段：把 imgs 转成 list of torch.Tensor 并移动到 model device ----
# ---------- 替换这两个函数（直接复制替换旧实现） ----------
import inspect

def tensor_batch_to_torch_list_on_model_device(imgs: torch.Tensor, model):
    """
    把 BxCxHxW tensor -> list of torch.Tensor (C,H,W) 且把每个 tensor 移到与 model 参数相同的 device。
    返回 list 和对应的 original_sizes 列表。
    """
    # 目标 device：取 model 的第一个参数 device（若没有参数则使用 imgs.device）
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        model_dev = imgs.device

    imgs_cpu = imgs.detach().cpu()
    # float 0..1 -> 0..255
    if imgs_cpu.dtype.is_floating_point:
        maxv = float(imgs_cpu.max())
        if maxv <= 1.1:
            imgs_proc = (imgs_cpu * 255.0).clamp(0, 255).to(torch.float32)
        else:
            imgs_proc = imgs_cpu.to(torch.float32)
    else:
        imgs_proc = imgs_cpu.to(torch.float32)

    tensor_list = []
    original_sizes = []
    for i in range(imgs_proc.shape[0]):
        t = imgs_proc[i].contiguous()   # C,H,W
        h, w = t.shape[1], t.shape[2]
        # move to model device
        t = t.to(model_dev)
        tensor_list.append(t)
        original_sizes.append((int(h), int(w)))
    return tensor_list, original_sizes

def model_forward_mobile_sam(model, imgs: torch.Tensor, device=None):
    """
    为 MobileSAM-like forward 构造 batched_input，保证包含 'image' (Tensor on model device)
    和 'original_size' (H,W tuple). 尝试传 multimask_output=False。
    返回 model(...) 的原始输出（上层负责规整返回值）。
    """
    # 识别 forward 的参数签名
    try:
        sig = inspect.signature(model.forward)
        param_names = [p for p in sig.parameters.keys() if p != 'self']
    except Exception:
        param_names = []

    # 如果 forward 期望 batched_input，就构造含必需字段的 list[dict]
    if 'batched_input' in param_names:
        imgs_list, orig_sizes = tensor_batch_to_torch_list_on_model_device(imgs, model)
        batched_input = []
        for i, t in enumerate(imgs_list):
            rec = {
                'image': t,                  # Tensor, C,H,W, on model device
                'original_size': orig_sizes[i],  # tuple (H, W)
                'file_name': f'input_{i}.png'    # 占位字段，若模型不需要可以忽略
            }
            batched_input.append(rec)
        # 调用时尽量传 multimask_output=False，避免多输出复杂后处理
        try:
            return model(batched_input, multimask_output=False)
        except TypeError:
            # 有的实现可能不接受该 kwarg
            return model(batched_input)
    # 否则尝试 tensor-forward 路径
    if 'multimask_output' in param_names:
        try:
            return model(imgs, multimask_output=False)
        except TypeError:
            pass
    # 最后兜底：直接传 tensor
    return model(imgs)
# ---------- 结束替换 ----------
import torch
import torch.nn.functional as F

def normalize_model_output_choose_best(model, raw_out, imgs_tensor, for_training: bool = False):
    """
    兼容训练与评估的输出选择器。

    参数:
      model, raw_out, imgs_tensor - same as before.
      for_training: bool - 如果为 True，则优先选择一个可微/带梯度的张量用于 loss/backprop。
                    如果为 False（默认），优先选择 full-res 'masks' / pred masks 用于可视化评估。

    返回:
      (logits_tensor, sel_key)  # logits_tensor: (B,1,H,W) 未经过 sigmoid（float, on imgs_tensor.device）
    行为要点:
      - for_training=False: 保持之前的“优先使用 masks（full-res）→ 否则 low_res_logits 等并上采样”策略（适合 evaluate）
      - for_training=True: 首先在 raw_out 中寻找 requires_grad 为 True 的 tensor（优先级：pred_logits/pred_masks/logits/low_res_logits），
                          找到则直接返回（并适当 collapse channel 和上采样到 imgs_tensor 大小）。若未找到，则抛错提示。
    """
    import torch
    import torch.nn.functional as F

    def is_tensor(x):
        return isinstance(x, torch.Tensor)

    # find first trainable tensor (requires_grad or grad_fn != None) in nested structure
    def find_trainable(x):
        if isinstance(x, torch.Tensor):
            if x.requires_grad or (hasattr(x, 'grad_fn') and x.grad_fn is not None):
                return x, None
            return None, None
        if isinstance(x, dict):
            # prefer named keys
            for k in ('pred_logits','pred_masks','logits','low_res_logits','outputs','masks'):
                if k in x and isinstance(x[k], torch.Tensor):
                    t = x[k]
                    if t.requires_grad or (hasattr(t, 'grad_fn') and t.grad_fn is not None):
                        return t, k
            # otherwise recurse
            for k, v in x.items():
                t, tk = find_trainable(v)
                if t is not None:
                    return t, tk or k
        if isinstance(x, (list, tuple)):
            for e in x:
                t, tk = find_trainable(e)
                if t is not None:
                    return t, tk
        return None, None

    # find any tensor by preferred keys (non-trainable allowed)
    def find_preferred_tensor(x):
        prefer = ('pred_logits','pred_masks','logits','low_res_logits','masks','outputs')
        if isinstance(x, dict):
            for k in prefer:
                if k in x and isinstance(x[k], torch.Tensor):
                    return x[k], k
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict):
            for k in prefer:
                if k in x[0] and isinstance(x[0][k], torch.Tensor):
                    return x[0][k], k
        # fallback: first tensor found
        found = []
        def gather(y):
            if isinstance(y, torch.Tensor):
                found.append((y, None))
            elif isinstance(y, dict):
                for kk, vv in y.items():
                    gather(vv)
            elif isinstance(y, (list, tuple)):
                for e in y:
                    gather(e)
        gather(x)
        if found:
            return found[0]
        return None, None

    # 1) Training mode: prefer an actually trainable tensor
    if for_training:
        t, key = find_trainable(raw_out)
        if t is None:
            # fallback: maybe normalize returned something useful but not trainable; try preferred keys and warn/raise
            pref, pkey = find_preferred_tensor(raw_out)
            raise RuntimeError(
                "normalize(for_training=True): could not find a trainable (requires_grad) tensor in model output. "
                f"Found a non-trainable candidate `{pkey}` (type={type(pref)}). "
                "Training requires a differentiable logits tensor (e.g. 'pred_logits'/'logits'). "
                "Possible fixes:\n"
                "  - Ensure model.forward returns logits/pred_logits with requires_grad in training mode.\n"
                "  - Call your model in training mode (model.train()) and avoid disabling grad on mask-producing parts.\n"
                "  - If your model only yields 'masks' (non-differentiable), change model to also return logits suitable for loss.\n"
            )
        chosen = t
        chosen_key = key or 'trainable_tensor'
        # ensure float and on correct device
        chosen = chosen.to(imgs_tensor.device).float()
    else:
        # 2) Evaluation/default mode: prefer full-res masks first
        pref_keys = ['masks', 'pred_masks', 'masks_final', 'outputs']
        chosen = None
        chosen_key = None
        if isinstance(raw_out, dict):
            for k in pref_keys:
                if k in raw_out and is_tensor(raw_out[k]):
                    chosen = raw_out[k]
                    chosen_key = k
                    break
        if chosen is None and isinstance(raw_out, (list, tuple)) and len(raw_out) > 0 and isinstance(raw_out[0], dict):
            for k in pref_keys:
                if k in raw_out[0] and is_tensor(raw_out[0][k]):
                    chosen = raw_out[0][k]
                    chosen_key = k
                    break
        if chosen is None:
            # fallback to preferred logits keys
            prefer = ['low_res_logits', 'pred_logits', 'logits', 'pred_masks', 'masks', 'outputs']
            if isinstance(raw_out, dict):
                for k in prefer:
                    if k in raw_out and is_tensor(raw_out[k]):
                        chosen = raw_out[k]; chosen_key = k; break
            elif isinstance(raw_out, (list, tuple)) and len(raw_out) > 0 and isinstance(raw_out[0], dict):
                for k in prefer:
                    if k in raw_out[0] and is_tensor(raw_out[0][k]):
                        chosen = raw_out[0][k]; chosen_key = k; break
            if chosen is None:
                # find any tensor
                found = []
                def gather_any(y):
                    if is_tensor(y):
                        found.append(y)
                    elif isinstance(y, dict):
                        for vv in y.values():
                            gather_any(vv)
                    elif isinstance(y, (list, tuple)):
                        for e in y:
                            gather_any(e)
                gather_any(raw_out)
                if found:
                    chosen = found[0]; chosen_key = 'any_tensor'

        if chosen is None:
            raise RuntimeError("normalize: could not find any tensor-like output in model raw_out.")

        chosen = chosen.to(imgs_tensor.device).float()

    # Now normalize chosen to shape (B,1,H,W)
    logits = chosen
    # Collapse multi-channel instance dim if necessary
    if logits.dim() == 4:
        if logits.shape[1] == 1:
            pass
        else:
            logits = logits.max(dim=1, keepdim=True)[0]
    elif logits.dim() == 3:
        # possibly (B,H,W) or (N,H,W)
        if logits.shape[0] == imgs_tensor.shape[0]:
            logits = logits.unsqueeze(1)
        else:
            logits = logits.unsqueeze(0).unsqueeze(1)
    elif logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0).repeat(imgs_tensor.shape[0],1,1,1)
    else:
        raise RuntimeError(f"normalize: unexpected chosen tensor shape {tuple(logits.shape)}")

    # Upsample to imgs_tensor size if needed (bilinear)
    target_h, target_w = imgs_tensor.shape[2], imgs_tensor.shape[3]
    if logits.shape[2] != target_h or logits.shape[3] != target_w:
        logits = F.interpolate(logits, size=(target_h, target_w), mode='bilinear', align_corners=False)

    return logits, chosen_key



def model_forward_safe(model, imgs, device='cuda'):
    """
    调用 model 的 forward，兼容需要额外参数（如 multimask_output）的实现。
    优先直接调用 model(imgs)。若报 TypeError 且提示 'multimask_output'，尝试补上该 kwarg。
    若还有其他必需参数，会尝试用常见的默认值推断（保守地填 False / None）。
    返回 model 的原始输出（可能是 tensor 或 tuple/list/dict）。
    """
    try:
        return model(imgs)
    except TypeError as e:
        msg = str(e)
        # 若明确提示缺少 multimask_output 参数，先尝试填 False
        if 'multimask_output' in msg:
            try:
                return model(imgs, multimask_output=False)
            except Exception:
                pass
        # 通用尝试：用 inspect 看 forward 的参数列表并为没有默认值的参数填保守值
        try:
            sig = inspect.signature(model.forward)
            kwargs = {}
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if name == 'images' or name == 'x' or name == 'input':
                    continue
                # 若 param 有默认值就跳过（会被使用默认）
                if param.default is not inspect._empty:
                    continue
                # 为常见布尔型参数填 False
                if 'multi' in name or 'mask' in name or 'return' in name:
                    kwargs[name] = False
                else:
                    kwargs[name] = None
            return model(imgs, **kwargs)
        except Exception:
            # 兜底：把错误抛回去
            raise


# --------- INTERNAL API TRAINING WRAPPER (paste into your script) ----------
import types, inspect

import inspect, types, torch

# 辅助：打印并分析 model.forward 返回的复杂结构
def analyze_model_output(out):
    """
    打印 out 的内部结构，返回 True 如果发现至少一个 Tensor 且该 Tensor 有 grad_fn / requires_grad True。
    便于诊断为什么 loss 无梯度。
    """
    found_good = False
    print("ANALYZE OUTPUT: type:", type(out))
    if isinstance(out, torch.Tensor):
        print("  Tensor shape:", out.shape, "dtype:", out.dtype, "device:", out.device,
              "requires_grad:", out.requires_grad, "grad_fn:", type(out.grad_fn).__name__ if out.grad_fn is not None else None)
        if out.requires_grad and out.grad_fn is not None:
            found_good = True
        return found_good

    if isinstance(out, (list, tuple)):
        print(f"  list/tuple len={len(out)} — inspecting elements...")
        for i, e in enumerate(out):
            print(f"   [{i}] type={type(e)}")
            if isinstance(e, torch.Tensor):
                print(f"       tensor shape={e.shape}, requires_grad={e.requires_grad}, grad_fn={type(e.grad_fn).__name__ if e.grad_fn is not None else None}")
                if e.requires_grad and e.grad_fn is not None:
                    found_good = True
            elif isinstance(e, dict):
                for k, v in e.items():
                    if isinstance(v, torch.Tensor):
                        print(f"       dict[{k}] tensor shape={v.shape}, requires_grad={v.requires_grad}, grad_fn={type(v.grad_fn).__name__ if v.grad_fn is not None else None}")
                        if v.requires_grad and v.grad_fn is not None:
                            found_good = True
                    else:
                        print(f"       dict[{k}] type={type(v)}")
            else:
                print(f"       element type: {type(e)}")
        return found_good

    if isinstance(out, dict):
        print("  dict keys:", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: tensor shape={v.shape}, requires_grad={v.requires_grad}, grad_fn={type(v.grad_fn).__name__ if v.grad_fn is not None else None}")
                if v.requires_grad and v.grad_fn is not None:
                    found_good = True
            elif isinstance(v, (list, tuple)):
                print(f"    {k}: list/tuple len={len(v)}")
                for i, e in enumerate(v):
                    if isinstance(e, torch.Tensor):
                        print(f"       [{i}] tensor shape={e.shape}, requires_grad={e.requires_grad}, grad_fn={type(e.grad_fn).__name__ if e.grad_fn is not None else None}")
                        if e.requires_grad and e.grad_fn is not None:
                            found_good = True
                    else:
                        print(f"       [{i}] type={type(e)}")
            else:
                print(f"    {k}: type={type(v)}")
        return found_good

    # fallback: try convert
    try:
        t = torch.as_tensor(out)
        print("  converted to tensor shape", t.shape, "requires_grad", t.requires_grad)
        return t.requires_grad and t.grad_fn is not None
    except Exception:
        print("  not convertible to tensor")
        return False


# reuse find_candidates & try_call_with_args patterns (robust scanning)
def find_candidates(model):
    mods = list(model.named_modules())
    names = [n for n,_ in mods]
    cand = {'backbone': [], 'prompt_encoder': [], 'mask_decoder': [], 'image_encoder': []}
    for n,m in mods:
        ln = n.lower()
        if any(k in ln for k in ('backbone','image_encoder','vision','visual','encoder','vit','patch_embed')):
            cand['backbone'].append((n,m))
        if 'prompt' in ln:
            cand['prompt_encoder'].append((n,m))
        if any(k in ln for k in ('mask_decoder','mask_head','mask_predictor','mask_generator','maskout','mask')):
            cand['mask_decoder'].append((n,m))
        if any(k in ln for k in ('image_encoder','image_embed','visual')):
            cand['image_encoder'].append((n,m))
    # also attributes
    for a in dir(model):
        la = a.lower()
        if any(k in la for k in ('backbone','image_encoder','vision','encoder','vit','patch_embed','image_embed')):
            try: cand['backbone'].append((a, getattr(model, a)))
            except Exception: pass
        if 'prompt' in la:
            try: cand['prompt_encoder'].append((a, getattr(model, a)))
            except Exception: pass
        if any(k in la for k in ('mask_decoder','mask_head','mask_predictor','mask_generator','maskout')):
            try: cand['mask_decoder'].append((a, getattr(model, a)))
            except Exception: pass
    # dedup
    for k in cand:
        seen=set(); new=[]
        for n,m in cand[k]:
            if n not in seen:
                seen.add(n); new.append((n,m))
        cand[k]=new
    return cand

def try_call_with_args(fn, *args, **kwargs):
    """尝试多种调用签名以调用模块/函数，失败返回 None"""
    if fn is None:
        return None
    try:
        if isinstance(fn, types.MethodType) or callable(fn):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                # 尝试常见 keyword
                try:
                    sig = inspect.signature(fn)
                    pnames = [p for p in sig.parameters.keys() if p!='self']
                except Exception:
                    pnames = []
                kw = {}
                if len(args)>0:
                    if 'image' in pnames: kw['image']=args[0]
                    if 'images' in pnames: kw['images']=args[0]
                    if 'batched_input' in pnames: kw['batched_input']=args[0]
                    if 'image_embeddings' in pnames: kw['image_embeddings']=args[0]
                if kw:
                    try:
                        return fn(**kw)
                    except Exception:
                        return None
                return None
        return None
    except Exception:
        return None

# 主函数：训练用 forward（更智能的降级与诊断）
def model_train_forward(model, imgs_tensor, device='cuda'):
    """
    更积极的训练用 forward wrapper：
      - 尝试 bypass decorator via model.forward.__wrapped__ (开启 grad)
      - 尝试高层调用 model(batched_input, multimask_output=False)
      - 尝试内部路径 backbone -> prompt_encoder -> mask_decoder
    返回 model 的原始输出（应包含可微分张量）。
    在失败时会打印详细诊断供你贴回以便定位。
    """
    import torch, types
    from torch import nn
    import inspect

    B, C, H, W = imgs_tensor.shape
    # prepare imgs on device (preserve grad-ability)
    try:
        model_dev = next(model.parameters()).device
    except StopIteration:
        model_dev = torch.device(device)

    imgs_proc = imgs_tensor
    if imgs_proc.dtype.is_floating_point and float(imgs_proc.max()) <= 1.1:
        imgs_proc = (imgs_proc * 255.0).to(torch.float32)
    else:
        imgs_proc = imgs_proc.to(torch.float32)
    imgs_on_dev = imgs_proc.to(model_dev)

    batched_input = [{'image': imgs_on_dev[i].contiguous(), 'original_size': (int(H), int(W)), 'file_name': f'input_{i}.png'} for i in range(B)]

    # helper to check grad connectivity
    def has_grad_tensor(out):
        # reuse analyze_model_output style but only return boolean
        import torch
        def recurse(x):
            if isinstance(x, torch.Tensor):
                if x.requires_grad or (hasattr(x, 'grad_fn') and x.grad_fn is not None):
                    return True
                return False
            if isinstance(x, dict):
                for v in x.values():
                    if recurse(v):
                        return True
            if isinstance(x, (list, tuple)):
                for e in x:
                    if recurse(e):
                        return True
            return False
        try:
            return recurse(out)
        except Exception:
            return False

    # 1) try calling wrapped forward (bypass decorator) if available
    if hasattr(model.forward, "__wrapped__"):
        try:
            print("TRAIN FORWARD: trying model.forward.__wrapped__ with grad enabled (bypass decorator)...")
            was_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            out = model.forward.__wrapped__(model, batched_input, multimask_output=False)
            torch.set_grad_enabled(was_grad)
            print("TRAIN FORWARD: __wrapped__ returned. checking for grad-connected tensors...")
            if has_grad_tensor(out):
                print("TRAIN FORWARD: __wrapped__ output contains grad-connected tensor(s). Using this output.")
                return out
            else:
                print("TRAIN FORWARD: __wrapped__ output NOT grad-connected. Will try other paths.")
        except Exception as e:
            print("TRAIN FORWARD: __wrapped__ call failed:", e)

    # 2) try high-level call
    try:
        print("TRAIN FORWARD: trying model(batched_input, multimask_output=False)...")
        out = model(batched_input, multimask_output=False)
        print("TRAIN FORWARD: high-level call returned. checking grad connectivity...")
        if has_grad_tensor(out):
            print("TRAIN FORWARD: high-level model(...) output contains grad-connected tensor(s). Using this output.")
            return out
        else:
            print("TRAIN FORWARD: high-level model(...) output NOT grad-connected. Will try alternate call patterns.")
    except TypeError as e:
        # try without multimask_output kw
        try:
            out = model(batched_input)
            print("TRAIN FORWARD: model(batched_input) succeeded. checking grad connectivity...")
            if has_grad_tensor(out):
                print("TRAIN FORWARD: model(batched_input) output contains grad-connected tensor(s). Using this output.")
                return out
            else:
                print("TRAIN FORWARD: model(batched_input) output NOT grad-connected.")
        except Exception as e2:
            print("TRAIN FORWARD: model(batched_input) also failed:", e2)
    except Exception as e:
        print("TRAIN FORWARD: model(batched_input, multimask_output=False) raised:", e)

    # 3) internal API path (backbone -> prompt_encoder -> mask_decoder)
    print("TRAIN FORWARD: attempting internal API path (backbone -> prompt -> mask_decoder)...")
    # find candidate modules
    def find_candidates_local(m):
        cand = {'backbone': [], 'prompt_encoder': [], 'mask_decoder': [], 'image_encoder': []}
        for n, module in list(m.named_modules()):
            ln = n.lower()
            if any(k in ln for k in ('backbone','image_encoder','vision','visual','encoder','vit','patch_embed')):
                cand['backbone'].append((n, module))
            if 'prompt' in ln:
                cand['prompt_encoder'].append((n, module))
            if any(k in ln for k in ('mask_decoder','mask_head','mask_predictor','mask_generator','maskout','mask')):
                cand['mask_decoder'].append((n, module))
            if any(k in ln for k in ('image_encoder','image_embed','visual')):
                cand['image_encoder'].append((n, module))
        # also attributes
        for a in dir(m):
            la = a.lower()
            try:
                attr = getattr(m, a)
            except Exception:
                continue
            if any(k in la for k in ('backbone','image_encoder','vision','encoder','vit','patch_embed','image_embed')):
                cand['backbone'].append((a, attr))
            if 'prompt' in la:
                cand['prompt_encoder'].append((a, attr))
            if any(k in la for k in ('mask_decoder','mask_head','mask_predictor','mask_generator','maskout')):
                cand['mask_decoder'].append((a, attr))
        # dedup names
        for k in cand:
            seen=set(); new=[]
            for n,mv in cand[k]:
                if n not in seen:
                    seen.add(n); new.append((n,mv))
            cand[k]=new
        return cand

    cand = find_candidates_local(model)
    print(f"TRAIN FORWARD: candidate summary: backbone {len(cand['backbone'])}, prompt {len(cand['prompt_encoder'])}, mask_decoder {len(cand['mask_decoder'])}")

    # try encoders to get feature
    feature = None
    for name, enc in cand['image_encoder'] + cand['backbone']:
        if enc is None:
            continue
        try:
            # try passing tensor directly
            try:
                out_enc = enc(imgs_on_dev)
            except Exception:
                out_enc = None
            if out_enc is None:
                # try batched_input
                try:
                    out_enc = enc(batched_input)
                except Exception:
                    out_enc = None
            if out_enc is not None:
                feature = out_enc
                print(f"TRAIN FORWARD: encoder `{name}` returned feature type {type(feature)}")
                break
        except Exception as e:
            print(f"TRAIN FORWARD: encoder `{name}` call failed: {e}")

    if feature is None:
        print("TRAIN FORWARD: failed to obtain encoder feature from candidates. Printing candidates for debugging:")
        print("backbone candidates:", [n for n,_ in cand['backbone']][:10])
        print("image_encoder candidates:", [n for n,_ in cand['image_encoder']][:10])
        raise RuntimeError("TRAIN FORWARD: could not obtain encoder feature via internal API. Please inspect model structure.")

    # try prompt encoder
    prompt_out = None
    for name, p in cand['prompt_encoder']:
        if p is None:
            continue
        try:
            po = None
            try:
                po = p(feature)
            except Exception:
                try:
                    po = p(batched_input)
                except Exception:
                    po = None
            if po is not None:
                prompt_out = po
                print(f"TRAIN FORWARD: prompt_encoder `{name}` returned type {type(prompt_out)}")
                break
        except Exception as e:
            print(f"TRAIN FORWARD: prompt_encoder `{name}` failed: {e}")

    # try mask_decoder candidates
    for name, md in cand['mask_decoder']:
        if md is None:
            continue
        try:
            # several attempt signatures
            attempts = [
                lambda md=md: md(feature, prompt_out) if callable(md) else None,
                lambda md=md: md(prompt_out, feature) if callable(md) else None,
                lambda md=md: md(feature) if callable(md) else None,
                lambda md=md: md(batched_input) if callable(md) else None,
                lambda md=md: md(imgs_on_dev) if callable(md) else None,
            ]
            for attempt in attempts:
                try:
                    res = attempt()
                except Exception as e:
                    res = None
                if res is not None:
                    print(f"TRAIN FORWARD: mask_decoder `{name}` produced type {type(res)} — checking for grad...")
                    if has_grad_tensor(res):
                        print("TRAIN FORWARD: mask_decoder output has grad-connected tensor(s). Using this output.")
                        return res
                    else:
                        print("TRAIN FORWARD: mask_decoder output NOT grad-connected; continuing search.")
        except Exception as e:
            print(f"TRAIN FORWARD: mask_decoder `{name}` attempt raised: {e}")

    # final fallback: direct model(imgs_on_dev)
    try:
        print("TRAIN FORWARD: final fallback try model(imgs_on_dev) ...")
        out = model(imgs_on_dev)
        if has_grad_tensor(out):
            print("TRAIN FORWARD: final fallback produced grad-connected output. Using it.")
            return out
        else:
            print("TRAIN FORWARD: final fallback output NOT grad-connected.")
    except Exception as e:
        print("TRAIN FORWARD: final fallback model(imgs_on_dev) failed:", e)

    # if we arrive here, none produced grad-connected outputs
    # Print diagnostics: list tensors in last 'out' or raw_out with their requires_grad and shapes
    print("TRAIN FORWARD: Diagnostics — listing tensors found in last outputs (raw_out and last internal outputs).")
    def list_tensors(x, prefix=''):
        import torch
        if isinstance(x, torch.Tensor):
            print(f"{prefix} TENSOR shape={tuple(x.shape)} dtype={x.dtype} requires_grad={x.requires_grad}")
            return
        if isinstance(x, dict):
            for k,v in x.items():
                list_tensors(v, prefix + f"{k}.")
            return
        if isinstance(x, (list, tuple)):
            for i,e in enumerate(x):
                list_tensors(e, prefix + f"[{i}].")
            return
        # else skip

    try:
        print("RAW_OUT tensors:")
        list_tensors(raw_out, 'raw_out.')
    except Exception as e:
        print("Error printing raw_out:", e)

    raise RuntimeError("TRAIN FORWARD: tried multiple calling strategies but could not get grad-enabled output. "
                       "Likely the model's forward is decorated with @torch.no_grad or the logits are detached. "
                       "Check model implementation to ensure it returns differentiable logits/predictions in training mode.")




def inject_adapter_to_mask_decoder(model, max_adapters=32):
    """
    更安全的 adapter 注入：
    - 先 snapshot 一份 model.named_modules() 列表（避免在遍历时修改结构）
    - 收集目标 module（名字含 mask_decoder/mask_head/mask_predictor）
    - 在第二遍中用 module.add_module(...) 附加 adapter
    - max_adapters 控制最多附加多少个 adapter，防止误注入太多
    """
    targets = []  # list of (name, module, sub_name, out_channels)
    # 1) snapshot 遍历（不会受后续 add_module 影响）
    for name, module in list(model.named_modules()):
        # 筛选感兴趣的 module 名称（你可以按实际 repo 调整匹配词）
        if ('mask_decoder' in name) or ('mask_head' in name) or ('mask_predictor' in name) or ('mask' in name and 'decoder' in name):
            # 收集该 module 的直接 child conv 层信息（保守处理）
            for sub_name, sub in module.named_children():
                if isinstance(sub, nn.Conv2d):
                    out_ch = getattr(sub, 'out_channels', None)
                    if out_ch is None:
                        continue
                    targets.append((name, module, sub_name, out_ch))

    # 2) 实际注入（一次性进行）
    added = 0
    for name, module, sub_name, out_ch in targets:
        if added >= max_adapters:
            break
        adapter_name = f"_adapter_{sub_name}"
        # 如果已经存在则跳过（避免重复注入）
        if hasattr(module, adapter_name):
            continue
        try:
            adapter = SimpleConvAdapter(out_ch)
            # use add_module so it is registered as submodule
            module.add_module(adapter_name, adapter)
            added += 1
        except Exception as e:
            print(f"Failed to attach adapter to {name}.{sub_name}: {e}")
            continue

    print(f"Safely injected {added} adapters (collected {len(targets)} candidate positions).")
    return model

def freeze_backbone_but_train_adapters(model):
    for n, p in model.named_parameters():
        if 'adapter' in n or 'mask' in n or 'mask_decoder' in n or 'mask_head' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

# ------------------------- Training & Eval -------------------------
def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=[1,2,3])
    union = ((pred + target) >= 1).float().sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=[1,2,3])
    denom = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()

def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / (total + 1e-12)).item()


# ------------------------- Evaluation (with smooth visualization) -------------------------
from PIL import Image, ImageDraw


def evaluate(model, loader, device, results_dir=None, prob_thr=0.5):
    """
    简洁版 evaluate：
      - 计算每张图的 iou/dice/acc 并写 eval_per_image.csv
      - 为每张图保存 overlay 图（pred_mask 叠加在原图上）
      - results_dir: 保存路径（会创建）
      - prob_thr: 二值化阈值
    依赖：model_train_forward(), normalize_model_output_choose_best() 已在脚本中定义。
    """
    import os
    import csv
    import numpy as np
    import cv2
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Eval")):
            imgs = batch['image'].to(device)     # B,C,H_i,W_i
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(device)        # B,1,H_m,W_m
            B = imgs.shape[0]

            # forward
            raw_out = model_train_forward(model, imgs, device)
            logits, sel_key = normalize_model_output_choose_best(model, raw_out, imgs)
            probs = torch.sigmoid(logits)  # Tensor B,1,h_m,w_m

            # Ensure probs on CPU for numpy ops when needed
            probs_cpu = probs.detach().cpu().numpy()  # B,1,h_m,w_m

            # Loop per item in batch
            for i in range(B):
                # get per-item original image (prefer dataset-provided path)
                orig_path = None
                if isinstance(batch.get('path', None), (list, tuple)):
                    orig_path = batch['path'][i]
                elif isinstance(batch.get('path', None), str):
                    orig_path = batch['path']

                # determine orig (W_o,H_o)
                if orig_path and os.path.exists(orig_path):
                    orig_pil = Image.open(orig_path).convert('RGB')
                    W_o, H_o = orig_pil.size
                    img_orig = np.array(orig_pil)
                else:
                    # fallback: use loader-provided tensor image (imgs)
                    img_t = imgs[i].cpu().permute(1,2,0).numpy()
                    if img_t.max() <= 1.1:
                        img_orig = (img_t * 255).astype(np.uint8)
                    else:
                        img_orig = img_t.astype(np.uint8)
                    H_o, W_o = img_orig.shape[0], img_orig.shape[1]

                # pick corresponding prob map and resize to orig size
                prob_map = probs_cpu[i, 0]  # h_m, w_m, float 0..1
                prob_resized = cv2.resize(prob_map.astype(np.float32), (W_o, H_o), interpolation=cv2.INTER_LINEAR)

                # binary mask
                mask_bin = (prob_resized > prob_thr).astype(np.uint8)  # H_o, W_o

                # morphology clean-up (optional, keeps it conservative)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

                # compute metrics — compare to provided ground-truth mask if available (upsample masks if needed)
                if masks is not None:
                    # ensure gt in cpu numpy, upsample to mask_clean size if needed
                    gt = masks[i, 0].detach().cpu().numpy()  # maybe already H_o,W_o or different
                    gt_h, gt_w = gt.shape
                    if (gt_h != H_o) or (gt_w != W_o):
                        gt_resized = cv2.resize(gt.astype(np.float32), (W_o, H_o), interpolation=cv2.INTER_NEAREST)
                        gt_bin = (gt_resized > 0.5).astype(np.uint8)
                    else:
                        gt_bin = (gt > 0.5).astype(np.uint8)

                    # compute IoU, Dice, PixelAcc (per-image)
                    inter = (mask_clean & gt_bin).sum()
                    union = ((mask_clean | gt_bin).sum())
                    iou = float((inter + 1e-6) / (union + 1e-6))
                    dice = float((2*inter + 1e-6) / (mask_clean.sum() + gt_bin.sum() + 1e-6))
                    acc = float((mask_clean == gt_bin).sum() / (mask_clean.size + 1e-12))
                else:
                    iou = float('nan'); dice = float('nan'); acc = float('nan')

                rows.append({'iou': iou, 'dice': dice, 'acc': acc})

                if results_dir:
                    # Save visualizations: prob, binary mask, overlay on original
                    idx_global = batch_idx * B + i
                    # pred_prob
                    prob_vis = (np.clip(prob_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
                    Image.fromarray(prob_vis).save(os.path.join(results_dir, f'pred_prob_{idx_global}.png'))

                    # pred_mask (clean)
                    Image.fromarray((mask_clean*255).astype(np.uint8)).save(os.path.join(results_dir, f'pred_mask_{idx_global}.png'))

                    # overlay mask on orig image (semi-transparent)
                    img_cv = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                    mask_color = np.stack([mask_clean*255]*3, axis=-1).astype(np.uint8)
                    overlay = cv2.addWeighted(img_cv, 1.0, mask_color, 0.35, 0)
                    # draw contours (optional), helps show boundaries clearly
                    contours, _ = cv2.findContours(mask_clean.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (255,0,0), 2)  # blue contours in BGR

                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    Image.fromarray(overlay_rgb).save(os.path.join(results_dir, f'pred_overlay_{idx_global}.png'))

    # write per-image csv (in dataset order)
    if results_dir:
        csv_path = os.path.join(results_dir, 'eval_per_image.csv')
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['iou','dice','acc'])
            writer.writeheader()
            writer.writerows(rows)

    # return mean metrics (nan-safe)
    import numpy as _np
    vals_iou = [r['iou'] for r in rows if not (r['iou'] is None or (_np.isnan(r['iou'])))]
    vals_dice = [r['dice'] for r in rows if not (r['dice'] is None or (_np.isnan(r['dice'])))]
    vals_acc = [r['acc'] for r in rows if not (r['acc'] is None or (_np.isnan(r['acc'])))]
    mean_iou = float(_np.mean(vals_iou)) if len(vals_iou)>0 else float('nan')
    mean_dice = float(_np.mean(vals_dice)) if len(vals_dice)>0 else float('nan')
    mean_acc = float(_np.mean(vals_acc)) if len(vals_acc)>0 else float('nan')

    print(f"Eval finished. mean IoU={mean_iou:.4f} mean Dice={mean_dice:.4f} mean Acc={mean_acc:.4f}")
    return mean_iou, mean_dice, mean_acc




def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=[1,2,3])
    denom = probs.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3])
    loss = 1.0 - ((num + eps) / (denom + eps))
    return loss.mean()

def train_loop(model, train_loader, val_loader, args):
    device = args.device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = GradScaler()
    best_val = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for i, batch in pbar:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)

            with autocast():  # 保持你当前的 AMP 写法或用 torch.amp.autocast('cuda') 更新
                # forward (use your robust forward wrapper if you have one)
                raw_out = model_train_forward(model, imgs, device) if 'model_train_forward' in globals() else model(imgs)
                # normalize output into logits (B,1,H,W). If you have normalize_model_output, use it.
                with autocast():
                    raw_out = model_train_forward(model, imgs,
                                                  device=args.device) if 'model_train_forward' in globals() else model(
                        imgs)
                    logits, sel_key = normalize_model_output_choose_best(model, raw_out, imgs , for_training = True)
                    # logits 是 (B,1,H,W)，connected to model if sel_key corresponds to grad-enabled tensor
                    loss_bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                    print("Test if no difference", logits.min().item(), logits.max().item(), logits.mean().item())
                    probs = torch.sigmoid(logits)
                    inter = (probs * masks).sum()
                    dice = dice_loss_from_logits(logits, masks)
                    loss = 0.6 * loss_bce + 0.4 * dice
                    loss = loss / args.accum_steps
                # optionally print which key was used
                pbar.set_postfix({'loss': running / (i + 1), 'used': sel_key})

            # --- Diagnostic & fallback block: check whether loss connects to any trainable param ---
            # collect current trainable params
            current_trainable = [p for p in model.parameters() if p.requires_grad]
            if len(current_trainable) == 0:
                raise RuntimeError("No trainable parameters found before backward. Aborting.")

            # Try to compute grads w.r.t current trainable params without actually performing backward yet.
            # Use allow_unused=True to avoid exceptions; result entries will be None if no grad can be computed for that param.
            grads = torch.autograd.grad(loss, current_trainable, retain_graph=True, allow_unused=True)
            none_count = sum(1 for g in grads if g is None)
            if none_count == len(grads):
                # No parameter receives gradient -> fallback
                print("WARNING: loss has no gradient path to current trainable parameters (all grads None). Attempting fallback.")
                # 1) Try selective unfreeze of mask/decoder/prompt-related params (non-invasive)
                unfrozen = 0
                for n, p in model.named_parameters():
                    if any(k in n.lower() for k in ('mask', 'decoder', 'pred', 'head', 'prompt', 'mask_downscaling')):
                        if not p.requires_grad:
                            p.requires_grad = True
                            unfrozen += 1
                print(f"Fallback unfreeze attempt: {unfrozen} params un-froze based on keywords.")
                # 2) Rebuild optimizer if we changed trainable set
                new_params = [p for p in model.parameters() if p.requires_grad]
                if set(new_params) != set(current_trainable):
                    optimizer = torch.optim.AdamW(new_params, lr=args.lr)
                    optimizer.zero_grad()
                    # Recompute forward & loss (do it outside the AMP context to be safe)
                    with autocast():
                        raw_out = model_forward_mobile_sam(model, imgs, device) if 'model_forward_mobile_sam' in globals() else model(imgs)
                        logits = normalize_model_output_choose_best(model, raw_out, imgs, for_training = True) if 'normalize_model_output' in globals() else raw_out
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0]
                        if logits.dim() == 4 and logits.shape[1] != 1:
                            logits = logits[:, :1, :, :]
                        loss_bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                        #probs = torch.sigmoid(logits)
                        #inter = (probs * masks).sum()
                        dice = dice_loss_from_logits(logits, masks)
                        loss = 0.6 * loss_bce + 0.4 * dice
                        loss = loss / args.accum_steps

                    # Check grads again
                    grads2 = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                    if all(g is None for g in grads2):
                        # 3) Last-resort: unfreeze ALL parameters (warning about memory)
                        print("Fallback step 2 failed: still no grads. Last-resort: unfreezing ALL model parameters (may increase memory usage).")
                        for n, p in model.named_parameters():
                            p.requires_grad = True
                        # rebuild optimizer for all params
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        # recompute forward & loss
                        with autocast():
                            raw_out = model_forward_mobile_sam(model, imgs,
                                                               device) if 'model_forward_mobile_sam' in globals() else model(
                                imgs)
                            logits = normalize_model_output_choose_best(model, raw_out,
                                                                        imgs, for_training = True) if 'normalize_model_output' in globals() else raw_out
                            if isinstance(logits, (list, tuple)):
                                logits = logits[0]
                            if logits.dim() == 4 and logits.shape[1] != 1:
                                logits = logits[:, :1, :, :]
                            loss_bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                            probs = torch.sigmoid(logits)
                            inter = (probs * masks).sum()
                            dice = dice_loss_from_logits(logits, masks)
                            loss = 0.6 * loss_bce + 0.4 * dice
                            loss = loss / args.accum_steps
                        # proceed to backward (hope this time grads exist)
                    else:
                        # grads now exist, continue
                        pass
                else:
                    # optimizer params unchanged but grads all None -> escalate to global unfreeze
                    print("Fallback: no change to optimizer but grads missing -> unfreezing all parameters.")
                    for n, p in model.named_parameters():
                        p.requires_grad = True
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    optimizer.zero_grad()
                    with autocast():
                        raw_out = model_forward_mobile_sam(model, imgs,
                                                           device) if 'model_forward_mobile_sam' in globals() else model(
                            imgs)
                        logits = normalize_model_output_choose_best(model, raw_out,
                                                                    imgs, for_training = True) if 'normalize_model_output' in globals() else raw_out
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0]
                        if logits.dim() == 4 and logits.shape[1] != 1:
                            logits = logits[:, :1, :, :]
                        loss_bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                        print("Test if no difference",logits.min().item(), logits.max().item(), logits.mean().item())
                        probs = torch.sigmoid(logits)
                        inter = (probs * masks).sum()
                        dice = dice_loss_from_logits(logits, masks)
                        loss = 0.6 * loss_bce + 0.4 * dice
                        loss = loss / args.accum_steps

            # --- end diagnostic/fallback ---

            # now safe to backward (assumes at least some grads exist)
            scaler.scale(loss).backward()
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running += loss.item() * args.accum_steps
            pbar.set_postfix({'loss': running / (i + 1)})
        # end epoch
        val_iou, val_dice, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} VAL IoU {val_iou:.4f} Dice {val_dice:.4f} Acc {val_acc:.4f}")
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = os.path.join(args.save_dir, f"ckpt_epoch{epoch}_iou{val_iou:.4f}.pth")
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_iou': val_iou}, ckpt)
        if val_iou > best_val:
            best_val = val_iou
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_iou': val_iou}, os.path.join(args.save_dir, 'best.pth'))
    print("Training finished")


# ---------------------------- CLI & Main -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=str, required=True, help='Local MobileSAM repo path')
    p.add_argument('--tt-root', type=str, required=True, help='Total-text root (contains train and test)')
    p.add_argument('--data-root', type=str, required=True, help='Directory where data/images and data/masks will be stored/loaded')
    p.add_argument('--pretrained-path', type=str, default='', help='Optional pretrained weights path')
    p.add_argument('--pretrained-url', type=str, default='', help='If specified, download pretrained from this URL when pretrained-path missing')
    p.add_argument('--hf-repo', type=str, default='', help='HuggingFace repo id (e.g. username/reponame)')
    p.add_argument('--hf-filename', type=str, default='', help='filename in hf repo to download')
    p.add_argument('--model-hint', type=str, default='vit_t', help='which build_sam variant to use (vit_t/vit_b/...)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accum-steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--save-dir', type=str, default='./ckpts')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--mask-sample-limit', type=int, default=500, help='Only generate masks for first N images (useful for testing). 0 -> all')
    p.add_argument('--mask-max-side', type=int, default=512, help='Longest side used for mask generation')
    p.add_argument('--mask-max-instances', type=int, default=5, help='Max instances per image to union')
    p.add_argument('--sample-limit', type=int, default=0, help='Only use first N images for dataset (0->all)')
    return p.parse_args()

def main():
    args = parse_args()
    repo = args.repo
    tt_root = args.tt_root
    data_root = args.data_root
    train_data_root = os.path.join(data_root, 'train')
    train_data_images_dir = os.path.join(train_data_root, 'img')
    train_data_ann_dir = os.path.join(train_data_root, 'ann')
    train_masks_dir = os.path.join(train_data_root, 'masks')

    test_data_root = os.path.join(data_root, 'test')
    test_data_images_dir = os.path.join(test_data_root, 'img')
    test_data_ann_dir = os.path.join(test_data_root, 'ann')
    test_masks_dir = os.path.join(test_data_root, 'masks')
    sample_limit = args.mask_sample_limit if args.mask_sample_limit and args.mask_sample_limit > 0 else None


    # prepare images
    processed, skipped, skipped_list = prepare_totaltext_masks_from_tar(tt_root, data_root, split='train', visualize=False, sample_limit=sample_limit)
    print (f'Prepared train images: {processed}, skipped: {skipped}')

    # add repo to path
    add_repo_to_path(repo)

    # ensure pretrained (if path provided or download settings)
    pretrained_to_use = ''
    if args.pretrained_path:
        try:
            ensured = ensure_pretrained(args.pretrained_path, pretrained_url=args.pretrained_url or None, hf_repo=(args.hf_repo or None), hf_filename=(args.hf_filename or None))
            pretrained_to_use = ensured
        except Exception as e:
            print('Pretrained ensure failed:', e)
            pretrained_to_use = ''
    else:
        # if not provided, still attempt if hf args present or env PRETRAINED_URL
        env_url = os.environ.get('PRETRAINED_URL', '')
        hf_repo = args.hf_repo if args.hf_repo else None
        hf_filename = args.hf_filename if args.hf_filename else None
        if env_url:
            try:
                tmp_path = os.path.join(args.save_dir, 'downloaded_pretrained.pth')
                ensure_pretrained(tmp_path, pretrained_url=env_url)
                pretrained_to_use = tmp_path
            except Exception as e:
                print('Env PRETRAINED_URL download failed:', e)
        elif hf_repo and hf_filename:
            try:
                tmp_path = os.path.join(args.save_dir, hf_filename)
                ensure_pretrained(tmp_path, hf_repo=hf_repo, hf_filename=hf_filename)
                pretrained_to_use = tmp_path
            except Exception as e:
                print('HF download failed:', e)

    # load model (used also for mask gen)
    model = load_mobilesam(pretrained_path=pretrained_to_use if pretrained_to_use else None, device=args.device, repo_path=repo, model_hint=args.model_hint)

    # generate masks for missing images (only first mask-sample-limit images if set)


    # dataset and dataloaders
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        train_tf = A.Compose([
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=0),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
        val_tf = A.Compose([A.LongestMaxSize(max_size=640), A.PadIfNeeded(640,640), A.Normalize(), ToTensorV2()])
    except Exception:
        train_tf = None
        val_tf = None

    sample_limit = args.sample_limit if args.sample_limit and args.sample_limit > 0 else None
    train_ds = SimpleMaskDataset(train_data_root, split='train', transforms=train_tf, max_side=640, sample_limit=sample_limit)
    val_ds = SimpleMaskDataset(train_data_root, split='val', transforms=val_tf, max_side=640, sample_limit=sample_limit)
    print(f"Train samples: {len(train_ds)} Val samples: {len(val_ds)}")
    if len(train_ds) == 0:
        print('No training samples found (check that masks exist or mask generation succeeded). Exiting.')
        return
    # Windows 下稳定配置
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # inject adapters, freeze backbone
    model = inject_adapter_to_mask_decoder(model)

    # 若注入适配器后没有任何 adapter 被注入（你的 inject 函数可能找不到匹配），降级策略：解冻模型中包含关键词的参数，让它们被训练
    def ensure_some_params_trainable(model):
        trainable = any(p.requires_grad for p in model.parameters())
        # 如果全部被 freeze（即没有 requires_grad True），则解冻 mask/decoder/pred/head 相关参数
        if not trainable:
            print("No trainable params detected — unfreezing mask/decoder/head related parameters as fallback.")
            for n, p in model.named_parameters():
                if any(k in n for k in ('mask', 'decoder', 'pred', 'head')):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            # print a quick summary
            tot = sum(1 for _ in model.parameters())
            train = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"Fallback: {train}/{tot} parameters set trainable.")
    ensure_some_params_trainable(model)
    freeze_backbone_but_train_adapters(model)

    # 放在 model 注入 & freeze 调用之后（即在 model = inject_adapter_to_mask_decoder(model); freeze_backbone_but_train_adapters(model) 之后）
    def print_trainable_summary(model):
        tot = 0
        train = 0
        trainable_names = []
        for n, p in model.named_parameters():
            tot += 1
            if p.requires_grad:
                train += 1
                trainable_names.append(n)
        print(f"Trainable parameters: {train}/{tot}")
        # 打印前若干 trainable name（便于调试）
        if train > 0:
            print("Some trainable param name samples:", trainable_names[:10])
        else:
            print("No trainable params found.")

    def ensure_some_params_trainable(model, keywords=('adapter', 'mask', 'decoder', 'head', 'pred', 'mlp')):
        """
        如果当前没有任何参数可训练，作为后备策略解冻所有 name 中包含 keywords 的参数，
        以保证有参数可训练（避免 loss 无梯度的问题）。
        返回解冻后的 trainable count。
        """
        # 先统计已有可训练参数
        has_trainable = any(p.requires_grad for p in model.parameters())
        if has_trainable:
            return sum(1 for p in model.parameters() if p.requires_grad)

        # 若无则解冻包含关键词的参数
        print("No trainable params detected — applying fallback: unfreezing parameters whose names contain keywords:",
              keywords)
        unfreeze_cnt = 0
        for n, p in model.named_parameters():
            if any(k in n.lower() for k in keywords):
                p.requires_grad = True
                unfreeze_cnt += 1
            else:
                p.requires_grad = False
        # 如果仍然没有解冻任何参数（关键词不匹配），最保守做法：解冻最后若干参数（例如模型最后 10 个）
        if unfreeze_cnt == 0:
            print(
                "Fallback keywords did not match any parameter names — unfreezing last 10 parameters as final fallback.")
            params = list(model.named_parameters())
            for n, p in params[-10:]:
                p.requires_grad = True
            unfreeze_cnt = 10
        # 打印结果
        print(f"Fallback unfroze {unfreeze_cnt} params.")
        print_trainable_summary(model)
        return sum(1 for p in model.parameters() if p.requires_grad)

    # 调用示例（把这三行放在注入 & freeze 调用后）
    # model = inject_adapter_to_mask_decoder(model)
    # freeze_backbone_but_train_adapters(model)
    ensure_some_params_trainable(model)
    print_trainable_summary(model)

    # train
    train_loop(model, train_loader, val_loader, args)

    # final evaluation on val set and print metrics
    print('Final evaluation on validation set...')
    results_dir = os.path.join(args.save_dir, 'results')
    safe_makedirs(results_dir)
    val_iou, val_dice, val_acc = evaluate(model, val_loader, args.device, results_dir=results_dir)
    print(f'FINAL VAL IoU: {val_iou:.4f}  Dice: {val_dice:.4f}  PixelAcc: {val_acc:.4f}')
    print(f'Per-image results saved to {os.path.join(results_dir, "eval_per_image.csv")}')
if __name__ == '__main__':
    main()
# (完整脚本结束)
