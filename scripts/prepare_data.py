"""
prepare_data.py
===============
Multi-Source Dataset Aggregator and Balancer for DroneInterceptor.
Combines and normalizes public Drone and Bird detection datasets into
a unified, strictly balanced 2-class YOLO dataset:

    Class 0 = drone
    Class 1 = bird

Key Features:
  - Multi-dataset ingestion with source-specific label remapping
  - SHA256 image-level deduplication to eliminate train/val leakage
  - Approx. 55% drone / 45% bird class rebalancing before splitting
  - Box quality validation (drops zero-area, out-of-bounds, and noise boxes)
  - Stratified 85/15 train/val split preserving the class ratio
  - Automatic dataset.yaml configuration generation

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --balance equal --max-per-class 2500
  python scripts/prepare_data.py --balance all   # ingest everything without capping
"""

import argparse
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
RAW_KAGGLE = ROOT / "data" / "raw" / "kaggle"
PROCESSED = ROOT / "data" / "processed"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_BOX_AREA = 1.5e-4  # Drop noise boxes smaller than ~6x6px @ 640x640
SEED = 42


def link_or_copy(src: Path, dst: Path) -> None:
    """Creates a hardlink if on the same filesystem, otherwise copies."""
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    """Computes SHA256 digest of image for exact deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset Ingestion Adapters
# ---------------------------------------------------------------------------

def clean_yolo_lines(raw_lines: List[str], mapping: Dict[int, int]) -> Tuple[str, Set[int]]:
    """
    Remaps source class IDs to target (0: drone, 1: bird) and validates boxes.
    Returns: (cleaned_label_str, set_of_target_classes_present)
    """
    cleaned = []
    classes_present = set()
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        if src_cls not in mapping:
            continue
        tgt_cls = mapping[src_cls]

        # Validate box bounds and minimum scale
        if not (0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            continue
        if bw * bh < MIN_BOX_AREA:
            continue

        cleaned.append(f"{tgt_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        classes_present.add(tgt_cls)

    out_text = "\n".join(cleaned) + "\n" if cleaned else ""
    return out_text, classes_present


def ingest_sshikamaru_drone(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """sshikamaru_drone: 0 -> drone (0). JPEG images in Database1/Database1/"""
    entries = []
    search_dir = root / "Database1" / "Database1"
    if not search_dir.exists():
        search_dir = root
    if not search_dir.exists():
        return entries

    for txt in search_dir.glob("*.txt"):
        if txt.stat().st_size == 0:
            continue
        stem = txt.stem
        img = None
        for ext in (".jpeg", ".JPEG", ".jpg", ".JPG", ".png"):
            candidate = search_dir / f"{stem}{ext}"
            if candidate.exists():
                img = candidate
                break
        if not img:
            continue

        text, classes = clean_yolo_lines(txt.read_text().splitlines(), {0: 0})
        if text and classes:
            entries.append((img, text, classes, f"sshikamaru_{stem}"))
    return entries


def ingest_drone_vs_bird(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """
    drone_vs_bird_v1i:
      In Roboflow drone-vs-bird challenge:
      Source class 1 = drone -> target 0
      Source class 0 = bird  -> target 1
    """
    entries = []
    if not root.exists():
        return entries

    for split in ("train", "valid", "test"):
        lbl_dir = root / split / "labels"
        img_dir = root / split / "images"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob("*.txt"):
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = img_dir / f"{txt.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if not img:
                continue

            text, classes = clean_yolo_lines(txt.read_text().splitlines(), {1: 0, 0: 1})
            if text and classes:
                entries.append((img, text, classes, f"dvb_{txt.stem}"))
    return entries


def ingest_samuelayman_bird(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """samuelayman_bird: class 24 -> bird (1). Images in bird/, labels in bird/labels/"""
    entries = []
    base = root / "bird" if (root / "bird").exists() else root
    lbl_dir = base / "labels"
    if not lbl_dir.exists():
        return entries

    for txt in lbl_dir.glob("*.txt"):
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            c = base / f"{txt.stem}{ext}"
            if c.exists():
                img = c
                break
        if not img:
            continue

        # In this dataset, birds are labeled as class 24 (COCO bird class index)
        text, classes = clean_yolo_lines(txt.read_text().splitlines(), {24: 1, 0: 1})
        if text and classes:
            entries.append((img, text, classes, f"sayman_{txt.stem}"))
    return entries


def ingest_birds_crow_pigeon(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """birds_crow_pigeon: 0: crow, 1: pegion, 2: sparrow -> all bird (1)"""
    entries = []
    base = root / "birds-crow-pegion-sparrow" if (root / "birds-crow-pegion-sparrow").exists() else root
    if not base.exists():
        return entries

    for split in ("train", "valid", "test"):
        lbl_dir = base / split / "labels"
        img_dir = base / split / "images"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob("*.txt"):
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = img_dir / f"{txt.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if not img:
                continue

            text, classes = clean_yolo_lines(txt.read_text().splitlines(), {0: 1, 1: 1, 2: 1})
            if text and classes:
                entries.append((img, text, classes, f"crowpig_{txt.stem}"))
    return entries


def ingest_airspace_v6(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """airspace_v6: 0: multirotor (0), 1: fixed_wing (0), 2: bird (1)"""
    entries = []
    if not root.exists():
        return entries

    lbl_root = root / "labels"
    img_root = root / "images"
    for split in ("train", "val", "test", ""):
        ld = lbl_root / split if split else lbl_root
        id_dir = img_root / split if split else img_root
        if not ld.exists():
            continue
        for txt in ld.glob("*.txt"):
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = id_dir / f"{txt.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if not img:
                continue

            text, classes = clean_yolo_lines(txt.read_text().splitlines(), {0: 0, 1: 0, 2: 1})
            if text and classes:
                entries.append((img, text, classes, f"airspace_{txt.stem}"))
    return entries


def ingest_drone_synth(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """drone_synth_v2: 0: drone (0)"""
    entries = []
    if not root.exists():
        return entries

    lbl_root = root / "labels"
    img_root = root / "images"
    for split in ("train", "val", "test", ""):
        ld = lbl_root / split if split else lbl_root
        id_dir = img_root / split if split else img_root
        if not ld.exists():
            continue
        for txt in ld.glob("*.txt"):
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = id_dir / f"{txt.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if not img:
                continue

            text, classes = clean_yolo_lines(txt.read_text().splitlines(), {0: 0})
            if text and classes:
                entries.append((img, text, classes, f"synth_{txt.stem}"))
    return entries


def ingest_airport_drone(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """airport_drone: 0: drone (0) in Airport_Drone_Dataset"""
    entries = []
    base = root / "Airport_Drone_Dataset" if (root / "Airport_Drone_Dataset").exists() else root
    if not base.exists():
        return entries

    img_dir = base / "images"
    lbl_dir = base / "labels"
    if not lbl_dir.exists():
        return entries

    for txt in lbl_dir.glob("*.txt"):
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            c = img_dir / f"{txt.stem}{ext}"
            if c.exists():
                img = c
                break
        if not img:
            continue

        text, classes = clean_yolo_lines(txt.read_text().splitlines(), {0: 0})
        if text and classes:
            entries.append((img, text, classes, f"airport_{txt.stem}"))
    return entries


def ingest_bird_birdies(root: Path) -> List[Tuple[Path, str, Set[int], str]]:
    """bird_birdies: Gpiosenka Birdies 2000 pairs dataset"""
    entries = []
    if not root.exists():
        return entries

    lbl_dir = root / "labels" if (root / "labels").exists() else root
    img_dir = root / "images" if (root / "images").exists() else root

    txt_files = list(lbl_dir.glob("*.txt")) if lbl_dir != root else list(root.rglob("*.txt"))
    for txt in txt_files:
        if txt.name.lower() in ("readme.txt", "classes.txt"):
            continue
        # Look for corresponding image
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            c = img_dir / f"{txt.stem}{ext}"
            if c.exists():
                img = c
                break
        if not img:
            continue

        # In birdies, all objects are birds -> map all classes to 1
        mapping = {c: 1 for c in range(250)}
        text, classes = clean_yolo_lines(txt.read_text().splitlines(), mapping)
        if text and classes:
            entries.append((img, text, classes, f"birdies_{txt.stem}"))
    return entries


# ---------------------------------------------------------------------------
# Main Aggregator and Balancer
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-source dataset builder and 55/45 drone/bird balancer")
    parser.add_argument("--balance", choices=["equal", "all"], default="equal",
                        help="Balancing strategy: 'equal' enforces exact 50/50 class balance, 'all' retains everything")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional maximum images per class (e.g. 2500)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of dataset to reserve for validation (default: 0.15)")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 70)
    print("  DRONEINTERCEPTOR DATASET AGGREGATOR & BALANCER")
    print(f"  Target Classes: 0: drone, 1: bird | Strategy: {args.balance.upper()}")
    print("=" * 70)

    # Ingest from all available raw repositories
    sources = [
        ("sshikamaru_drone", RAW_KAGGLE / "sshikamaru_drone", ingest_sshikamaru_drone),
        ("drone_vs_bird_v1i", RAW_KAGGLE / "drone_vs_bird_v1i", ingest_drone_vs_bird),
        ("samuelayman_bird", RAW_KAGGLE / "samuelayman_bird", ingest_samuelayman_bird),
        ("birds_crow_pigeon", RAW_KAGGLE / "birds_crow_pigeon", ingest_birds_crow_pigeon),
        ("airspace_v6", RAW_KAGGLE / "airspace_v6", ingest_airspace_v6),
        ("drone_synth_v2", RAW_KAGGLE / "drone_synth_v2", ingest_drone_synth),
        ("airport_drone", RAW_KAGGLE / "airport_drone", ingest_airport_drone),
        ("bird_birdies", RAW_KAGGLE / "bird_birdies", ingest_bird_birdies),
    ]

    all_raw_entries = []
    print("\n[1/4] Ingesting raw datasets...")
    for name, path, fn in sources:
        if path.exists():
            entries = fn(path)
            print(f"  -> {name:<22}: {len(entries):>5} images parsed")
            all_raw_entries.extend(entries)
        else:
            print(f"  -> {name:<22}: [NOT PRESENT]")

    if not all_raw_entries:
        raise SystemExit("[ERROR] No labeled images found in data/raw/kaggle!")

    # Deduplicate by SHA256 image content hash
    print("\n[2/4] Deduplicating images (SHA256)...")
    seen_hashes = set()
    deduped_entries = []
    for img_path, text, classes, uniq_id in all_raw_entries:
        try:
            if img_path.stat().st_size < 128:
                continue
            h = sha256_file(img_path)
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_entries.append((img_path, text, classes, uniq_id, h))
        except Exception:
            continue
    print(f"  Retained {len(deduped_entries)} unique images from {len(all_raw_entries)} candidates.")

    # Partition into drone and bird pools
    drone_pool = []
    bird_pool = []
    mixed_pool = []

    for item in deduped_entries:
        classes = item[2]
        if 0 in classes and 1 in classes:
            mixed_pool.append(item)
        elif 0 in classes:
            drone_pool.append(item)
        elif 1 in classes:
            bird_pool.append(item)

    print(f"\n[3/4] Pool Distribution:")
    print(f"  Pure Drone Images: {len(drone_pool)}")
    print(f"  Pure Bird Images : {len(bird_pool)}")
    print(f"  Mixed Images     : {len(mixed_pool)}")

    if args.balance == "equal":
        # Rebalance to ~55% drone / 45% bird before train/val splitting so
        # birds (flying targets) are properly represented.
        DRONE_RATIO = 0.55
        BIRD_RATIO = 0.45

        random.shuffle(drone_pool)
        random.shuffle(bird_pool)

        # Determine which pool is limiting for the 55/45 split:
        # drone : bird = 55 : 45  =>  drone = bird * 55/45
        if len(drone_pool) * BIRD_RATIO >= len(bird_pool) * DRONE_RATIO:
            # birds are limiting; use all birds, scale drones up to ratio
            target_birds = len(bird_pool)
            target_drones = int(round(target_birds * DRONE_RATIO / BIRD_RATIO))
        else:
            # drones are limiting; use all drones, scale birds down to ratio
            target_drones = len(drone_pool)
            target_birds = int(round(target_drones * BIRD_RATIO / DRONE_RATIO))

        if args.max_per_class:
            target_drones = min(target_drones, args.max_per_class)
            target_birds = min(target_birds, args.max_per_class)

        target_drones = min(target_drones, len(drone_pool))
        target_birds = min(target_birds, len(bird_pool))

        selected_drones = drone_pool[:target_drones]
        selected_birds = bird_pool[:target_birds]

        final_selection = selected_drones + selected_birds
        total = len(final_selection)
        drone_pct = 100 * target_drones / total if total else 0.0
        print(f"  Balancing: Selected {target_drones} drone images and {target_birds} bird images")
        print(f"  Total Balanced Dataset Size: {total} images (~{drone_pct:.1f}% drone / {100 - drone_pct:.1f}% bird)")
    else:
        final_selection = drone_pool + bird_pool + mixed_pool
        print(f"  All Mode: Using all {len(final_selection)} unique images.")

    # Stratified Train/Val split
    val_fraction = float(args.val_fraction)
    drone_train, drone_val = [], []
    bird_train, bird_val = [], []

    for item in final_selection:
        if 0 in item[2]:
            drone_train.append(item)
        else:
            bird_train.append(item)

    random.shuffle(drone_train)
    random.shuffle(bird_train)

    n_val_drone = int(len(drone_train) * val_fraction)
    n_val_bird = int(len(bird_train) * val_fraction)

    val_set = drone_train[:n_val_drone] + bird_train[:n_val_bird]
    train_set = drone_train[n_val_drone:] + bird_train[n_val_bird:]

    random.shuffle(train_set)
    random.shuffle(val_set)

    # Clean and write to data/processed
    print("\n[4/4] Writing processed dataset to data/processed/ ...")
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)

    for split in ("train", "val"):
        (PROCESSED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(dataset, split_name):
        img_out = PROCESSED / "images" / split_name
        lbl_out = PROCESSED / "labels" / split_name
        for img_path, text, _, uniq_id, _ in dataset:
            ext = img_path.suffix.lower()
            dst_img = img_out / f"{uniq_id}{ext}"
            dst_lbl = lbl_out / f"{uniq_id}.txt"
            link_or_copy(img_path, dst_img)
            dst_lbl.write_text(text)

    write_split(train_set, "train")
    write_split(val_set, "val")

    # Write updated dataset.yaml
    dataset_yaml_path = ROOT / "data" / "dataset.yaml"
    dataset_yaml_path.write_text(
        f"path: {PROCESSED.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names: ['drone', 'bird']\n"
    )

    # Compute final audit statistics
    box_stats = Counter()
    for _, text, _, _, _ in train_set:
        for l in text.splitlines():
            if l.strip():
                box_stats[f"train_cls_{l.split()[0]}"] += 1
    for _, text, _, _, _ in val_set:
        for l in text.splitlines():
            if l.strip():
                box_stats[f"val_cls_{l.split()[0]}"] += 1

    tot_drones = box_stats["train_cls_0"] + box_stats["val_cls_0"]
    tot_birds = box_stats["train_cls_1"] + box_stats["val_cls_1"]
    tot_boxes = tot_drones + tot_birds

    print("\n" + "=" * 70)
    print("  DATASET PREPARATION & BALANCING COMPLETE")
    print("=" * 70)
    print(f"  Configuration : {dataset_yaml_path}")
    print(f"  Total Images  : {len(train_set) + len(val_set)} (Train: {len(train_set)}, Val: {len(val_set)})")
    print(f"  Images/Class  : Drone: {len(drone_train)} ({100*len(drone_train)/(len(train_set)+len(val_set)):.1f}%) | "
          f"Bird: {len(bird_train)} ({100*len(bird_train)/(len(train_set)+len(val_set)):.1f}%)")
    print(f"  Boxes/Class   : Drone: {tot_drones} ({100*tot_drones/tot_boxes:.1f}%) | "
          f"Bird: {tot_birds} ({100*tot_birds/tot_boxes:.1f}%)")
    print(f"  Train Set     : Drones: {box_stats['train_cls_0']} boxes | Birds: {box_stats['train_cls_1']} boxes")
    print(f"  Val Set       : Drones: {box_stats['val_cls_0']} boxes | Birds: {box_stats['val_cls_1']} boxes")
    print("=" * 70)


if __name__ == "__main__":
    main()
