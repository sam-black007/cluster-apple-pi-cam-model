"""
download_datasets.py
=====================
Download public drone + bird detection datasets used to build the 2-class
(drone, bird) YOLO model.

Sources are configured in DATA_SOURCES below. Secrets are read from a local
.env file (KAGGLE_API_TOKEN, ROBOFLOW_API_KEY) which is gitignored.

Kaggle auth: the new-style API token (KGAT_...) is written to
~/.kaggle/access_token and exported as KAGGLE_API_TOKEN.

Usage:
    python scripts/download_datasets.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
KAGGLE_RAW = RAW / "kaggle"
ROBOFLOW_RAW = RAW / "roboflow"

# ---------------------------------------------------------------------------
# Data sources
#   kind : "kaggle" | "roboflow"
#   slug : Kaggle dataset slug "owner/name"  (kaggle)
#   rf   : dict(workspace, project, version) (roboflow)
#   name : local folder name under data/raw/<kind>/<name>
#   klass: "drone" or "bird"  (consumed by prepare_data.py)
#   fmt  : "yolo" | "voc"  label format  (consumed by prepare_data.py)
# ---------------------------------------------------------------------------
DATA_SOURCES = [
    {
        "kind": "kaggle",
        "slug": "troykueh/multi-class-drone-detection-dataset-yolov8-ready",
        "name": "drone_multiclass_yolov8",
        "klass": "drone",
        "fmt": "yolo",
    },
    {
        "kind": "kaggle",
        "slug": "chunggr/fbd-sv-2024",
        "name": "bird_flying_fbd",
        "klass": "bird",
        "fmt": "voc",
    },
    # Extra drone data (small, YOLO) - adds more drone variety
    {
        "kind": "kaggle",
        "slug": "muki2003/yolo-drone-detection-dataset",
        "name": "drone_yolo_extra",
        "klass": "drone",
        "fmt": "yolo",
    },
    # Optional extras (uncomment to include)
    # Large flying-bird set (12.5 GB) - enable on a machine with stable
    # bandwidth; labels are VOC/XML (fmt="voc").
    # {
    #     "kind": "kaggle",
    #     "slug": "chunggr/fbd-sv-2024",
    #     "name": "bird_fbd_sv_2024",
    #     "klass": "bird",
    #     "fmt": "voc",
    # },
    # Bird diversity (VOC/XML, 200 species)
    # {
    #     "kind": "kaggle",
    #     "slug": "sovitrath/cub-200-bird-species-xml-detection-dataset",
    #     "name": "bird_cub200",
    #     "klass": "bird",
    #     "fmt": "voc",
    # },
]


def setup_kaggle_auth(token: str) -> None:
    os.environ["KAGGLE_API_TOKEN"] = token
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    access_token = kaggle_dir / "access_token"
    access_token.write_text(token.strip())
    try:
        os.chmod(access_token, 0o600)
    except Exception:
        pass


def download_kaggle(slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[KAGGLE] downloading {slug} -> {out_dir}", flush=True)
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", slug, "-p", str(out_dir), "--unzip", "--force",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # fall back to the kaggle CLI entrypoint name
        cmd[1] = "kaggle"
        subprocess.run(cmd, check=True)
    # remove leftover archives so the sentinel below is the only marker
    for ext in ("*.zip", "*.zip.kaggle-partial"):
        for leftover in out_dir.glob(ext):
            try:
                leftover.unlink()
            except OSError:
                pass
    (out_dir / ".done").write_text(slug)


def download_roboflow(workspace: str, project: str, version: int,
                      api_key: str, out_dir: Path) -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ROBOFLOW] roboflow package not installed; skipping", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ROBOFLOW] downloading {workspace}/{project}/{version} -> {out_dir}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    proj.version(version).download("yolov8", location=str(out_dir))


def main() -> None:
    load_dotenv(ROOT / ".env")

    kaggle_token = os.getenv("KAGGLE_API_TOKEN")
    roboflow_key = os.getenv("ROBOFLOW_API_KEY")

    if not kaggle_token:
        print("ERROR: KAGGLE_API_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    setup_kaggle_auth(kaggle_token)

    KAGGLE_RAW.mkdir(parents=True, exist_ok=True)
    ROBOFLOW_RAW.mkdir(parents=True, exist_ok=True)

    for src in DATA_SOURCES:
        try:
            if src["kind"] == "kaggle":
                target = KAGGLE_RAW / src["name"]
                if (target / ".done").exists():
                    print(f"[SKIP] {src['name']} already downloaded")
                    continue
                download_kaggle(src["slug"], target)
            elif src["kind"] == "roboflow":
                if not roboflow_key:
                    print("[SKIP] ROBOFLOW_API_KEY missing; skipping roboflow source")
                    continue
                target = ROBOFLOW_RAW / src["name"]
                if target.exists() and any(target.iterdir()):
                    print(f"[SKIP] {src['name']} already present")
                    continue
                rf = src.get("rf", {})
                download_roboflow(
                    rf["workspace"], rf["project"], int(rf["version"]),
                    roboflow_key, target,
                )
            else:
                print(f"[WARN] unknown kind {src['kind']}; skipping")
        except Exception as e:
            print(f"[WARN] source {src['name']} failed: {e}")

    print("\nDownload complete. Raw data under:", RAW)


if __name__ == "__main__":
    main()
