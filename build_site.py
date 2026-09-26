"""Build a simple static gallery site for GitHub Pages, showing egocentric RGB +
tactile GT + tactile generated for the egotouch_tactileonly and opentouch_tactileonly
test sets. Compresses/downscales every video (480x480 -> 200x200, CRF 30) to keep
the whole site small enough to push to a normal git repo.
"""
import os
import re
import glob
import subprocess
import concurrent.futures
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SITE_DIR = "/weka/scratch/jhu/abhatt40/rbehrav1/tactile_gallery_site"
VIDEOS_DIR = f"{SITE_DIR}/videos"

CONFIGS = {
    "egotouch_tactileonly": {
        "src": "/weka/scratch/jhu/abhatt40/rbehrav1/DiffSynth-fracture/infer_eval/egotouch_tactileonly/epoch-14/egotouch_test",
        "hands": ["left", "right"],
        "title": "EgoTouch (tactile-only)",
    },
    "opentouch_tactileonly": {
        "src": "/weka/scratch/jhu/abhatt40/rbehrav1/DiffSynth-fracture/infer_eval/opentouch_tactileonly/step-2280/opentouch_test",
        "hands": ["right"],
        "title": "OpenTouch (tactile-only)",
    },
}

SCALE = 200
CRF = 30


def compress(src, dst):
    if os.path.exists(dst):
        return
    subprocess.run(
        [FFMPEG, "-y", "-i", src, "-vf", f"scale={SCALE}:{SCALE}", "-c:v", "libx264",
         "-crf", str(CRF), "-preset", "veryfast", "-an", "-loglevel", "error", dst],
        check=True,
    )


def find_clips(src_dir):
    clips = {}
    for f in os.listdir(src_dir):
        m = re.match(r"^(\d{4})_(.+)_gt_video\.mp4$", f)
        if m:
            clips[m.group(1)] = m.group(2)
    return dict(sorted(clips.items()))


def main():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    tasks = []
    manifest = {}

    for key, cfg in CONFIGS.items():
        out_dir = f"{VIDEOS_DIR}/{key}"
        os.makedirs(out_dir, exist_ok=True)
        clips = find_clips(cfg["src"])
        manifest[key] = {"title": cfg["title"], "hands": cfg["hands"], "clips": []}
        for idx, name in clips.items():
            prefix = f"{cfg['src']}/{idx}_{name}"
            rgb_dst = f"{out_dir}/{idx}_rgb.mp4"
            tasks.append((f"{prefix}_gt_video.mp4", rgb_dst))
            hand_files = {"rgb": f"{idx}_rgb.mp4"}
            for hand in cfg["hands"]:
                suffix = f"_{hand}_tactile" if len(cfg["hands"]) > 1 else "_right_tactile"
                gt_dst = f"{out_dir}/{idx}_gt_{hand}.mp4"
                gen_dst = f"{out_dir}/{idx}_gen_{hand}.mp4"
                tasks.append((f"{prefix}_gt{suffix}.mp4", gt_dst))
                tasks.append((f"{prefix}_gen{suffix}.mp4", gen_dst))
                hand_files[f"gt_{hand}"] = f"{idx}_gt_{hand}.mp4"
                hand_files[f"gen_{hand}"] = f"{idx}_gen_{hand}.mp4"
            manifest[key]["clips"].append({"idx": idx, "name": name, "files": hand_files})

    print(f"compressing {len(tasks)} videos with {os.cpu_count()} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(compress, s, d) for s, d in tasks if os.path.exists(s)]
        done = 0
        for f in concurrent.futures.as_completed(futs):
            f.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(futs)} done")
    print("compression done")

    import json
    with open(f"{SITE_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(f"{SITE_DIR}/manifest.js", "w") as f:
        f.write("window.GALLERY_MANIFEST = ")
        json.dump(manifest, f)
        f.write(";\n")
    print(f"wrote manifest with {sum(len(c['clips']) for c in manifest.values())} clips")


if __name__ == "__main__":
    main()
