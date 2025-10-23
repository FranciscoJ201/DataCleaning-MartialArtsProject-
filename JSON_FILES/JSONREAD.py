import json, os, time
from collections import defaultdict
from util import distTwoPoints


def frame_number(k: str) -> int:
# turns "000123.jpg" -> 123, "123.png" -> 123, "img_123.jpg" -> 123
    base = k.split('.')[0]
    try:
        return int(base)
    except:
        for part in base.split('_')[::-1]:
            if part.isdigit():
                return int(part)
        return 0

def _resolve_selected_keys(sorted_keys, selected):
    """
    Normalize various 'selected' formats into a set of image_id keys to keep.
    - selected = None -> keep all
    - selected = (start_idx, end_idx) -> inclusive range over sorted_keys indices
    - selected = [int, ...] or {int, ...} -> indices into sorted_keys
    - selected = [str, ...] or {str, ...} -> exact image_id keys
    """
    if selected is None:
        return set(sorted_keys)

    # tuple range of indices
    if isinstance(selected, tuple) and len(selected) == 2 and all(isinstance(x, int) for x in selected):
        s, e = selected
        s = max(0, s)
        e = min(len(sorted_keys) - 1, e)
        if s > e:
            return set()
        return set(sorted_keys[s:e+1])

    # list/set of ints (indices)
    if isinstance(selected, (list, set)) and selected and all(isinstance(x, int) for x in selected):
        keep = set()
        for i in selected:
            if 0 <= i < len(sorted_keys):
                keep.add(sorted_keys[i])
        return keep

    # list/set of strings (image_ids)
    if isinstance(selected, (list, set)) and selected and all(isinstance(x, str) for x in selected):
        return set(k for k in sorted_keys if k in selected)

    # anything else -> keep none (fail-closed)
    return set()


def filecleanup(input_path,output_path,selected = None):
    with open(input_path, "r") as f:
        data = json.load(f)

    # Group entries by frame key
    framedata = defaultdict(list)
    initial_ids = set()
    for entry in data:
        key = entry["image_id"]
        framedata[key].append(entry)
        initial_ids.add(entry['idx'])

    # Sort frame keys (by numeric component)
    sorted_frame_keys = sorted(framedata.keys(), key=frame_number)
    if not sorted_frame_keys:
        raise RuntimeError("No frames found in the selected JSON.")

    # Resolve which frames to keep
    keep_keys = _resolve_selected_keys(sorted_frame_keys, selected)
    if not keep_keys:
        # If empty selection, write empty mapping for clarity (or raise if you prefer)
        pose_dict = {f"Id:{id_}": [] for id_ in sorted(initial_ids)}
        with open(output_path, "w") as f:
            json.dump(pose_dict, f)
        return

    # Build per-id list of keypoints, restricted to keep_keys (preserve chronological order)
    pose_dict = {f"Id:{id_}": [] for id_ in sorted(initial_ids)}
    for key in sorted_frame_keys:
        if key not in keep_keys:
            continue
        for entry in framedata[key]:
            idx = entry['idx']
            # Only append if we know this id (keeps original behavior of "initial ids")
            if f"Id:{idx}" in pose_dict:
                pose_dict[f"Id:{idx}"].append(entry['keypoints'])

    with open(output_path, "w") as f:
        json.dump(pose_dict, f, separators=(",", ":"))

        
def filecleanupsingle(input_path, output_dir, target_id, selected=None, prefix="kp"):
    """
    Extract only target_id's keypoints from input_path and write ONE JSON PER FRAME to output_dir.
    Each file content: {"keypoints": [x1,y1,s1, x2,y2,s2, ...]}
    
    Params:
      input_path:  path to AlphaPose-like JSON (list of entries with 'image_id','idx','keypoints')
      output_dir:  directory to write per-frame JSON files (created if needed)
      target_id:   integer track id to extract
      selected:    same semantics as your filecleanup(selected): None, (start_idx,end_idx), list of indices/keys
      prefix:      filename prefix (default "kp")
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load & group by frame
    with open(input_path, "r") as f:
        data = json.load(f)
    framedata = defaultdict(list)
    for entry in data:
        framedata[entry["image_id"]].append(entry)

    # Sort frames and resolve selection
    sorted_frame_keys = sorted(framedata.keys(), key=frame_number)
    if not sorted_frame_keys:
        raise RuntimeError("No frames found in the selected JSON.")
    keep_keys = _resolve_selected_keys(sorted_frame_keys, selected)

    # Write one file per frame where target_id appears
    written = 0
    for key in sorted_frame_keys:
        if keep_keys and key not in keep_keys:
            continue
        # find target_id in this frame
        for entry in framedata[key]:
            if entry.get("idx") == target_id:
                # unique-ish name: include frame key + monotonic timestamp
                ts = time.time_ns()
                fname = f"{prefix}_{target_id}_{key}_{ts}.json"
                out_path = os.path.join(output_dir, fname)
                with open(out_path, "w") as out_f:
                    json.dump({"keypoints": entry["keypoints"]}, out_f, separators=(",", ":"))
                written += 1
                break  # only one sample per frame for this id

    if written == 0:
        # Optional: raise to signal no samples written
        raise RuntimeError(f"No frames contained target_id={target_id} within the selected range.")
