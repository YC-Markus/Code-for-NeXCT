"""Expand explicit patient-to-array assignments into portable slice manifests."""
import argparse
import json
from pathlib import Path
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assignments", required=True,
                   help='JSON: {"train": [{"patient": "id", "path": "volume.npy"}], ...}')
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import numpy as np
    src = Path(args.assignments).resolve()
    dst = Path(args.output).resolve()
    if dst.exists():
        raise FileExistsError(dst)
    assignments = json.loads(src.read_text())
    patients, output = {}, {}
    for split, rows in assignments.items():
        if split not in ("train","val","test","ood"):
            raise ValueError(f"Unknown split: {split}")
        output[split] = []
        for row in rows:
            patient = row["patient"]
            if patient in patients and patients[patient] != split:
                raise ValueError(f"Patient appears in multiple splits: {patient}")
            patients[patient] = split
            path = (src.parent/row["path"]).resolve()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            base = dict(patient=patient,path=os.path.relpath(path,dst.parent))
            if array.ndim == 2:
                output[split].append(base)
            elif array.ndim == 3:
                output[split].extend({**base,"slice":z} for z in range(len(array)))
            else:
                raise ValueError(f"Expected 2D slice/sinogram or [Z,H,W] volume: {path}")
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open("x") as f:
        json.dump(output,f,indent=2)
    print({k:len(v) for k,v in output.items()})


if __name__ == "__main__":
    main()
