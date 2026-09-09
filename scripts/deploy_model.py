#!/usr/bin/env python3
"""Deploy a research run's exported model as THE sequence classifier app.py
loads by default.

  python scripts/deploy_model.py experiments/<run_dir>

Copies <run>/model.tflite -> model/sequence_classifier/sequence_classifier.tflite
and    <run>/labels.csv   -> model/sequence_classifier/sequence_classifier_label.csv
(the previous files are kept as *.bak), then re-runs the same class-count
self-check the training scripts use. Also writes deployed_from.json next to
the model so app.py --benchmark can record which experiment it ran.

Note: runs trained with data.feature_groups / data.normalize other than the
defaults need their preprocessor at inference time -- app.py applies it when
you pass the run DIRECTORY (--seq_model experiments/<run>), so those runs are
refused here unless --force is given.
"""
import argparse
import json
import shutil
from pathlib import Path

from _common import *  # noqa: F401,F403
from dataset_utils import load_labels
from sh_research.config import SEQ_LABEL_CSV, SEQ_MODEL_TFLITE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--force", action="store_true", help="deploy even if the run needs a non-identity preprocessor")
    a = p.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    pre = meta.get("preprocessor", {})
    identity = (not pre.get("feature_groups")) and pre.get("normalize", "none") == "none"
    if not identity and not a.force:
        raise SystemExit("This run uses feature selection / normalisation. app.py only applies that when given the "
                         "run directory (python app.py --seq_model <run_dir>); use --force to deploy anyway.")
    src_model, src_labels = run / "model.tflite", run / "labels.csv"
    for f in (src_model, src_labels):
        if not f.exists():
            raise SystemExit(f"missing {f}")
    labels = load_labels(str(src_labels))
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(src_model))
    out_classes = int(it.get_output_details()[0]["shape"][-1])
    in_shape = [int(v) for v in it.get_input_details()[0]["shape"]]
    if out_classes != len(labels):
        raise SystemExit(f"REFUSING: model outputs {out_classes} classes, labels.csv has {len(labels)}")
    for src, dst in ((src_model, Path(SEQ_MODEL_TFLITE)), (src_labels, Path(SEQ_LABEL_CSV))):
        if dst.exists():
            shutil.copy2(dst, dst.with_suffix(dst.suffix + ".bak"))
        shutil.copy2(src, dst)
        print(f"{src} -> {dst}")
    info = {"experiment_id": meta.get("experiment_id", run.name), "architecture": meta.get("architecture"),
            "input_shape": in_shape, "class_names": labels, "source_dir": str(run), "dataset": meta.get("dataset")}
    Path(SEQ_MODEL_TFLITE).with_name("deployed_from.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"deployed {meta.get('architecture')} ({in_shape} -> {out_classes} classes: {labels}); "
          f"run  python app.py  to use it.")


if __name__ == "__main__":
    main()
