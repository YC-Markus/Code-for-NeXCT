import json
from pathlib import Path

import tempfile
import unittest
import torch

from herkry.data import ManifestDataset
from herkry.metrics import metric_images
from herkry.training import image_only_loss, pre_cgls_image_loss


class Model:
    output_resolutions = [32]*3 + [64]*3 + [128]*3 + [256]*3


def check_supervision_formula_and_gradients():
    model = Model()
    target = torch.zeros(1, 1, 2, 2)
    visual = [torch.ones_like(target, requires_grad=True) for _ in range(12)]
    initial = [torch.ones_like(target, requires_grad=True) for _ in range(11)]
    aux = [{"cgls_initial_image_for_loss": x} for x in initial] + [{}]
    d = image_only_loss(model, visual, target, 2)
    a = pre_cgls_image_loss(model, aux, target, 2, (1,.5,.25,0))
    assert abs(d.item()-1) < 1e-6
    assert abs(a.item()-1.125/5.625) < 1e-6
    (d+a).backward()
    assert all(x.grad is not None for x in initial[:9])
    assert all(x.grad is None for x in initial[9:])


def check_ldct_weights_same_normalization():
    model = Model()
    model.output_resolutions = [46]*3+[92]*3+[184]*3+[368]*3
    target = torch.zeros(1,1,2,2)
    assert abs(image_only_loss(model, [target+1]*12, target, 2).item()-1) < 1e-6
    aux = [{"cgls_initial_image_for_loss": target+1} for _ in range(11)] + [{}]
    value = pre_cgls_image_loss(model, aux, target, 2, (1,.5,.25,0))
    assert abs(value.item()-.2) < 1e-6


def check_split_leakage_rejected(tmp_path):
    path = tmp_path/"manifest.json"
    row = dict(patient="case_a", path="placeholder.npy")
    path.write_text(json.dumps(dict(train=[row], val=[row])))
    try:
        ManifestDataset(path, "train")
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("Expected patient leakage rejection")


def check_center_crop_and_clipping():
    a = torch.full((1,1,368,368), 2.)
    p, t = metric_images(a, -a, 256)
    assert p.shape == (1,1,256,256)
    assert p.min() == 1 and t.max() == 0


def check_config(name):
    config = json.loads((Path(__file__).parents[1]/"configs"/f"{name}.json").read_text())
    assert abs(sum(config["view_probabilities"])-1) < 1e-6
    assert config["stage_resolutions"][-1] == config["image_size"]
    assert config["eval_depths"] == [2,4,6,8]
    assert config["pre_cgls_stage_ratios"] == [1,.5,.25,0]


class Contracts(unittest.TestCase):
    def test_supervision(self):
        check_supervision_formula_and_gradients()

    def test_ldct_supervision(self):
        check_ldct_weights_same_normalization()

    def test_patient_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            check_split_leakage_rejected(Path(directory))

    def test_metric_crop(self):
        check_center_crop_and_clipping()

    def test_aapm_config(self):
        check_config("aapm")

    def test_msd_config(self):
        check_config("msd")

    def test_ldct_config(self):
        check_config("ldct")
