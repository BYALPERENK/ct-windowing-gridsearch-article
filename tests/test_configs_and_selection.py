import pandas as pd

from ctwin.configs import EXPERIMENT_A_CONFIGS, GRID_CONFIGS
from ctwin.constants import SEEDS, WC_VALUES, WW_VALUES
from ctwin.slice_selection import above_median_top_k


def test_grid_has_352_windows_and_five_seeds():
    assert len(WW_VALUES) * len(WC_VALUES) == 352
    assert sorted(SEEDS) == [7, 42, 123, 999, 2026]


def test_grid_configs_match_paper_table_2():
    # (batch size, epoch ceiling, learning rate, scheduler patience, early-stop patience, clipping)
    expected = {
        "ischemic": {"resnet50": (340, 45), "densenet121": (256, 45),
                     "efficientnet_b0": (384, 70), "mobilevitv2_175": (128, 35)},
        "hemorrhagic": {"resnet50": (340, 17), "densenet121": (256, 15),
                        "efficientnet_b0": (256, 35), "mobilevitv2_175": (128, 15)},
    }
    for task, archs in expected.items():
        for arch, (bs, ep) in archs.items():
            c = GRID_CONFIGS[task][arch]
            assert (c.batch_size, c.max_epochs) == (bs, ep)
            if task == "ischemic":
                assert (c.learning_rate, c.lr_patience, c.early_stop_patience, c.grad_clip_norm) == (3e-5, 5, 13, 1.0)
            else:
                assert c.learning_rate == 1e-4 and c.grad_clip_norm is None
    pat = {a: (c.lr_patience, c.early_stop_patience) for a, c in GRID_CONFIGS["hemorrhagic"].items()}
    assert pat == {"resnet50": (3, 5), "densenet121": (3, 5), "efficientnet_b0": (5, 11), "mobilevitv2_175": (3, 7)}


def test_experiment_a_configs():
    for arch, c in EXPERIMENT_A_CONFIGS.items():
        assert (c.learning_rate, c.lr_patience, c.early_stop_patience, c.grad_clip_norm) == (1e-4, 4, 7, None)
        assert c.max_epochs == (35 if arch == "efficientnet_b0" else 20)


def test_above_median_top_k():
    df = pd.DataFrame({"pid": ["a"] * 7 + ["b"] * 3,
                       "cov": [1, 2, 3, 4, 5, 6, 7, 0.5, 8, 9]})
    sel, median = above_median_top_k(df, "cov", "pid", sort_by="cov", ascending=False, k=2)
    assert median == 4.5
    assert sorted(sel["cov"]) == [6, 7, 8, 9]
