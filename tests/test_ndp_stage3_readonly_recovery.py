from pathlib import Path


def test_stage3_protocol_initialized_before_choose_groups_and_no_step():
    source = (Path(__file__).resolve().parents[1] / "scripts/ndp_real_trajectory_activation.py").read_text(encoding="utf-8")
    assert "runtime_protocol = v2.install_protocol" in source
    assert "choose_groups(dataset, split, args.seed, graph_path, run_dir, runtime_protocol)" in source
    assert "optimizer.step(" not in source
    choose = source.split("def choose_groups", 1)[1].split("def encode", 1)[0]
    assert "v2.install_protocol" not in choose
    assert choose.count('protocol["restore"][2]') == 1
