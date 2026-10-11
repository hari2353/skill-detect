from skill_project.training import LoRAConfig, smoke_check


def test_smoke_requires_4bit_and_valid_shape(tmp_path):
    cfg = LoRAConfig()
    out = smoke_check(cfg)
    assert out.startswith("smoke-ok") and "r=16" in out


def test_from_yaml_respects_file(tmp_path):
    p = tmp_path / "lora.yaml"
    p.write_text("base_model: foo/bar\nr: 8\nload_in_4bit: true\n")
    cfg = LoRAConfig.from_yaml(str(p))
    assert cfg.base_model == "foo/bar" and cfg.r == 8
