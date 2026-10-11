from skill_project.training import LoRAConfig, mask_labels, smoke_check


def test_smoke_requires_4bit_and_valid_shape(tmp_path):
    cfg = LoRAConfig()
    out = smoke_check(cfg)
    assert out.startswith("smoke-ok") and "r=16" in out


def test_from_yaml_respects_file(tmp_path):
    p = tmp_path / "lora.yaml"
    p.write_text("base_model: foo/bar\nr: 8\nload_in_4bit: true\n")
    cfg = LoRAConfig.from_yaml(str(p))
    assert cfg.base_model == "foo/bar" and cfg.r == 8


def test_mask_labels_exact_prefix():
    labels, ok = mask_labels([1, 2, 3, 4], [1, 2])
    assert (labels, ok) == ([-100, -100, 3, 4], True)


def test_mask_labels_boundary_merge_tolerance():
    labels, ok = mask_labels([1, 2, 7, 4], [1, 2, 3])  # BPE glued token 3+7
    assert (labels, ok) == ([-100, -100, 7, 4], True)


def test_mask_labels_no_match_falls_back_openly():
    labels, ok = mask_labels([9, 9, 3, 4], [1, 2])
    assert (labels, ok) == ([9, 9, 3, 4], False)


def test_mask_labels_empty_prompt_is_not_a_mask():
    assert mask_labels([1, 2], []) == ([1, 2], False)
