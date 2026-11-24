from skill_project.footprint import (
    qlora_share,
    qlora_trainable_params,
    training_carbon,
    training_energy,
)


def test_energy_math():
    assert training_energy(2.0, 400, pue=1.0)["kwh"] == 0.8


def test_carbon_math():
    assert (
        training_carbon(2.0, 400, pue=1.0, carbon_intensity_g_per_kwh=400)["kgco2e"]
        == 0.32
    )


def test_qlora_params_math():
    out = qlora_trainable_params(16, 896, ["q", "k", "v", "o"], 24)
    assert (
        out["trainable_params"] == 16 * 2 * 896 * 4 * 24
    )  # rank*2*hidden*targets*layers
    assert out["targets"] == 4


def test_share():
    assert qlora_share(114_688, 494_032_768) == 0.000232


def test_validation():
    for args in (
        (0, 896, ["q"], 24),
        (16, 0, ["q"], 24),
        (16, 896, [], 24),
        (16, 896, ["q"], 0),
    ):
        try:
            qlora_trainable_params(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")
    for args in ((-1, 10), (1, 0)):
        try:
            qlora_share(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")
