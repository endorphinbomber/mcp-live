from tonematch.config import ParamSpec, load_config
from tonematch.live.params import LiveParam, expand_pattern, resolve, resolve_all, split_alternatives


def lp(i, name, q=False, items=(), lo=0.0, hi=1.0):
    return LiveParam(i, name, lo, hi, lo, q, list(items))


PARAMS = [
    lp(0, "Amp Type", True, ["Clean", "Rust", "Hot"], 0, 2),
    lp(1, "Clean Gain"), lp(2, "Rust Gain"), lp(3, "Hot Gain"),
    lp(4, "Rust Treble"), lp(5, "Gate Threshold"), lp(6, "Input Gain"),
]


def test_split_respects_groups():
    assert split_alternatives("amp ?(type|select)|model") == ["amp ?(type|select)", "model"]


def test_placeholder_alternatives_dropped_until_chosen():
    assert expand_pattern("{amp}.*gain|amp.*gain", {}) == ["amp.*gain"]
    assert expand_pattern("{amp}.*gain|amp.*gain", {"amp": "Rust"}) == ["Rust.*gain", "amp.*gain"]


def test_resolve_uses_choice():
    spec = ParamSpec("gain", "{amp}.*gain|amp.*gain")
    assert resolve(spec, PARAMS) is None            # "Amp Type" is not a gain; Input Gain must not match
    assert resolve(spec, PARAMS, {"amp": "Hot"}).name == "Hot Gain"


def test_categorical_requires_quantized():
    spec = ParamSpec("amp", "amp ?(type|select)", kind="categorical")
    p = resolve(spec, PARAMS)
    assert p.name == "Amp Type"
    assert p.choices() == [(0, "Clean"), (1, "Rust"), (2, "Hot")]


def test_resolve_all_no_double_assignment():
    specs = [ParamSpec("a", "gain"), ParamSpec("b", "gain")]
    resolved, missing = resolve_all(specs, PARAMS)
    assert len({r.param.index for r in resolved}) == len(resolved)


def test_native_mapping():
    p = LiveParam(0, "x", -10.0, 10.0, 0.0, False, [])
    assert p.native(0.25) == -5.0 and p.normalized(-5.0) == 0.25


def test_default_config_loads():
    cfg = load_config()
    assert [t.name for t in cfg.tracks] == ["Drums", "Bass", "Gtr L", "Gtr R"]
    assert cfg.track("Gtr R").double_of == "Gtr L"
    assert cfg.plugin("archetype_gojira").params
    assert cfg.plugin("EQ Eight").is_stock
