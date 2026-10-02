from aex.common.manifest import build_manifest, canonical_json, manifest_hash


def test_hash_ignores_time_and_host():
    a = build_manifest({"arms": ["oracle"]}, models={"mock": "r1"})
    b = dict(a, created_at="1999-01-01T00:00:00Z", hostname="elsewhere")
    assert manifest_hash(a) == manifest_hash(b)
    assert manifest_hash(a).startswith("sha256:")


def test_hash_changes_with_config_or_model_revision():
    base = build_manifest({"arms": ["oracle"]}, models={"mock": "r1"})
    assert manifest_hash(base) != manifest_hash(build_manifest({"arms": ["x"]}, models={"mock": "r1"}))
    assert manifest_hash(base) != manifest_hash(build_manifest({"arms": ["oracle"]}, models={"mock": "r2"}))


def test_config_hash_is_key_order_independent():
    a = build_manifest({"a": 1, "b": 2}, models={})
    b = build_manifest({"b": 2, "a": 1}, models={})
    assert a["config_hash"] == b["config_hash"]


def test_git_fields_present_even_outside_a_repo(tmp_path):
    m = build_manifest({}, models={}, repo_dir=tmp_path)
    assert m["git_sha"] == "unknown" and m["git_dirty"] is False


def test_canonical_json_is_compact_and_sorted():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
