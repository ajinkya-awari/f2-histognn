import numpy as np
import pytest

from data.labels import GroupExtractionError, LabelParseError, extract_group_id, parse_label
from data.manifest import (
    ManifestValidationError,
    validate_manifest,
    verify_artifact_bytes,
)
from data.schema import SchemaValidationError, build_node_features, validate_graph_record
from data.split import GroupOverlapError, assert_group_disjoint, partition_groups


def valid_manifest():
    return {
        "source": "synthetic://histognn-fixture",
        "license": "CC-BY-4.0",
        "checksum": "sha256:" + "a" * 64,
        "schema_version": "1.0",
        "label_mapping": {"LUAD": 0, "LUSC": 1},
        "grouping_field": "patient_id",
        "rights": {
            "authorized": True,
            "usage": "synthetic-contract-testing",
            "license_url": "synthetic://license",
        },
        "provenance": {
            "artifact_id": "synthetic-fixture-v1",
            "source_uri": "synthetic://histognn-fixture",
        },
    }


def test_manifest_requires_provenance_schema_labels_and_grouping():
    validated = validate_manifest(valid_manifest())

    assert validated.source == "synthetic://histognn-fixture"
    assert validated.schema_version == "1.0"
    assert validated.label_mapping == {"LUAD": 0, "LUSC": 1}

    for field in (
        "source",
        "license",
        "checksum",
        "schema_version",
        "label_mapping",
        "grouping_field",
        "rights",
        "provenance",
    ):
        invalid = valid_manifest()
        invalid.pop(field)
        with pytest.raises(ManifestValidationError, match=field):
            validate_manifest(invalid)


def test_manifest_rejects_blank_required_metadata():
    invalid = valid_manifest()
    invalid["source"] = "  "

    with pytest.raises(ManifestValidationError, match="source"):
        validate_manifest(invalid)


def test_manifest_requires_authorized_rights_metadata():
    invalid = valid_manifest()
    invalid["rights"] = {
        "authorized": False,
        "usage": "synthetic-contract-testing",
        "license_url": "synthetic://license",
    }

    with pytest.raises(ManifestValidationError, match="rights.authorized"):
        validate_manifest(invalid)

    invalid = valid_manifest()
    invalid["rights"] = {
        "authorized": True,
        "usage": " ",
        "license_url": "synthetic://license",
    }

    with pytest.raises(ManifestValidationError, match="rights.usage"):
        validate_manifest(invalid)


@pytest.mark.parametrize(
    "label_mapping",
    [
        {"LUAD": 0},
        {"LUAD": 0, "LUSC": 0},
        {"LUAD": "0", "LUSC": 1},
        {"OTHER": 0, "LUSC": 1},
    ],
)
def test_manifest_rejects_non_bijective_luad_lusc_label_mappings(label_mapping):
    invalid = valid_manifest()
    invalid["label_mapping"] = label_mapping

    with pytest.raises(ManifestValidationError, match="label_mapping"):
        validate_manifest(invalid)


def test_manifest_checksum_format_and_byte_verification_are_explicit(tmp_path):
    manifest = valid_manifest()
    manifest["checksum"] = "sha256:BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
    artifact = tmp_path / "synthetic-artifact.bin"
    artifact.write_bytes(b"abc")

    validated = validate_manifest(manifest)
    verification = verify_artifact_bytes(artifact, validated.checksum)

    assert validated.checksum == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verification.checksum == validated.checksum
    assert verification.byte_size == 3
    assert verification.artifact_path == artifact

    with pytest.raises(ManifestValidationError, match="checksum mismatch"):
        verify_artifact_bytes(artifact, "sha256:" + "0" * 64)


def test_manifest_rejects_malformed_checksum_and_empty_artifact(tmp_path):
    invalid = valid_manifest()
    invalid["checksum"] = "md5:" + "a" * 32

    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_manifest(invalid)

    empty_artifact = tmp_path / "empty.bin"
    empty_artifact.write_bytes(b"")

    with pytest.raises(ManifestValidationError, match="non-empty"):
        verify_artifact_bytes(empty_artifact, "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def test_schema_builds_exactly_seven_features_and_preserves_probabilities():
    centroid = np.array([[10.0, 20.0], [20.0, 40.0], [15.0, 30.0]])
    type_prob = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.4, 0.5, 0.6]]
    )

    features = build_node_features(centroid, type_prob)

    assert features.shape == (3, 7)
    np.testing.assert_allclose(features[:, :2], [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(features[:, 2:], type_prob)

    record = validate_graph_record(centroid, type_prob, ["n0", "n1", "n2"])
    assert record.features.shape == (3, 7)
    assert record.identifiers == ("n0", "n1", "n2")


@pytest.mark.parametrize(
    ("centroid", "type_prob", "identifiers", "message"),
    [
        (np.empty((0, 2)), np.empty((0, 5)), [], "non-empty"),
        (np.zeros((2, 3)), np.zeros((2, 5)), ["n0", "n1"], "centroid"),
        (np.zeros((2, 2)), np.zeros((2, 4)), ["n0", "n1"], "type_prob"),
        (np.array([[0.0, np.nan]]), np.zeros((1, 5)), ["n0"], "finite"),
        (np.zeros((1, 2)), np.array([[0.0, np.inf, 0.0, 0.0, 0.0]]), ["n0"], "finite"),
        (np.zeros((1, 2)), np.array([[-0.1, 0.0, 0.0, 0.0, 0.0]]), ["n0"], "between 0 and 1"),
        (np.zeros((1, 2)), np.array([[1.1, 0.0, 0.0, 0.0, 0.0]]), ["n0"], "between 0 and 1"),
        (np.zeros((2, 2)), np.zeros((2, 5)), ["n0", "n0"], "duplicate"),
        (np.zeros((1, 2)), np.zeros((1, 5)), [""], "identifier"),
    ],
)
def test_schema_rejects_invalid_graph_records(centroid, type_prob, identifiers, message):
    with pytest.raises(SchemaValidationError, match=message):
        validate_graph_record(centroid, type_prob, identifiers)


def test_labels_accept_only_explicit_luad_lusc_mapping():
    assert parse_label("LUAD") == "LUAD"
    assert parse_label("lusc") == "LUSC"

    with pytest.raises(LabelParseError, match="ambiguous"):
        parse_label("LUAD/LUSC")
    with pytest.raises(LabelParseError, match="unknown"):
        parse_label("normal lung")


def test_group_extraction_requires_authorized_record_string_and_is_separate_from_features():
    assert extract_group_id("synthetic_patient-001_slide-02_LUAD", authorized=True) == "patient-001"
    assert extract_group_id("synthetic_case-002_slide-01_LUSC", authorized=True) == "case-002"

    with pytest.raises(GroupExtractionError, match="authorized"):
        extract_group_id("synthetic_patient-001_slide-02", authorized=False)


def test_group_partition_is_seeded_sorted_and_disjoint():
    groups = ["patient-04", "patient-01", "patient-03", "patient-02", "patient-05", "patient-06"]

    first = partition_groups(groups, seed=17, train_fraction=0.5, validation_fraction=1 / 3)
    second = partition_groups(list(reversed(groups)), seed=17, train_fraction=0.5, validation_fraction=1 / 3)

    assert first == second
    assert first.train == tuple(sorted(first.train))
    assert first.validation == tuple(sorted(first.validation))
    assert first.test == tuple(sorted(first.test))
    assert_group_disjoint(first)
    assert set(first.train + first.validation + first.test) == set(groups)


def test_group_overlap_assertion_fails_closed():
    contaminated = {
        "train": ("patient-01",),
        "validation": ("patient-01",),
        "test": ("patient-02",),
    }

    with pytest.raises(GroupOverlapError, match="patient-01"):
        assert_group_disjoint(contaminated)


def test_group_partition_rejects_fractions_exceeding_one():
    groups = ["patient-01", "patient-02", "patient-03"]

    with pytest.raises(ValueError, match="exceed"):
        partition_groups(groups, train_fraction=0.8, validation_fraction=0.3)


def test_group_disjoint_assertion_rejects_blank_partition_entries():
    invalid = {
        "train": ("patient-01",),
        "validation": (" ",),
        "test": ("patient-02",),
    }

    with pytest.raises(GroupOverlapError, match="non-blank"):
        assert_group_disjoint(invalid)
