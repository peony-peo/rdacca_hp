import rdacca_hp


def test_public_api_exports():
    expected_names = [
        "rdacca_hp",
        "RdaccaHpResult",
        "calculate_rda",
        "calculate_cca",
        "calculate_dbrda",
        "create_test_data",
        "create_cca_test_data",
        "create_distance_test_data",
        "permu_hp",
        "plot_rdaccahp",
        "plot_comparison",
    ]

    for name in expected_names:
        assert hasattr(rdacca_hp, name), f"Missing public API export: {name}"


def test_public_api_all_contains_core_exports():
    exported = set(rdacca_hp.__all__)
    assert "rdacca_hp" in exported
    assert "permu_hp" in exported
    assert "plot_rdaccahp" in exported
    assert "plot_comparison" in exported


def test_package_metadata_exists():
    assert isinstance(rdacca_hp.__version__, str)
    assert isinstance(rdacca_hp.__author__, str)
    assert isinstance(rdacca_hp.__email__, str)