"""Pytest hooks. Decode speed benches are opt-in: ``pytest --bench``."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--bench",
        action="store_true",
        default=False,
        help="run packed-model decode speed benches (loads maple-2bit)",
    )
    parser.addoption(
        "--template-file",
        action="store",
        default=None,
        help="path to a chat template jinja file to test in test_chat_template.py",
    )
    parser.addoption(
        "--template-text",
        action="store",
        default=None,
        help="raw chat template string to test in test_chat_template.py",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bench: packed-model decode speed (run with --bench)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--bench"):
        return
    skip_bench = pytest.mark.skip(reason="need --bench (sampled decode speed)")
    for item in items:
        if "bench" in item.keywords:
            item.add_marker(skip_bench)
