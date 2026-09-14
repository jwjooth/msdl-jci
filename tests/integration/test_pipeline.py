"""Integration tests for msdl_jci main pipeline."""

from msdl_jci.main import main


def test_main_pipeline_execution():
    """Verify end-to-end execution of main entrypoint returns 0 exit code."""
    exit_code = main()
    assert exit_code == 0
