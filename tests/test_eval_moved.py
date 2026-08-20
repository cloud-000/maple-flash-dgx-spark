"""Quality evals moved to the bench project."""

from maple_run.cli import main


def test_eval_subcommand_points_at_bench(capsys):
    assert main(["eval"]) == 2
    err = capsys.readouterr().err
    assert "bench project" in err
    assert "bench eval" in err
    assert "--base-url" in err


def test_eval_legacy_flags_still_point_at_bench(capsys):
    assert main(["eval", "--model", "checkpoints/maple-2bit"]) == 2
    err = capsys.readouterr().err
    assert "bench project" in err
