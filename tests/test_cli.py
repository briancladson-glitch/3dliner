from linerscan.cli import main


def test_cli_plan(capsys):
    main(["plan", "--bore", "980", "--stroke", "2660", "--conrod", "2885"])
    out = capsys.readouterr().out
    assert "stop_angle_btdc" in out and "sweep_rate_dps" in out
