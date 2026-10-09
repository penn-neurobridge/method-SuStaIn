"""Embed the sweep results into the interactive HTML page (single offline file).

    uv run python codes/build_explorer.py --sweep output/noise_sweep/sweep.json --output output/sustain_explorer.html
"""
from pathlib import Path
from typing import Annotated

import typer

TEMPLATE = Path(__file__).with_name("explorer_template.html")


def main(
    sweep: Annotated[Path, typer.Option(help="sweep.json written by noise_sweep.py")] = Path("output/noise_sweep/sweep.json"),
    output: Annotated[Path, typer.Option()] = Path("output/sustain_explorer.html"),
) -> None:
    """Write the interactive explorer with the real pySuStaIn results embedded."""
    html = TEMPLATE.read_text().replace("/*__SWEEP__*/null", sweep.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    typer.echo(f"wrote {output} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    typer.run(main)
