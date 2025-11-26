import typer

from .. import __version__
from . import dataset, embed

app = typer.Typer(
    help=(
        "GPCRBeam: An inference pipeline for determining the native "
        "G-protein coupling of a GPCR protein based on sequence analysis."
    ),
    no_args_is_help=True,
)
# Nest dataset creation tools inside main CLI
app.add_typer(dataset.app, name="dataset")
app.add_typer(embed.app, name="embed")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


if __name__ == "__main__":
    app()
