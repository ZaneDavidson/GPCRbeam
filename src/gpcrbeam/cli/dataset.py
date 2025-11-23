from pathlib import Path
from typing import Annotated

import typer

from gpcrbeam.config import ALIASES, GPCR_CLASSES, RAW_DATA_DIR
from gpcrbeam.data.loaders.gpcrdb_handler import pull_gpcr_sequences

app = typer.Typer(help="Dataset creation commands", no_args_is_help=True)


def _normalize(token: str) -> str | None:
    t = token.strip()
    if not t:
        return None
    # robust alias search
    return ALIASES.get(t) or ALIASES.get(t.upper()) or ALIASES.get(t.lower())


def _valid_list() -> str:
    # E.g., "A: Class A (Rhodopsin), B1: Class B1 (Secretin), ..."
    return ", ".join(f"{k}: {v}" for k, v in GPCR_CLASSES.items())


@app.callback()
def format_class_callback(
    gpcr_classes: str | None = typer.Option(None, "--gpcr-class", hidden=True),
) -> str | None:

    if not gpcr_classes:
        return None

    # parse and normalize comma-separated input
    tokens = [t.strip() for t in gpcr_classes.split(",")]
    seen, out, invalid = set(), [], []

    for tok in tokens:
        if not tok:  # skip empty tokens from ",,"
            continue
        m = _normalize(tok)  # returns canonical class name from alias or None
        if m is None:
            invalid.append(tok)  # append to invalid classes
            continue
        if m not in seen:
            seen.add(m)
            out.append(m)

    if invalid:
        raise typer.BadParameter(
            f"Invalid GPCR class(es): {', '.join(invalid)}. Valid options are: {_valid_list()}",
        )

    return ",".join(out)


# TODO : simplify commands or add aliases
@app.command("pull-sequences")
def handle_pull_sequences(
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Set output directory",
        ),
    ] = RAW_DATA_DIR,
    gpcr_classes: Annotated[
        str | None,
        typer.Option(
            "--gpcr-classes",
            "--gc",
            help="Comma-separated list of GPCR classes to download from database (default: all)",
            callback=format_class_callback,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            "--ow",
            help="Overwrite existing files",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            "--t",
            help="HTTP timeout seconds",
        ),
    ] = 30.0,
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            "--m",
            help="HTTP retries for 429/5xx",
        ),
    ] = 5,
) -> None:

    if out_dir is None:
        out_dir = Path(RAW_DATA_DIR)

    report = pull_gpcr_sequences(
        out_dir=out_dir,
        gpcr_classes=gpcr_classes.split(",") if gpcr_classes else None,
        overwrite=overwrite,
        timeout=timeout,
        max_retries=max_retries,
    )

    typer.echo(f"Wrote {report.n_sequences} sequences to {report.fasta_path}")
    typer.echo(f"Wrote sequence metadata to {report.metadata_path}")


if __name__ == "__main__":
    app()
