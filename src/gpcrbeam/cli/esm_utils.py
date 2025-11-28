from pathlib import Path
from typing import Annotated

import typer

from gpcrbeam.config import PROCESSED_DATA_DIR
from gpcrbeam.model.batch_embed_ESM2 import ESM2Embedder, ESM2EmbeddingConfig

app = typer.Typer(help="GPCRbeam embeddings utilities", no_args_is_help=True)


@app.command("embed")
def with_esm2_embed(
    fasta_file: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Set input FASTA file location"),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            "-O",
            help="Change output directory",
        ),
    ] = PROCESSED_DATA_DIR,
    model_id: Annotated[
        str,
        typer.Option(
            "--model",
            "-M",
            help=(
                "Specify the HuggingFace module name for the ESM2 model "
                "(default: facebook/esm2_t33_650M_UR50D)"
            ),
        ),
    ] = "facebook/esm2_t33_650M_UR50D",
    layer: Annotated[
        int,
        typer.Option(
            "--layer",
            help="Specify pooling layer (default: -1 for last hidden state)",
        ),
    ] = -1,
    mixed_precision: Annotated[
        bool,
        typer.Option(
            "--AMP/--no-AMP",
            help="Use mixed precision for CUDA/MPS if available (default: enabled)",
        ),
    ] = True,
) -> None:
    """
    Extract mean-pooled sequence embeddings with ESM2 to one .pt per FASTA entry
    """
    if out_dir is None:
        out_dir = Path(PROCESSED_DATA_DIR)

    cfg = ESM2EmbeddingConfig(
        model_id=model_id,
        layer=layer,
        mixed_precision=mixed_precision,
    )
    embedder = ESM2Embedder(cfg)
    embedder.extract_to_files(
        fasta_file=fasta_file,
        out_dir=out_dir,
    )
