import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gpcrbeam.config import GPCR_CLASSES
from gpcrbeam.data import GPCRDBClient, GPCRDBConfig


@dataclass(frozen=True)
class DownloadReport:
    n_sequences: int
    fasta_path: Path
    metadata_path: Path


def pull_gpcr_sequences(
    #
    # Handle GPCR sequence and metadata responses from GPCRdb
    #
    out_dir: Path,
    gpcr_classes: list[str] | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
    max_retries: int = 5,
) -> DownloadReport:
    if not out_dir.exists():
        raise FileNotFoundError(f"Improper config! {out_dir} does not exist.")

    fasta_path = out_dir / f"{gpcr_classes or 'all_classes'}.fasta"
    metadata_path = out_dir / "metadata.csv"

    if not overwrite and (fasta_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"{out_dir} already contains outputs (use --overwrite).")

    client = GPCRDBClient(GPCRDBConfig(timeout=timeout, max_retries=max_retries))

    if not gpcr_classes:
        receptors = client.get_receptorlist()
    else:
        # map from canonical class codes to GPCRdb class names
        gpcr_classes = [GPCR_CLASSES[c] for c in gpcr_classes]
        receptors = [r for r in client.get_receptorlist() if r.receptor_class in gpcr_classes]
    make_fasta(fasta_path, receptors)
    make_metadata(client, metadata_path, receptors)

    return DownloadReport(
        n_sequences=len(receptors),
        fasta_path=fasta_path,
        metadata_path=metadata_path,
    )


def make_fasta(fasta_path: Path, receptors: Sequence[Any]) -> None:
    # use biopython to convert from response -> seq records -> fasta
    records = []
    for _i, receptor in enumerate(receptors):
        seq = receptor.sequence or "unknown"
        id = receptor.accession or ""
        name = receptor.name_text or ""
        record = SeqRecord(Seq(seq), id=id, name=name)
        records.append(record)

    with open(fasta_path, "w") as output:
        SeqIO.write(records, output, "fasta")


def make_metadata(client: GPCRDBClient, metadata_path: Path, receptors: Sequence[Any]) -> None:
    with metadata_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "accession",
                "entry_name",
                "name",
                "receptor_class",
                "receptor_family",
                "ligand_type",
                "subfamily",
                "endogenous_ligands",
                "species",
            ],
        )
        w.writeheader()
        for meta in client.iter_metadata(receptors):
            if meta["accession"]:
                w.writerow(meta)
