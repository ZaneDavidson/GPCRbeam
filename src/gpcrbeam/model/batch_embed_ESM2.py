from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from Bio import SeqIO
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)


# ---------------------------
# FASTA streaming utilities
# ---------------------------
@dataclass(frozen=True)
class FastaEntry:
    id: str
    seq: str


def read_fasta(fasta_path: Path) -> Iterator[FastaEntry]:
    if not fasta_path.exists():
        raise FileNotFoundError(f"Improper config! {fasta_path} does not exist.")  # safety in CLI

    with fasta_path.open("r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            yield FastaEntry(id=record.id, seq=str(record.seq))


# ---------------------------
# Dataset utilities
# ---------------------------
class FastaDataset(Dataset):
    def __init__(self, items: Sequence[FastaEntry]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> FastaEntry:
        return self.items[idx]


# interim dataset for batching by index lists, alternatively use batch_sampler
class _BatchIndexDataset(Dataset[list[int]]):
    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> list[int]:
        return self.batches[idx]


def pack_indices(
    lengths: Sequence[int],
    tokens_per_batch: int,
    special_tokens: int = 2,
) -> list[list[int]]:
    # greedy pack sequence indices into batches not exceeding cap token count
    batches: list[list[int]] = []
    current_batch: list[int] = []
    cur_tokens = 0
    for i, L in enumerate(lengths):
        toks = L + special_tokens
        if current_batch and cur_tokens + toks > tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            cur_tokens = 0
        current_batch.append(i)
        cur_tokens += toks
    if current_batch:
        batches.append(current_batch)
    return batches


# ---------------------------
# Pooling utilities
# ---------------------------
class TokenIds(NamedTuple):
    cls_id: int | None
    eos_id: int | None
    pad_id: int | None


def _mean_pool_last(
    last_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    token_ids: TokenIds,
) -> torch.Tensor:
    # there's gotta be a built in pytorch mask for special tokens
    is_valid = attn_mask.to(torch.bool)
    if token_ids.cls_id is not None:
        is_valid &= input_ids.ne(token_ids.cls_id)  # mask CLS token
    if token_ids.eos_id is not None:
        is_valid &= input_ids.ne(token_ids.eos_id)  # mask EOS token
    if token_ids.pad_id is not None:
        is_valid &= input_ids.ne(token_ids.pad_id)  # mask PAD token

    mask_f = is_valid.unsqueeze(-1)  # [B, L, 1]
    sums = (last_hidden * mask_f).sum(1)  # [B, H]
    counts = mask_f.sum(1).clamp(min=1e-6)  # [B, 1] clamp to avoid a division by 0
    return sums / counts


# -----------------------------
# Embedding model configuration
# -----------------------------
@dataclass(frozen=True)
class ESM2EmbeddingConfig:
    model_id: str = "facebook/esm2_t33_650M_UR50D"
    seq_length: int = 1022  # ESM2 limit for residues (will just keep hard-coded for now)
    tokens_per_batch: int = 4096
    layer: int = -1  # last hidden layer (can use t33)
    mixed_precision: bool = True
    num_workers: int = 0  # leave this hard-coded unless I can remove the lambda for collation


class ESM2Embedder:
    def __init__(self, cfg: ESM2EmbeddingConfig) -> None:
        self.cfg = cfg
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.model_id,
            use_fast=True,
        )
        # widen scope to nn.Module for type hinting, use PreTrainedModel if needed
        self.model: nn.Module = AutoModel.from_pretrained(
            cfg.model_id,
            output_hidden_states=True,
        )
        self.model.eval()

        self.device_type = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.device = torch.device(self.device_type)
        self.model.to(self.device)

        # Token ids for filtering CLS/EOS/PAD during mean pooling
        self.cls_id = getattr(self.tokenizer, "cls_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None)

    @torch.inference_mode()
    def extract_to_files(
        self,
        fasta_file: Path,
        out_dir: Path,
    ) -> None:

        # Load entries and prepare batches by token budget
        entries = list(read_fasta(fasta_file))  # again force path in CLI
        dataset = FastaDataset(entries)
        lengths = [min(len(entry.seq), self.cfg.seq_length) for entry in entries]  # cap lengths
        index_batches = pack_indices(
            lengths,
            self.cfg.tokens_per_batch,
            special_tokens=2,
        )

        # dereference index dataset to collate sequences from FastaDataset
        def _collate(indices: list[int]) -> tuple[list[str], BatchEncoding]:
            ids = [dataset[i].id for i in indices]
            seqs = [dataset[i].seq[: self.cfg.seq_length] for i in indices]
            batch = self.tokenizer(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.cfg.seq_length + 2,
                return_tensors="pt",
            )
            return ids, batch  # return ids and tokenized batch

        # loader over index batches
        loader = DataLoader(
            _BatchIndexDataset(index_batches),
            batch_size=1,  # multi-element packed batch
            shuffle=False,
            num_workers=0,
            collate_fn=lambda items: _collate(items[0]),
        )

        use_amp = self.cfg.mixed_precision and self.device_type in ("cuda", "mps")

        total_batches = len(index_batches)

        # check tqdm implementation cuz total_batches might not work the way I think it does
        for batch_idx, (ids, batch) in tqdm(
            enumerate(loader),
            total=total_batches,
            desc="Embedding progress",
        ):
            # unwrap singleton batch
            cur_ids = ids[0]
            cur_batch = {k: v[0].to(self.device, non_blocking=True) for k, v in batch.items()}

            # remove or fold into tqdm?
            print(f"Processing batch {batch_idx+1} of {total_batches} (size={len(cur_ids)})")

            with torch.autocast(device_type=self.device_type, enabled=use_amp):
                out = self.model(**cur_batch)  # out: last_hidden_state, tuple of hidden_states

            # layer selection
            if self.cfg.layer == -1:
                token_reps = out.last_hidden_state  # [B, L, H]
            else:
                # safety clamp
                max_layer = len(out.hidden_states) - 1
                last_safe_layer = max(1, min(self.cfg.layer, max_layer))
                token_reps = out.hidden_states[last_safe_layer]  # [B, L, H]

            pooled = _mean_pool_last(
                last_hidden=token_reps,
                input_ids=batch["input_ids"],
                attn_mask=batch["attention_mask"],
                token_ids=TokenIds(
                    cls_id=self.cls_id,
                    eos_id=self.eos_id,
                    pad_id=self.pad_id,
                ),
            )  # [B, H] for each sequence

            if self.cfg.layer == -1:
                layer_key = -1
            else:
                layer_key = int(self.cfg.layer)

            pooled_cpu = pooled.to("cpu")
            for entry_idx, entry_id in enumerate(ids):
                out_path = out_dir / f"{entry_id}.pt"
                torch.save(
                    {
                        "entry_id": entry_id,
                        "mean_reps": {layer_key: pooled_cpu[entry_idx].clone()},
                    },
                    out_path,
                )
