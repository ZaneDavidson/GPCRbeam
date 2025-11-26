from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True, slots=True)
class Receptor:
    #
    # Normalized receptor record from GPCRdb API
    #
    entry_name: str
    name: str
    accession: str
    receptor_class: str
    receptor_family: str
    ligand_type: str
    subfamily: str
    endogenous_ligands: list[str]
    species: str
    sequence: str
    # keep the original unformatted API payload for debugging
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    @property
    def name_text(self) -> str:
        # HTML stripper
        return (
            self.name.replace("<sub>", "")
            .replace("</sub>", "")
            .replace("<sup>", "")
            .replace("</sup>", "")
        )

    @classmethod
    def from_api(cls, d: Mapping[str, Any]) -> Receptor:
        #
        # Create an instance of a Receptor from API response
        #
        return cls(
            entry_name=str(d.get("entry_name", "")),
            name=str(d.get("name", "")),
            accession=str(d.get("accession", "")),
            receptor_class=str(d.get("receptor_class", "")),
            receptor_family=str(d.get("receptor_family", "")),
            ligand_type=str(d.get("ligand_type", "")),
            subfamily=str(d.get("subfamily", "")),
            endogenous_ligands=list(map(str, d.get("endogenous_ligands", []))),
            species=str(d.get("species", "")),
            sequence=str(d.get("sequence", "")),
            raw=d,
        )


@dataclass(frozen=True)
class GPCRDBConfig:
    #
    # Configuration for the GPCRdb API client
    #
    base_url: str = "https://gpcrdb.org/services"
    timeout: float = 30.0
    max_retries: int = 5


class GPCRDBClient:
    #
    # Client instance for interacting with the GPCRdb REST API
    #
    def __init__(self, config: GPCRDBConfig | None = None) -> None:
        self.config = config or GPCRDBConfig()
        self.session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=2.0,  # exponential backoff, consider a maximum and jitter if db is difficult  # noqa: E501
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, url_endpoint: str, params: dict | None = None) -> requests.Response:
        url = f"{self.config.base_url.rstrip('/')}/{url_endpoint.lstrip('/')}"
        response = self.session.get(url, params=params or {}, timeout=self.config.timeout)
        response.raise_for_status()
        return response

    def get_receptorlist(self) -> list[Receptor]:
        #
        # Fetch all receptors from /services/receptorlist/ and return the dataclass objects
        #
        resp = self._get("receptorlist/")
        raw = cast(list[Mapping[str, Any]], resp.json())
        return [Receptor.from_api(d) for d in raw]

    def get_receptor_payloads(self) -> list[Mapping[str, Any]]:
        #
        # Alternative method for raw JSON payloads
        #
        resp = self._get("receptorlist/")
        return cast(list[Mapping[str, Any]], resp.json())

    def iter_metadata(self, receptors: Sequence[Any]) -> Iterator[dict]:
        for r in receptors:
            yield {
                "accession": r.accession or "",
                "entry_name": r.entry_name or "",
                "name": r.name_text or "",
                "receptor_class": r.receptor_class or "",
                "receptor_family": r.receptor_family or "",
                "ligand_type": r.ligand_type or "",
                "subfamily": r.subfamily or "",
                "endogenous_ligands": (
                    ";".join(r.endogenous_ligands) if r.endogenous_ligands else ""
                ),
                "species": r.species or "",
            }
