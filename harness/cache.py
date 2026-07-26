from __future__ import annotations
import hashlib
import os

import numpy as np

from typing import TYPE_CHECKING

from .config import ACTIVATIONS_DIR
from .conditions import Condition
from .scenarios import Scenario

if TYPE_CHECKING:
    from .model_io import ModelRunner


def scenario_key(scenarios: list[Scenario]) -> str:
    h = hashlib.sha1("|".join(s.scenario_id for s in scenarios).encode())
    return f"{len(scenarios)}-{h.hexdigest()[:8]}"


def cache_path(model_id: str, cond_name: str, paraphrase: int,
               quant: str | None = None, scen_key: str = "") -> str:
    tag = f"{model_id.replace('/', '--')}"
    q = quant or "native"
    suffix = f"__{scen_key}" if scen_key else ""
    return os.path.join(ACTIVATIONS_DIR,
                        f"{tag}__{cond_name}__p{paraphrase}__{q}{suffix}.npz")


def cache_condition(runner: "ModelRunner", cond: Condition, scenarios: list[Scenario],
                    paraphrase: int = 0, force: bool = False) -> str:
    path = cache_path(runner.model_id, cond.name, paraphrase,
                      getattr(runner, "quant", None), scenario_key(scenarios))
    if os.path.exists(path) and not force:
        return path
    system = cond.systems[min(paraphrase, len(cond.systems) - 1)]
    finals, precues, ids = [], [], []
    for sc in scenarios:
        user = cond.user_prefix + sc.user_message
        hs = runner.hidden_states_at_positions(system, user, precue_marker=sc.text)
        if hs["precue"] is None:
            raise RuntimeError(f"pre-cue position not found for {sc.scenario_id} "
                               f"({cond.name}) — offset mapping failed, do not proceed")
        finals.append(hs["final"].astype(np.float16))
        precues.append(hs["precue"].astype(np.float16))
        ids.append(sc.scenario_id)
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    np.savez_compressed(path, final=np.stack(finals), precue=np.stack(precues),
                        scenario_ids=np.array(ids))
    return path


def load_cache(model_id: str, cond_name: str, paraphrase: int = 0,
               quant: str | None = None, scen_key: str = "",
               path: str | None = None) -> dict:
    resolved = path or cache_path(model_id, cond_name, paraphrase, quant, scen_key)
    if not os.path.exists(resolved):
        local = os.path.join(ACTIVATIONS_DIR, os.path.basename(resolved))
        if os.path.exists(local):
            resolved = local
        else:
            raise FileNotFoundError(
                f"{resolved} not found, and no {os.path.basename(resolved)} in "
                f"{ACTIVATIONS_DIR}. cache_paths.json records absolute paths from the "
                f"machine that produced them; copy results_7b/activations/ across too.")
    d = np.load(resolved, allow_pickle=False)
    return {"final": d["final"].astype(np.float32),
            "precue": d["precue"].astype(np.float32),
            "scenario_ids": list(d["scenario_ids"])}