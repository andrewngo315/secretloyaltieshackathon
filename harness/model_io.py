from __future__ import annotations
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


def resolve_device(cfg: Config) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelRunner:
    def __init__(self, cfg: Config, model_id: str | None = None, device: str | None = None,
                 quant: str | None = None, adapter: str | None = None):
        self.cfg = cfg
        self.model_id = model_id or cfg.model_id
        self.device = device or resolve_device(cfg)
        self.quant = quant
        if self.device == "cpu":
            dtype = torch.float32
        elif cfg.dtype in ("bfloat16", "bf16"):
            dtype = torch.bfloat16
        elif cfg.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = {"dtype": dtype}
        if quant in ("int4", "int8"):
            from transformers import QuantoConfig
            kwargs["quantization_config"] = QuantoConfig(weights=quant)
        elif quant is not None:
            raise ValueError(f"quant must be None, 'int4', or 'int8'; got {quant!r}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.adapter = adapter
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self._opt_ids = {o: self._option_token_ids(o)
                         for o in ("A", "B", "C", "D")}

    def chat_text(self, system: str | None, user: str,
                  assistant_prefix: str | None = None) -> str:
        msgs = ([] if system is None else [{"role": "system", "content": system}])
        msgs += [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if assistant_prefix:
            text += assistant_prefix
        return text

    def _option_token_ids(self, opt: str) -> list[int]:
        ids = []
        for s in (opt, " " + opt):
            t = self.tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
        if not ids:
            ids = [self.tok.encode(opt, add_special_tokens=False)[0]]
        return ids

    @torch.no_grad()
    def option_logprobs(self, system: str, user: str,
                        options: tuple = ("A", "B")) -> dict[str, float]:
        text = self.chat_text(system, user)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"non-finite logits from {self.model_id} on {self.device} "
                f"dtype={self.model.dtype}: refusing to return a degenerate choice")
        logp = torch.log_softmax(logits, dim=-1)
        out = {}
        for opt in options:
            out[opt] = torch.logsumexp(logp[self._opt_ids[opt]], dim=0).item()
        if not all(math.isfinite(v) for v in out.values()):
            raise RuntimeError(
                f"non-finite option logprobs from {self.model_id}: {out}")
        return out

    @torch.no_grad()
    def generate(self, system: str, user: str, max_new_tokens: int = 200,
                 temperature: float = 0.0, assistant_prefix: str | None = None) -> str:
        text = self.chat_text(system, user, assistant_prefix)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        out = self.model.generate(**enc, **kwargs)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def hidden_states_at_positions(self, system: str, user: str,
                                   precue_marker: str | None = None) -> dict:
        text = self.chat_text(system, user)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        stack = torch.stack(hs, dim=0)[:, 0]

        result = {"final": stack[:, -1].float().cpu().numpy(), "precue": None}
        if precue_marker is not None:
            end_char = text.rfind(precue_marker)
            if end_char != -1:
                end_char += len(precue_marker)
                idx = None
                for i, (s, e) in enumerate(offsets):
                    if s < end_char and e <= end_char and e > s:
                        idx = i
                if idx is not None:
                    result["precue"] = stack[:, idx].float().cpu().numpy()
        for key, arr in result.items():
            if arr is not None and not np.isfinite(arr).all():
                raise RuntimeError(
                    f"non-finite activations ({key}) from {self.model_id} on "
                    f"{self.device} dtype={self.model.dtype}")
        return result

    @torch.no_grad()
    def continuation_logprob(self, system: str, user: str, continuation: str) -> float:
        prompt = self.chat_text(system, user)
        p_ids = self.tok(prompt, return_tensors="pt")["input_ids"]
        full = self.tok(prompt + continuation, return_tensors="pt")["input_ids"]
        full_dev = full.to(self.device)
        logits = self.model(input_ids=full_dev).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        start = p_ids.shape[1]
        tgt = full[0, start:]
        token_lp = logp[torch.arange(start - 1, full.shape[1] - 1), tgt]
        return token_lp.mean().item()