"""Reviewed compatibility patch for the pinned HoVer-Net checkout."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


def _replace_once(path: Path, target: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(target) != 1:
        raise RuntimeError(f"HoVer-Net patch target changed: {path}")
    path.write_text(text.replace(target, replacement), encoding="utf-8")


def patch_hovernet_checkout(checkout: Path) -> str:
    """Patch exact pinned sources to retain per-pixel type softmax outputs."""

    for source in checkout.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        updated = re.sub(r"\bnp\.(int|float|bool)\b", lambda match: match.group(1), text)
        if updated != text:
            source.write_text(updated, encoding="utf-8")

    _replace_once(
        checkout / "infer" / "base.py",
        'torch.load(self.method["model_path"])["desc"]',
        'torch.load(self.method["model_path"], weights_only=False)["desc"]',
    )
    _replace_once(
        checkout / "models" / "hovernet" / "run_desc.py",
        '            type_map = F.softmax(pred_dict["tp"], dim=-1)\n'
        '            type_map = torch.argmax(type_map, dim=-1, keepdim=True)\n'
        '            type_map = type_map.type(torch.float32)\n'
        '            pred_dict["tp"] = type_map\n'
        '        pred_output = torch.cat(list(pred_dict.values()), -1)',
        '            type_probs = F.softmax(pred_dict["tp"], dim=-1)\n'
        '            type_map = torch.argmax(type_probs, dim=-1, keepdim=True)\n'
        '            type_map = type_map.type(torch.float32)\n'
        '            pred_output = torch.cat(\n'
        '                [type_map, pred_dict["np"], pred_dict["hv"], type_probs], dim=-1\n'
        '            )\n'
        '        else:\n'
        '            pred_output = torch.cat(list(pred_dict.values()), -1)',
    )
    _replace_once(
        checkout / "models" / "hovernet" / "post_proc.py",
        '        pred_type = pred_map[..., :1]\n        pred_inst = pred_map[..., 1:]',
        '        pred_type = pred_map[..., :1]\n'
        '        pred_inst = pred_map[..., 1:4]\n'
        '        pred_type_probs = pred_map[..., 4 : 4 + nr_types]',
    )
    _replace_once(
        checkout / "models" / "hovernet" / "post_proc.py",
        '            inst_info_dict[inst_id]["type_prob"] = float(type_prob)',
        '            inst_info_dict[inst_id]["type_prob"] = float(type_prob)\n'
        '            inst_probs = pred_type_probs[rmin:rmax, cmin:cmax][inst_map_crop]\n'
        '            mean_probs = np.mean(inst_probs, axis=0)\n'
        '            inst_info_dict[inst_id]["probs"] = mean_probs.tolist()',
    )
    digest = hashlib.sha256()
    for relative in ("infer/base.py", "models/hovernet/run_desc.py", "models/hovernet/post_proc.py"):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((checkout / relative).read_bytes())
    return digest.hexdigest()
