"""Resource lookup panel for the PNG Info tab.

An image's infotext names the checkpoint and every ``<lora:name:weight>`` it used,
but nothing tells you whether you actually have them — you find out when the
generation comes out wrong. This adds a panel above the raw metadata listing each
model and LoRA as installed or missing, with links to get the missing ones.

When the metadata carries a hash (SwarmUI records a full SHA256 per model, and
A1111 writes ``Model hash``) the missing resource is identified exactly through
Civitai's by-hash API, so you get its real name and a direct download link rather
than a guess. Without a hash it falls back to search links by name.

It wraps ``extras.run_pnginfo``, which returns the tab's HTML, so no UI surgery.
"""
import html
import os
import re
import types
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from modules import extras, script_callbacks, shared

RE_LORA = re.compile(r"<lora:([^:>]+?)(?::([^>]*))?>", re.IGNORECASE)
# "Lora hashes: "name: hash, name2: hash2"" — quoted because of the commas.
RE_LORA_HASHES = re.compile(r'Lora hashes:\s*"([^"]*)"')
RE_MODEL = re.compile(r"(?:^|,)\s*Model:\s*([^,\n]+)")
RE_MODEL_HASH = re.compile(r"(?:^|,)\s*Model hash:\s*([0-9a-fA-F]+)")

CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/"
CIVITAI_SEARCH = "https://civitai.com/search/models?sortBy=models_v9&query="
HF_SEARCH = "https://huggingface.co/models?search="

# Only these lengths mean anything to Civitai: AutoV2 (10) and full SHA256 (64).
# A1111's 12-char LoRA hashes are AddNet hashes of the tensor data and will not
# resolve there, so they go straight to a name search instead.
CIVITAI_HASH_LENGTHS = (10, 64)

_civitai_cache = {}


def _opt(name, default=True):
    return shared.opts.data.get(name, default)


# --------------------------------------------------------------------------
# local lookups
# --------------------------------------------------------------------------

def find_lora(name):
    """The installed LoRA for a prompt tag name, or None."""
    try:
        import networks
        found = (networks.available_networks.get(name)
                 or networks.available_network_aliases.get(name))
        if found is None:
            low = name.lower()
            for key, net in list(networks.available_networks.items()) + list(networks.available_network_aliases.items()):
                if key.lower() == low:
                    found = net
                    break
        if found is not None:
            return found
    except Exception:
        pass
    # The registry is the source of truth, but don't report every LoRA as missing
    # just because it wasn't importable here — check the folder too.
    return _lora_on_disk(name)


def _lora_on_disk(name):
    root = getattr(getattr(shared, "cmd_opts", None), "lora_dir", None)
    if not root or not os.path.isdir(root):
        return None
    low = name.lower()
    # followlinks, like the WebUI's own walk_files — LoRA folders are commonly a
    # symlink to a shared store outside the install.
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            stem, ext = os.path.splitext(filename)
            if ext.lower() in (".safetensors", ".ckpt", ".pt") and stem.lower() == low:
                return types.SimpleNamespace(filename=os.path.join(dirpath, filename))
    return None


def find_checkpoint(name, digest=None):
    """The installed checkpoint for a ``Model``/``Model hash`` pair, or None."""
    try:
        from modules import sd_models
        if name:
            found = sd_models.get_closet_checkpoint_match(name)
            if found:
                return found
        if digest:
            for info in sd_models.checkpoints_list.values():
                if (info.shorthash or "").lower() == digest.lower()[:10]:
                    return info
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Civitai
# --------------------------------------------------------------------------

def civitai_by_hash(digest):
    """``{name, page, download}`` for a hash, or None. Cached; never raises."""
    digest = (digest or "").strip().lower()
    if len(digest) not in CIVITAI_HASH_LENGTHS or not _opt("pnginfo_lookup_civitai"):
        return None
    if digest in _civitai_cache:
        return _civitai_cache[digest]

    result = None
    try:
        import requests  # the WebUI already depends on it, and it uses certifi's CAs
        response = requests.get(CIVITAI_BY_HASH + urllib.parse.quote(digest), timeout=5,
                                headers={"User-Agent": "sd-webui-pnginfo"})
        data = response.json() if response.ok else {}
        if isinstance(data, dict) and data.get("modelId"):
            model = data.get("model") or {}
            result = {
                "name": f"{model.get('name') or data.get('name')} — {data.get('name')}",
                "page": f"https://civitai.com/models/{data['modelId']}?modelVersionId={data.get('id')}",
                "download": data.get("downloadUrl"),
            }
    except Exception:
        result = None  # offline, rate limited, unknown hash — fall back to search

    _civitai_cache[digest] = result
    return result


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def parse_resources(geninfo):
    """``[(kind, name, hash), ...]`` for every model and LoRA named in infotext."""
    if not geninfo:
        return []
    lastline = geninfo.strip().split("\n")[-1]

    hashes = {}
    match = RE_LORA_HASHES.search(lastline)
    if match:
        for pair in match.group(1).split(","):
            key, _, value = pair.rpartition(":")
            if key.strip() and value.strip():
                hashes[key.strip().lower()] = value.strip()

    out = []
    model = RE_MODEL.search(lastline)
    if model:
        digest = RE_MODEL_HASH.search(lastline)
        out.append(("Checkpoint", model.group(1).strip(), digest.group(1) if digest else None))
    for name, _weight in RE_LORA.findall(geninfo):
        name = name.strip()
        if name and not any(name == n for _k, n, _h in out):
            out.append(("LoRA", name, hashes.get(name.lower())))
    return out


def _links(name, digest):
    known = civitai_by_hash(digest)
    if known:
        parts = [f'<a href="{html.escape(known["page"])}" target="_blank" rel="noopener">'
                 f'{html.escape(known["name"])}</a>']
        if known.get("download"):
            parts.append(f'<a href="{html.escape(known["download"])}" target="_blank" '
                         f'rel="noopener">download</a>')
        return " &middot; ".join(parts)
    query = urllib.parse.quote(name)
    links = (f'search <a href="{CIVITAI_SEARCH}{query}" target="_blank" rel="noopener">Civitai</a>'
             f' &middot; <a href="{HF_SEARCH}{query}" target="_blank" rel="noopener">HuggingFace</a>')
    if digest:  # worth showing — it identifies the exact file on either site
        links += f' &middot; <code style="opacity:.6">{html.escape(digest)}</code>'
    return links


def build_report(geninfo):
    """HTML listing every resource the image used, installed or not."""
    resources = parse_resources(geninfo)
    if not resources:
        return ""

    resolved = [(kind, name, digest,
                 find_checkpoint(name, digest) if kind == "Checkpoint" else find_lora(name))
                for kind, name, digest in resources]
    # Query Civitai for all the unknown hashes at once, so a slow or unreachable
    # API costs one timeout for the whole panel instead of one per resource.
    todo = [d for _k, _n, d, found in resolved if found is None and d and d.lower() not in _civitai_cache]
    if todo:
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(civitai_by_hash, todo[:8]))
        except Exception:
            pass

    rows = []
    missing = 0
    for kind, name, digest, found in resolved:
        if found is not None:
            status = '<span style="color:#3fb950">&#10003; installed</span>'
            detail = html.escape(os.path.basename(str(getattr(found, "filename", "") or "")))
        else:
            missing += 1
            status = '<span style="color:#f85149">&#10007; missing</span>'
            detail = _links(name, digest)
        rows.append(
            f'<tr><td style="padding:2px 10px 2px 0;opacity:.6">{kind}</td>'
            f'<td style="padding:2px 10px 2px 0"><code>{html.escape(name)}</code></td>'
            f'<td style="padding:2px 10px 2px 0;white-space:nowrap">{status}</td>'
            f'<td style="padding:2px 0">{detail}</td></tr>')

    heading = f"Resources &mdash; {len(resources) - missing}/{len(resources)} installed"
    return (f'<div style="margin-bottom:1em"><p><b>{heading}</b></p>'
            f'<table style="width:100%;font-size:.9em">{"".join(rows)}</table></div>')


# "Reload UI" re-executes extension scripts in-process; unwrap our own previous
# patch instead of stacking on it, or the panel gets appended once per reload.
_previous_run_pnginfo = getattr(extras.run_pnginfo, "_pnginfo_inner", None) or extras.run_pnginfo


def run_pnginfo(image):
    # ("", geninfo, items_html) — the first slot is an unused HTML component
    # above the raw listing, which is exactly where this report belongs.
    report, geninfo, items_html = _previous_run_pnginfo(image)
    if image is None or not _opt("pnginfo_lookup_enabled"):
        return report, geninfo, items_html
    try:
        report = (report or "") + build_report(geninfo)
    except Exception:
        pass
    return report, geninfo, items_html


def on_ui_settings():
    section = ("comfyui_pnginfo", "AI PNGinfo")
    shared.opts.add_option("pnginfo_lookup_enabled", shared.OptionInfo(
        True, "Show which models / LoRAs an image used, and whether they are installed",
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("pnginfo_lookup_civitai", shared.OptionInfo(
        True, "Identify missing models by hash through the Civitai API",
        gr.Checkbox, {"interactive": True}, section=section).info(
        "turns a missing model into its exact page and download link. Results are "
        "cached; if the API is unreachable it falls back to search links"))


run_pnginfo._pnginfo_inner = _previous_run_pnginfo
extras.run_pnginfo = run_pnginfo

script_callbacks.on_ui_settings(on_ui_settings)
