"""PNG Info converter for sd-webui-forge (neo).

The WebUI only understands flat A1111 "parameters" text. Everything else —
ComfyUI node graphs, SwarmUI's ``sui_image_params`` JSON, NovelAI's ``Comment``
blob, Fooocus/InvokeAI JSON — either shows nothing in the PNG Info tab or dumps
a wall of raw JSON, and "Send to txt2img" gets nothing usable.

This extension converts all of them to ordinary A1111 infotext, so the info
panel reads cleanly and every paste button works. NovelAI prompts additionally
get their V4 per-character captions restored and their ``1.5::tag::`` /
``{{tag}}`` emphasis rewritten as ``(tag:1.5)``, which is what the WebUI's
prompt parser actually understands.

It chains ``images.read_info_from_image``, and exposes the converter as
``shared.ai_pnginfo_convert`` so the stealth pnginfo extension can hand its
decoded alpha-channel payload to the same code instead of duplicating it.

It also decodes that hidden payload itself when nothing else carries metadata,
so a metadata-stripped NovelAI image still reads correctly with this extension
installed on its own.
"""
import gzip
import json
import os
import re

import gradio as gr
from PIL import Image

from modules import images, script_callbacks, shared


# ComfyUI sampler_name -> neoforge sampler label (only confident matches; the
# resolver below falls back to the live sampler map / raw name for the rest).
COMFY_SAMPLER_MAP = {
    "euler": "Euler",
    "euler_cfg_pp": "Euler",
    "euler_ancestral": "Euler a",
    "euler_ancestral_cfg_pp": "Euler a",
    "heun": "Heun",
    "heunpp2": "Heun",
    "dpm_2": "DPM2",
    "lms": "LMS",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_cfg_pp": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpmpp_3m_sde_gpu": "DPM++ 3M SDE",
    "dpmpp_2s_ancestral": "DPM++ 2s a RF",
    "lcm": "LCM",
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": "UniPC",
    "res_multistep": "Res Multistep",
    "er_sde": "ER SDE",
    "plms": "PLMS",
    "restart": "Restart",
}

# ComfyUI/SwarmUI scheduler -> neoforge scheduler name (most names match 1:1).
SCHEDULER_MAP = {
    "ddim_uniform": "ddim",
    "sgm_uniform": "sgm_uniform",
    "ays": "align_your_steps",
    "linear quadratic": "linear_quadratic",
}

# NovelAI noise_schedule -> neoforge scheduler name.
NAI_SCHEDULE_MAP = {
    "native": "normal",
    "karras": "karras",
    "exponential": "exponential",
    "polyexponential": "polyexponential",
}

# Node class_types that act as the main sampler.
SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}

# Input keys that carry prompt text, in preference order.
TEXT_KEYS = ("text", "text_g", "text_l", "populated_text", "wildcard_text", "prompt", "string", "value")

# Scalar generation parameters to harvest from the sampler subgraph.
SCALAR_KEYS = ("seed", "noise_seed", "steps", "cfg", "guidance", "sampler_name", "scheduler", "denoise")

# Metadata blobs replaced by the converted infotext — dropped from the info
# panel so it doesn't show both.
RAW_KEYS = ("prompt", "workflow", "Comment", "Description", "Title", "Software", "Source",
            "Generation time", "Generation_time", "invokeai_metadata", "sd-metadata", "Dream")

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".sft", ".gguf", ".bin")


# --------------------------------------------------------------------------
# prompt syntax
# --------------------------------------------------------------------------

# NovelAI weights a whole span: "1.5::a, b::" == A1111 "(a, b:1.5)".
RE_NAI_WEIGHT = re.compile(r"(-?\d+(?:\.\d+)?)\s*::(.*?)::", re.DOTALL)
# NovelAI V3 emphasis: each {} multiplies by 1.05, each [] divides by it.
RE_NAI_BRACES = re.compile(r"(\{+)([^{}\[\]]+)(\}+)|(\[+)([^{}\[\]]+)(\]+)")


def _num(v):
    """Render numbers without a pointless trailing .0 (7.0 -> 7)."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def nai_prompt_to_a1111(text):
    """Rewrite NovelAI emphasis as A1111 weights.

    ``1.5::gigantic breasts::`` -> ``(gigantic breasts:1.5)``
    ``{{tag}}`` -> ``(tag:1.1025)``, ``[tag]`` -> ``(tag:0.9524)``
    Commas inside a weighted span are kept, so the individual tags survive.
    """
    if not isinstance(text, str) or not text:
        return text or ""

    def weight(m):
        inner = m.group(2).strip().strip(",").strip()
        return f"({inner}:{_num(float(m.group(1)))})" if inner else ""

    for _ in range(8):  # repeat for nested spans; NovelAI rarely nests deeply
        new = RE_NAI_WEIGHT.sub(weight, text)
        if new == text:
            break
        text = new
    text = text.replace("::", " ")  # stray markers with no weight attached

    def braces(m):
        if m.group(2) is not None:
            inner, depth, factor = m.group(2), min(len(m.group(1)), len(m.group(3))), 1.05
        else:
            inner, depth, factor = m.group(5), min(len(m.group(4)), len(m.group(6))), 1 / 1.05
        return f"({inner.strip()}:{round(factor ** depth, 4)})"

    for _ in range(8):
        new = RE_NAI_BRACES.sub(braces, text)
        if new == text:
            break
        text = new
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _lora_tags(loras):
    """``[(name, weight), ...]`` -> ``<lora:name:weight>`` tags, de-duplicated."""
    out, seen = [], set()
    for name, weight in loras:
        name = _model_name(name)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            w = _num(round(float(weight), 4))
        except (TypeError, ValueError):
            w = "1"
        out.append(f"<lora:{name}:{w}>")
    return out


def _model_name(name):
    """``SDXL\\anima_v2.safetensors`` -> ``anima_v2``; plain labels pass through."""
    if not isinstance(name, str):
        return None
    name = name.strip().replace("\\", "/")
    if "/" in name and name.lower().endswith(MODEL_EXTS):
        name = name.rsplit("/", 1)[1]
    if name.lower().endswith(MODEL_EXTS):
        name = os.path.splitext(name)[0]
    return name or None


# --------------------------------------------------------------------------
# infotext assembly
# --------------------------------------------------------------------------

def _map_sampler(name):
    if not name:
        return None
    cand = COMFY_SAMPLER_MAP.get(str(name).lower())
    try:
        from modules import sd_samplers
        proper = sd_samplers.samplers_map.get(str(name).lower())
        if proper:
            return proper
        if cand and cand in sd_samplers.all_samplers_map:
            return cand
    except Exception:
        pass
    return cand or name


def _map_scheduler(name):
    if not name:
        return None
    n = str(name).lower()
    # SwarmUI appends its own parameters: "beta_1_1", "karras_0.75"
    n = SCHEDULER_MAP.get(n) or re.sub(r"[_-][\d.]+(?:[_-][\d.]+)*$", "", n)
    try:
        from modules import sd_schedulers
        sched = sd_schedulers.schedulers_map.get(n) or sd_schedulers.schedulers_map.get(str(name))
        # Custom node packs put arbitrary values in their "scheduler" widget; an
        # unknown one is noise that would only confuse the paste dropdown.
        return sched.label if sched else None
    except Exception:
        return n


def _field(fields, key, value, *, skip_empty=True):
    if value is None or (skip_empty and value == ""):
        return
    value = _num(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    if "," in value or '"' in value or "\n" in value:
        value = json.dumps(value)  # A1111 splits fields on commas
    fields.append(f"{key}: {value}")


def _infotext(positive, negative, fields):
    if not positive and not fields:
        return None
    text = positive or ""
    if negative:
        text += f"\nNegative prompt: {negative}"
    if fields:
        text += "\n" + ", ".join(fields)
    return text


def _size_fields(fields, width, height, image):
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width and height:
        _field(fields, "Size", f"{int(width)}x{int(height)}")
    elif image is not None:
        _field(fields, "Size", f"{image.width}x{image.height}")


def _load_json(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# NovelAI
# --------------------------------------------------------------------------

def _map_nai_sampler(name):
    if not name:
        return None
    try:
        from modules import sd_samplers
        proper = sd_samplers.samplers_map.get(str(name).lower())
        if proper:
            return proper
    except Exception:
        pass
    return _map_sampler(re.sub(r"^k_", "", str(name).lower()))


def _v4_caption(v4):
    """Base caption plus every per-character caption of a V4 prompt object.

    The flat ``Description``/``prompt``/``uc`` fields only ever hold the base
    caption, which is why multi-character images lose their CHAR: text.
    """
    caption = (v4 or {}).get("caption") if isinstance(v4, dict) else None
    if not isinstance(caption, dict):
        return ""
    parts = [caption.get("base_caption") or ""]
    for ch in caption.get("char_captions") or []:
        if isinstance(ch, dict):
            parts.append(ch.get("char_caption") or "")
    parts = [p.strip().strip(",").strip() for p in parts]
    return ", ".join(p for p in parts if p)


def build_novelai_infotext(data, image):
    """A1111 infotext from a NovelAI metadata dict — the same dict NovelAI puts
    in the PNG text chunks and in the stealth alpha payload
    (Title/Description/Software/Source/Comment). Returns text or None."""
    if not isinstance(data, dict):
        return None
    comment = data.get("Comment")
    if data.get("Software") != "NovelAI" and comment is None:
        return None
    c = _load_json(comment) or {}

    positive = _v4_caption(c.get("v4_prompt")) or c.get("prompt") or data.get("Description") or ""
    negative = _v4_caption(c.get("v4_negative_prompt")) or c.get("uc") or ""
    positive = nai_prompt_to_a1111(positive)
    negative = nai_prompt_to_a1111(negative)

    fields = []
    _field(fields, "Steps", c.get("steps"))
    _field(fields, "Sampler", _map_nai_sampler(c.get("sampler")))
    _field(fields, "Schedule type", _map_scheduler(NAI_SCHEDULE_MAP.get(str(c.get("noise_schedule")).lower(), c.get("noise_schedule"))))
    _field(fields, "CFG scale", c.get("scale"))
    if c.get("cfg_rescale"):
        _field(fields, "Rescale CFG", c["cfg_rescale"])
    if c.get("uncond_scale") is not None and float(c["uncond_scale"]) != 1.0:
        _field(fields, "Undesired Content Strength", c["uncond_scale"])
    _field(fields, "Seed", c.get("seed"))
    _size_fields(fields, c.get("width"), c.get("height"), image)
    if c.get("sm_dyn"):
        _field(fields, "SMEA", "DYN")
    elif c.get("sm"):
        _field(fields, "SMEA", "true")
    if c.get("dynamic_thresholding"):
        _field(fields, "Decrisper", "true")
    _field(fields, "Model", data.get("Source"))
    _field(fields, "Clip skip", 2)
    _field(fields, "ENSD", 31337)
    return _infotext(positive, negative, fields)


# --------------------------------------------------------------------------
# SwarmUI
# --------------------------------------------------------------------------

def build_swarmui_infotext(data, image):
    """A1111 infotext from SwarmUI's ``sui_image_params`` block."""
    p = data.get("sui_image_params")
    if not isinstance(p, dict):
        return None
    extra = data.get("sui_extra_data") if isinstance(data.get("sui_extra_data"), dict) else {}

    # original_prompt already carries the <lora:...> tags Swarm stripped out.
    positive = extra.get("original_prompt") or p.get("prompt") or ""
    negative = p.get("negativeprompt") or p.get("negative_prompt") or ""
    if not extra.get("original_prompt"):
        names, weights = p.get("loras") or [], p.get("loraweights") or []
        tags = _lora_tags(zip(names, list(weights) + [1] * len(names)))
        if tags:
            positive = f"{positive.rstrip().rstrip(',')} {' '.join(tags)}".strip()

    fields = []
    _field(fields, "Steps", p.get("steps"))
    _field(fields, "Sampler", _map_sampler(p.get("sampler")))
    _field(fields, "Schedule type", _map_scheduler(p.get("scheduler")))
    _field(fields, "CFG scale", p.get("cfgscale"))
    _field(fields, "Seed", p.get("seed"))
    _size_fields(fields, p.get("width"), p.get("height"), image)
    _field(fields, "Model", _model_name(p.get("model")))
    if p.get("refinermodel"):
        _field(fields, "Refiner", _model_name(p["refinermodel"]))
    if extra.get("initimage_filename") and p.get("initimagecreativity") is not None:
        _field(fields, "Denoising strength", p["initimagecreativity"])
    if p.get("clipstopatlayer"):
        _field(fields, "Clip skip", abs(int(p["clipstopatlayer"])))
    if p.get("swarm_version"):
        _field(fields, "Version", f"SwarmUI {p['swarm_version']}")
    _swarm_hashes(fields, data.get("sui_models"))
    return _infotext(positive, negative, fields)


def _swarm_hashes(fields, models):
    """SwarmUI records a full SHA256 per model in ``sui_models``. Emit them in
    A1111's shape, truncated to the 10 chars Civitai calls AutoV2 so the lookup
    panel can identify anything that isn't installed locally."""
    if not isinstance(models, list):
        return
    lora_hashes = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        digest = str(entry.get("hash") or "").lower().removeprefix("0x")
        name = _model_name(entry.get("name"))
        if not digest or not name:
            continue
        if entry.get("param") == "model":
            _field(fields, "Model hash", digest[:10])
        elif entry.get("param") == "loras":
            lora_hashes.append(f"{name}: {digest[:10]}")
    if lora_hashes:
        _field(fields, "Lora hashes", ", ".join(lora_hashes))


# --------------------------------------------------------------------------
# InvokeAI
# --------------------------------------------------------------------------

def _parse_dream(dream, image):
    """Legacy InvokeAI ``Dream`` string: ``"a painting" -s 50 -S 12 -C 7.5 -A k_lms``."""
    if not isinstance(dream, str) or not dream.strip():
        return None
    m = re.match(r'\s*"(.*?)"\s*(.*)', dream, re.DOTALL)
    prompt, flags = (m.group(1), m.group(2)) if m else (dream, "")

    def flag(name):
        mm = re.search(rf"(?:^|\s){re.escape(name)}\s*(\S+)", flags)
        return mm.group(1) if mm else None

    fields = []
    _field(fields, "Steps", flag("-s"))
    _field(fields, "Sampler", _map_sampler(flag("-A")))
    _field(fields, "CFG scale", flag("-C"))
    _field(fields, "Seed", flag("-S"))
    _size_fields(fields, flag("-W") and int(flag("-W")), flag("-H") and int(flag("-H")), image)
    return _infotext((prompt or "").strip(), "", fields)


def build_invokeai_infotext(m, image):
    """Newer InvokeAI ``invokeai_metadata`` JSON."""
    if not isinstance(m, dict):
        return None
    positive = m.get("positive_prompt") or m.get("prompt") or ""
    if not isinstance(positive, str):
        return None
    model = m.get("model")
    if isinstance(model, dict):
        model = model.get("model_name") or model.get("name") or model.get("key")

    fields = []
    _field(fields, "Steps", m.get("steps"))
    _field(fields, "Sampler", _map_sampler(m.get("scheduler")))
    _field(fields, "CFG scale", m.get("cfg_scale"))
    if m.get("cfg_rescale_multiplier"):
        _field(fields, "Rescale CFG", m["cfg_rescale_multiplier"])
    _field(fields, "Seed", m.get("seed"))
    _size_fields(fields, m.get("width"), m.get("height"), image)
    _field(fields, "Model", _model_name(model))
    return _infotext(positive, m.get("negative_prompt") or "", fields)


def convert_invokeai_metadata(items, image):
    """InvokeAI PNG chunks: ``invokeai_metadata`` / ``sd-metadata`` / ``Dream``."""
    if not isinstance(items, dict):
        return None
    text = build_invokeai_infotext(_load_json(items.get("invokeai_metadata")), image)
    if text:
        return text

    m = _load_json(items.get("sd-metadata"))
    if m:
        img = m.get("image") or {}
        prompt = img.get("prompt")
        if isinstance(prompt, list):  # [{"prompt": "...", "weight": 1.0}, ...]
            prompt = " ".join(p.get("prompt", "") for p in prompt if isinstance(p, dict))
        fields = []
        _field(fields, "Steps", img.get("steps"))
        _field(fields, "Sampler", _map_sampler(img.get("sampler")))
        _field(fields, "CFG scale", img.get("cfg_scale"))
        _field(fields, "Seed", img.get("seed"))
        _size_fields(fields, img.get("width"), img.get("height"), image)
        model = m.get("model_weights") or m.get("model")
        if model and not isinstance(model, dict):
            _field(fields, "Model", _model_name(model))
        return _infotext(prompt or "", "", fields)

    return _parse_dream(items.get("Dream"), image)


# --------------------------------------------------------------------------
# Fooocus and other flat JSON writers
# --------------------------------------------------------------------------

GENERIC_POSITIVE = ("full_prompt", "prompt", "positive_prompt", "positive", "Prompt")
GENERIC_NEGATIVE = ("full_negative_prompt", "negative_prompt", "negative", "Negative Prompt", "uc")
GENERIC_FIELDS = (
    ("Steps", ("steps", "num_inference_steps", "Steps")),
    ("Sampler", ("sampler", "sampler_name", "Sampler")),
    ("Schedule type", ("scheduler", "schedule", "Scheduler")),
    ("CFG scale", ("cfg", "cfg_scale", "guidance_scale", "CFG scale", "cfgscale")),
    ("Seed", ("seed", "Seed")),
    ("Model", ("base_model", "base_model_name", "model", "model_name", "sd_model_checkpoint", "Model")),
)


def build_generic_infotext(data, image):
    """Last-resort conversion for any flat JSON blob with prompt-ish keys
    (Fooocus, Easy Diffusion, one-off scripts)."""
    if not isinstance(data, dict):
        return None
    get = lambda keys: next((data[k] for k in keys if isinstance(data.get(k), str) and data[k].strip()), "")
    positive = get(GENERIC_POSITIVE)
    if not positive:
        return None
    fields = []
    for label, keys in GENERIC_FIELDS:
        value = next((data[k] for k in keys if data.get(k) not in (None, "")), None)
        if label == "Sampler":
            value = _map_sampler(value)
        elif label == "Schedule type":
            value = _map_scheduler(value)
        elif label == "Model":
            value = _model_name(value if not isinstance(value, dict) else value.get("name"))
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            continue
        _field(fields, label, value)

    width, height = data.get("width"), data.get("height")
    res = data.get("resolution")  # Fooocus: "(896, 1152)"
    if not width and isinstance(res, str):
        nums = re.findall(r"\d+", res)
        if len(nums) >= 2:
            width, height = int(nums[0]), int(nums[1])
    _size_fields(fields, width, height, image)

    # Fooocus: "loras": [["name.safetensors", 0.8], ...]
    loras = data.get("loras")
    if isinstance(loras, list):
        pairs = [(l[0], l[1]) for l in loras if isinstance(l, (list, tuple)) and len(l) >= 2]
        tags = _lora_tags(pairs)
        if tags:
            positive = f"{positive.rstrip().rstrip(',')} {' '.join(tags)}".strip()
    return _infotext(positive, get(GENERIC_NEGATIVE), fields)


# --------------------------------------------------------------------------
# ComfyUI graphs
# --------------------------------------------------------------------------

def _is_link(v):
    """ComfyUI links are [node_id, output_slot]; node_id str/int, slot int."""
    return (isinstance(v, list) and len(v) == 2
            and isinstance(v[0], (str, int)) and not isinstance(v[0], bool)
            and isinstance(v[1], int) and not isinstance(v[1], bool))


def _node(prompt, ref):
    if _is_link(ref):
        nid = str(ref[0])
        node = prompt.get(nid)
        if isinstance(node, dict):
            return nid, node
    return None, None


def _trace(prompt, ref, want, visited=None, depth=0):
    """Walk back from a link, depth-first, returning the first ``want`` hit.

    ``want`` is ``f(node_id, node, inputs) -> value or None``."""
    if visited is None:
        visited = set()
    if depth > 64:
        return None
    nid, node = _node(prompt, ref)
    if node is None or nid in visited:
        return None
    visited.add(nid)
    inputs = node.get("inputs", {}) or {}
    found = want(nid, node, inputs)
    if found is not None:
        return found
    # Conditioning pass-through nodes (ControlNetApplyAdvanced and friends) carry
    # both chains: output slot 0 continues the positive one, slot 1 the negative.
    # Without this the negative prompt resolves to the positive text. Only prompt
    # lookups care — a model/size lookup must keep walking every input.
    if want is _want_text and _is_link(inputs.get("positive")) and _is_link(inputs.get("negative")):
        branch = inputs["negative"] if ref[1] == 1 else inputs["positive"]
        return _trace(prompt, branch, want, visited, depth + 1)
    for v in inputs.values():
        if _is_link(v):
            r = _trace(prompt, v, want, visited, depth + 1)
            if r is not None:
                return r
    return None


def _want_text(nid, node, inputs):
    if _lora_node(node, inputs):  # a lora widget's "text" is not a prompt
        return None
    for k in TEXT_KEYS:
        val = inputs.get(k)
        if isinstance(val, str):  # accept "" so empty negatives resolve correctly
            return val
    return None


# Loaders that hold a filename but not *the* checkpoint — without this the VAE
# or text-encoder file gets reported as the model.
MODEL_SKIP_CLASSES = ("vae", "clip", "textencode", "textencoder", "t5", "upscale",
                      "controlnet", "lora", "ipadapter", "detector", "sam")


def _want_model(nid, node, inputs):
    ct = str(node.get("class_type", "") or "").lower()
    if any(s in ct for s in MODEL_SKIP_CLASSES):
        return None
    for k in ("ckpt_name", "unet_name", "model", "model_path", "base_ckpt_name", "model_name"):
        val = inputs.get(k)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _want_value(nid, node, inputs):
    """Constant nodes (INTConstant, Primitive, ...) feeding a scalar widget."""
    for k in ("value", "Value", "int", "float", "number"):
        val = inputs.get(k)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
    return None


def _want_size(nid, node, inputs):
    w, h = inputs.get("width"), inputs.get("height")
    if isinstance(w, (int, float)) and not isinstance(w, bool) and isinstance(h, (int, float)) and not isinstance(h, bool):
        return int(w), int(h)
    return None


def _lora_node(node, inputs):
    """``[(name, weight), ...]`` for any known LoRA loader node, else None."""
    ct = str(node.get("class_type", "") or "")
    if "lora" not in ct.lower():
        return None

    name = inputs.get("lora_name") or inputs.get("lora")  # LoraLoader / WanVideoLoraSelect
    if isinstance(name, str) and name.strip():
        return [(name, inputs.get("strength_model", inputs.get("strength", 1)))]

    out = []
    # LoraManager: {"loras": {"__value__": [{"name":..,"strength":..,"active":..}]}}
    value = inputs.get("loras")
    if isinstance(value, dict):
        for entry in value.get("__value__") or []:
            if isinstance(entry, dict) and entry.get("active", True) and entry.get("name"):
                out.append((entry["name"], entry.get("strength", 1)))
    # rgthree Power Lora Loader: {"lora_1": {"on":true,"lora":..,"strength":..}}
    for key, entry in inputs.items():
        if re.fullmatch(r"lora_\d+", str(key)) and isinstance(entry, dict) and entry.get("on", True):
            if entry.get("lora"):
                out.append((entry["lora"], entry.get("strength", 1)))
    return out or None


def _find_sampler_node(prompt):
    """Pick the node most likely to be the main sampler."""
    best = None
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        ct = node.get("class_type", "") or ""
        score = 0
        if ct in SAMPLER_CLASSES:
            score += 5
        elif "KSampler" in ct or "Sampler" in ct:
            score += 2
        if "positive" in inputs and "negative" in inputs:
            score += 3
        if "sampler_name" in inputs:
            score += 2
        if "steps" in inputs:
            score += 1
        if "latent_image" in inputs or "latent" in inputs:
            score += 1
        # Prefer the full-strength pass: in a two-pass graph the upscale/detail
        # sampler runs at a low denoise, and the base one holds the real params.
        denoise = inputs.get("denoise", 1.0)
        if not isinstance(denoise, (int, float)) or isinstance(denoise, bool) or float(denoise) == 1.0:
            score += 1
        if score >= 4 and (best is None or score > best[0]):
            best = (score, nid, node)
    return (best[1], best[2]) if best else (None, None)


def _find_cond_refs(prompt, start_nid):
    """Find the nearest node carrying positive/negative conditioning links.

    For KSampler these live on the sampler itself; for SamplerCustomAdvanced they
    live on the linked CFGGuider node (positive/negative) or BasicGuider node
    (a single ``conditioning`` input, used by Flux)."""
    visited = set()
    stack = [start_nid]
    cond_fallback = None
    while stack:
        nid = stack.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        pos, neg = inputs.get("positive"), inputs.get("negative")
        if _is_link(pos) or _is_link(neg):
            return (pos if _is_link(pos) else None), (neg if _is_link(neg) else None)
        if cond_fallback is None and _is_link(inputs.get("conditioning")):
            cond_fallback = inputs["conditioning"]
        for v in inputs.values():
            if _is_link(v):
                stack.append(str(v[0]))
    return cond_fallback, None


def _walk_subgraph(prompt, start_nid, keys):
    """Breadth-first over the sampler subgraph; returns (scalars, loras, pair).

    Both plain KSampler (values on the node) and SamplerCustom* (values spread
    across KSamplerSelect / BasicScheduler / CFGGuider / FluxGuidance nodes) work.
    LoRA loaders anywhere in the model chain are collected on the way, and
    ``pair`` is the (positive, negative) text of a wrapper node that keeps both
    prompts on itself (WanVideo, Hunyuan and similar custom packs)."""
    scalars, loras, pair, visited, queue = {}, [], None, set(), [start_nid]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {}) or {}
        for k, v in inputs.items():
            if k not in keys or k in scalars:
                continue
            if _is_link(v):
                # steps/cfg are often fed by an INTConstant or Primitive node
                resolved = _trace(prompt, v, _want_value)
                if resolved is not None:
                    scalars[k] = resolved
            elif isinstance(v, (int, float, str)) and not isinstance(v, bool):
                scalars[k] = v
        if "RescaleCFG" in str(node.get("class_type", "")) and isinstance(inputs.get("multiplier"), (int, float)):
            scalars.setdefault("rescale_cfg", inputs["multiplier"])
        if pair is None and isinstance(inputs.get("positive_prompt"), str) and isinstance(inputs.get("negative_prompt"), str):
            pair = (inputs["positive_prompt"], inputs["negative_prompt"])
        loras.extend(_lora_node(node, inputs) or [])
        for v in inputs.values():
            if _is_link(v):
                child = str(v[0])
                if child not in visited:
                    queue.append(child)
    return scalars, loras, pair


def build_comfyui_infotext(prompt, image):
    """A1111 infotext from a ComfyUI API-format ``prompt`` graph, or None."""
    if not isinstance(prompt, dict) or not prompt:
        return None
    sampler_nid, sampler_node = _find_sampler_node(prompt)
    if sampler_node is None:
        return None

    pos_ref, neg_ref = _find_cond_refs(prompt, sampler_nid)
    positive = (_trace(prompt, pos_ref, _want_text) if pos_ref else None) or ""
    negative = (_trace(prompt, neg_ref, _want_text) if neg_ref else None) or ""

    scalars, loras, pair = _walk_subgraph(prompt, sampler_nid, SCALAR_KEYS)
    if not positive and pair:
        positive, negative = pair
    tags = _lora_tags(loras)
    if tags:
        positive = f"{positive.rstrip().rstrip(',')} {' '.join(tags)}".strip()

    model = _trace(prompt, [sampler_nid, 0], _want_model)
    size = None
    inputs = sampler_node.get("inputs", {}) or {}
    for k in ("latent_image", "latent", "samples"):
        if _is_link(inputs.get(k)):
            size = _trace(prompt, inputs[k], _want_size)
            if size:
                break

    fields = []
    _field(fields, "Steps", scalars.get("steps"))
    _field(fields, "Sampler", _map_sampler(scalars.get("sampler_name")))
    _field(fields, "Schedule type", _map_scheduler(scalars.get("scheduler")))
    _field(fields, "CFG scale", scalars.get("cfg"))
    if scalars.get("guidance") is not None:
        _field(fields, "Distilled CFG Scale", scalars["guidance"])
    if scalars.get("rescale_cfg"):
        _field(fields, "Rescale CFG", scalars["rescale_cfg"])
    _field(fields, "Seed", scalars.get("seed", scalars.get("noise_seed")))
    _size_fields(fields, size and size[0], size and size[1], image)
    _field(fields, "Model", _model_name(model))
    denoise = scalars.get("denoise")
    if isinstance(denoise, (int, float)) and not isinstance(denoise, bool) and float(denoise) != 1.0:
        _field(fields, "Denoising strength", denoise)
    return _infotext(positive, negative, fields)


def convert_comfyui_metadata(image, items):
    """Read the ``prompt`` (API-format) graph chunk; ``workflow`` alone is not
    enough to reliably recover parameters, so those are skipped."""
    raw = items.get("prompt") if isinstance(items, dict) else None
    if raw is None:
        raw = (getattr(image, "info", None) or {}).get("prompt")
    return build_comfyui_infotext(_load_json(raw), image)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _is_comfy_graph(data):
    return any(isinstance(v, dict) and "class_type" in v for v in data.values())


def convert_json_metadata(data, image):
    """Convert a JSON metadata blob, whichever generator wrote it."""
    if not isinstance(data, dict):
        return None
    if "sui_image_params" in data:
        return build_swarmui_infotext(data, image)
    if data.get("Software") == "NovelAI" or "Comment" in data:
        text = build_novelai_infotext(data, image)
        if text:
            return text
    if "invokeai_metadata" in data or "sd-metadata" in data or "Dream" in data:
        text = convert_invokeai_metadata(data, image)
        if text:
            return text
    if data.get("positive_prompt") is not None:
        text = build_invokeai_infotext(data, image)
        if text:
            return text
    if _is_comfy_graph(data):
        text = build_comfyui_infotext(data, image)
        if text:
            return text
    return build_generic_infotext(data, image)


# ashen-sensored's stealth format: a signature, a 32-bit big-endian bit count,
# then the payload - gzipped for the *comp variants. One bit per byte of either
# the alpha channel or the interleaved RGB bytes.
STEALTH_SIGS = {
    "stealth_pnginfo": ("alpha", False),
    "stealth_pngcomp": ("alpha", True),
    "stealth_rgbinfo": ("rgb", False),
    "stealth_rgbcomp": ("rgb", True),
}
_SIG_LEN = 15                     # every signature above is 15 characters
_STEALTH_MAX = 8 * 1024 * 1024    # refuse absurd lengths from a bad read


def _lsb_bits(raw, start, count):
    return "".join("1" if b & 1 else "0" for b in raw[start:start + count])


def _bits_to_bytes(bits):
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits) // 8 * 8, 8))


def read_stealth_payload(image):
    """Text hidden in an image's low bits, or None.

    NovelAI writes its metadata this way, and so does sd-webui's stealth
    pnginfo extension, which is how images survive sites that strip PNG chunks.
    """
    if image is None or not getattr(image, "width", 0) or not getattr(image, "height", 0):
        return None
    try:
        # the writer walks columns (x outer, y inner); tobytes() walks rows
        transposed = image.transpose(Image.TRANSPOSE)
    except Exception:
        return None

    for mode in ("alpha", "rgb"):
        try:
            if mode == "alpha":
                if image.mode != "RGBA":
                    continue
                raw = transposed.getchannel("A").tobytes()
            else:
                raw = transposed.convert("RGB").tobytes()

            header = _SIG_LEN * 8 + 32
            if len(raw) < header:
                continue
            sig = _bits_to_bytes(_lsb_bits(raw, 0, _SIG_LEN * 8)).decode("utf-8", "ignore")
            if STEALTH_SIGS.get(sig, (None, None))[0] != mode:
                continue

            length = int(_lsb_bits(raw, _SIG_LEN * 8, 32), 2)
            if not 0 < length <= _STEALTH_MAX or header + length > len(raw):
                continue
            payload = _bits_to_bytes(_lsb_bits(raw, header, length))
            if STEALTH_SIGS[sig][1]:
                payload = gzip.decompress(payload)
            text = payload.decode("utf-8", "ignore").strip()
            if text:
                return text
        except Exception:
            continue
    return None


def _from_stealth(image):
    """Convert a hidden payload. NovelAI hides its whole PNG chunk set as JSON;
    sd-webui hides plain A1111 text, which needs no conversion."""
    text = read_stealth_payload(image)
    if not text:
        return None
    data = _load_json(text)
    if data is None:
        return text                      # already A1111 infotext
    if data.get("Software") == "NovelAI" or "Comment" in data:
        converted = build_novelai_infotext(data, image)
        if converted:
            return converted
    return convert_json_metadata(data, image) or text


def convert_metadata(image, items=None, geninfo=None):
    """A1111 infotext for any AI image, or None when there is nothing to add.

    Also exposed as ``shared.ai_pnginfo_convert`` for the stealth pnginfo
    extension, which passes its decoded alpha-channel payload as ``geninfo``.
    """
    items = items if isinstance(items, dict) else {}

    if items.get("Software") == "NovelAI" and items.get("Comment") is not None:
        text = build_novelai_infotext(items, image)
        if text:
            return text

    data = _load_json(geninfo) or _load_json(items.get("parameters"))
    if data is not None:
        text = convert_json_metadata(data, image)
        if text:
            return text

    text = convert_invokeai_metadata(items, image)
    if text:
        return text

    text = convert_comfyui_metadata(image, items)
    if text:
        return text

    # Nothing in the chunks. The image may still carry a hidden payload - that is
    # how NovelAI images survive sites that strip metadata.
    return _from_stealth(image)


def _needs_conversion(geninfo, items):
    """A1111 text is already perfect; JSON, NovelAI and empty geninfo are not."""
    if not geninfo:
        return True
    if isinstance(items, dict) and items.get("Software") == "NovelAI":
        return True
    return isinstance(geninfo, str) and geninfo.lstrip().startswith("{")


def chain_patch(current):
    """What our wrapper should call. "Reload UI" re-executes extension scripts in
    the same process, so unwrap a previous patch of ours instead of stacking on
    it — otherwise every reload adds another layer."""
    return getattr(current, "_pnginfo_inner", None) or current


_previous_read_info_from_image = chain_patch(images.read_info_from_image)


def read_info_from_image_comfyui(image):
    geninfo, items = _previous_read_info_from_image(image)

    if not shared.opts.data.get("comfyui_pnginfo_enabled", True):
        return geninfo, items
    if not _needs_conversion(geninfo, items):
        return geninfo, items

    try:
        converted = convert_metadata(image, items, geninfo)
    except Exception:
        converted = None
    if converted is None:
        return geninfo, items

    # Drop the raw blobs the conversion replaces, so the info panel is readable.
    if isinstance(items, dict):
        items = {k: v for k, v in items.items() if k not in RAW_KEYS}
    return converted, items


def on_ui_settings():
    section = ("comfyui_pnginfo", "AI PNGinfo")
    shared.opts.add_option("comfyui_pnginfo_enabled", shared.OptionInfo(
        True, "Convert ComfyUI / SwarmUI / NovelAI / InvokeAI metadata in PNG Info",
        gr.Checkbox, {"interactive": True}, section=section))


read_info_from_image_comfyui._pnginfo_inner = _previous_read_info_from_image
images.read_info_from_image = read_info_from_image_comfyui
# Published for the stealth pnginfo extension (loads later, so it can just read
# this off shared at call time).
shared.ai_pnginfo_convert = convert_metadata

script_callbacks.on_ui_settings(on_ui_settings)
