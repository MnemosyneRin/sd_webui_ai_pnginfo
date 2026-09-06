"""Self-check for the PNG Info converter. Run it directly:

    python extensions/sd_webui_ai_pnginfo/test_pnginfo.py

It stubs the webui modules, so it needs nothing but PIL.
"""
import gzip
import importlib.util
import io
import json
import os
import re
import sys
import types

from PIL import Image, PngImagePlugin

HERE = os.path.dirname(os.path.abspath(__file__))


# The regex the WebUI uses to read the parameters line (infotext_utils.py).
RE_PARAM = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def stub_webui():
    gr = types.ModuleType("gradio")
    gr.Checkbox = object
    gr.Dropdown = object
    gr.Image = type("Image", (), {})
    sys.modules["gradio"] = gr

    def read_info_from_image(image):
        """The host's own reader, reduced to what matters here."""
        items = (image.info or {}).copy()
        geninfo = items.pop("parameters", None)
        if items.get("Software") == "NovelAI":
            try:
                j = json.loads(items["Comment"])
                geninfo = (f'{items["Description"]}\nNegative prompt: {j["uc"]}\n'
                           f'Steps: {j["steps"]}, Sampler: Euler a, CFG scale: {j["scale"]}, '
                           f'Seed: {j["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337')
            except Exception:
                pass
        return geninfo, items

    images = types.ModuleType("modules.images")
    images.read_info_from_image = read_info_from_image
    images.resize_image = lambda mode, im, w, h, upscaler=None, *a, **k: im
    sc = types.ModuleType("modules.script_callbacks")
    sc.on_ui_settings = lambda *a, **k: None
    sc.on_before_image_saved = lambda *a, **k: None
    sc.on_after_component = lambda *a, **k: None

    class ImageSaveParams:
        def __init__(self, image, p=None, filename="", pnginfo=None):
            self.image, self.p, self.filename = image, p, filename
            self.pnginfo = pnginfo if pnginfo is not None else {}
    sc.ImageSaveParams = ImageSaveParams

    infotext = types.ModuleType("modules.infotext_utils")
    infotext.send_image_and_dimensions = lambda x: (x, 0, 0)
    sys.modules["modules.infotext_utils"] = infotext
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(data={}, add_option=lambda *a, **k: None)
    class OptionInfo:
        def __init__(self, *a, **k):
            pass

        def info(self, *a, **k):
            return self

    shared.OptionInfo = OptionInfo
    samplers = types.ModuleType("modules.sd_samplers")
    samplers.samplers_map = {
        "k_euler_ancestral": "Euler a", "k_euler_a": "Euler a", "k_euler": "Euler",
        "k_dpmpp_2m": "DPM++ 2M", "k_dpmpp_2m_sde": "DPM++ 2M SDE", "k_dpmpp_sde": "DPM++ SDE",
        "k_dpmpp_3m_sde": "DPM++ 3M SDE", "er_sde": "ER SDE", "k_lms": "LMS", "k_heun": "Heun",
        "k_dpm_2": "DPM2", "res_multistep": "Res Multistep", "unipc": "UniPC", "k_lcm": "LCM",
        "restart": "Restart", "ddim": "DDIM",
    }
    samplers.all_samplers_map = {v: v for v in samplers.samplers_map.values()}
    scheds = types.ModuleType("modules.sd_schedulers")

    class S:
        def __init__(s, n, l):
            s.name, s.label = n, l

    table = [S("automatic", "Automatic"), S("karras", "Karras"), S("exponential", "Exponential"),
             S("polyexponential", "Polyexponential"), S("normal", "Normal"), S("simple", "Simple"),
             S("uniform", "Uniform"), S("sgm_uniform", "SGM Uniform"), S("linear_quadratic", "Linear Quadratic"),
             S("kl_optimal", "KL Optimal"), S("ddim", "DDIM"), S("align_your_steps", "Align Your Steps"),
             S("beta", "Beta"), S("turbo", "Turbo"), S("bong_tangent", "Bong Tangent"),
             S("flow_match", "FlowMatchEulerDiscrete"), S("flux2", "Flux2")]
    scheds.schedulers = table
    scheds.schedulers_map = {**{x.name: x for x in table}, **{x.label: x for x in table}}

    modules = types.ModuleType("modules")
    modules.images, modules.script_callbacks, modules.shared = images, sc, shared
    modules.sd_samplers, modules.sd_schedulers = samplers, scheds
    modules.infotext_utils = infotext
    sys.modules.update({"modules": modules, "modules.images": images, "modules.script_callbacks": sc,
                        "modules.shared": shared, "modules.sd_samplers": samplers,
                        "modules.sd_schedulers": scheds, "modules.infotext_utils": infotext})


def load(name="ai_pnginfo"):
    stub_webui()
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_lookup():
    extras = types.ModuleType("modules.extras")
    extras.run_pnginfo = lambda image: ("", "", "")
    sys.modules["modules.extras"] = extras
    sys.modules["modules"].extras = extras
    return load("lora_lookup")


def png(size=(64, 64), **chunks):
    meta = PngImagePlugin.PngInfo()
    for k, v in chunks.items():
        meta.add_text(k.replace("__", " "), v)
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "PNG", pnginfo=meta)
    return Image.open(buf)


def stealth_png(text, mode="alpha", compress=True, size=(256, 256)):
    """Write a payload into the low bits the way NovelAI and stealth_pnginfo do."""
    sig = {("alpha", True): "stealth_pngcomp", ("alpha", False): "stealth_pnginfo",
           ("rgb", True): "stealth_rgbcomp", ("rgb", False): "stealth_rgbinfo"}[(mode, compress)]
    payload = gzip.compress(text.encode()) if compress else text.encode()
    bits = ("".join(f"{b:08b}" for b in sig.encode())
            + f"{len(payload) * 8:032b}"
            + "".join(f"{b:08b}" for b in payload))
    img = Image.new("RGBA" if mode == "alpha" else "RGB", size,
                    (0, 0, 0, 255) if mode == "alpha" else (0, 0, 0))
    px, i = img.load(), 0
    for x in range(img.width):                     # the writer walks columns
        for y in range(img.height):
            if i >= len(bits):
                return img
            v = list(px[x, y])
            if mode == "alpha":
                v[3] = (v[3] & ~1) | int(bits[i]); i += 1
            else:
                for c in range(3):
                    if i < len(bits):
                        v[c] = (v[c] & ~1) | int(bits[i]); i += 1
            px[x, y] = tuple(v)
    assert i >= len(bits), "test image too small for the payload"
    return img


def check_pastable(text, label):
    """The WebUI reads the prompt from the leading lines and every parameter from
    the last one; if that line doesn't parse, "Send to txt2img" silently drops
    the settings."""
    *lines, lastline = text.strip().split("\n")
    assert len(RE_PARAM.findall(lastline)) >= 3, (label, "unparseable parameter line", lastline)
    assert not lastline.startswith("Negative prompt:"), (label, "negative prompt eaten by the param line")
    fields = dict(RE_PARAM.findall(lastline))
    return fields


NAI_COMMENT = {
    "prompt": "2::gigantic, breasts::, {{tall}}, [small]",
    "uc": "lowres, 1.1::bad quality::",
    "steps": 28, "sampler": "k_euler_ancestral", "noise_schedule": "karras",
    "scale": 5.0, "cfg_rescale": 0.4, "seed": 12345, "width": 832, "height": 1216,
    "v4_prompt": {"caption": {
        "base_caption": "2::gigantic, breasts::, {{tall}}, [small]",
        "char_captions": [{"char_caption": "zhu yuan, police uniform"},
                          {"char_caption": "jane doe, rat tail"}]}},
    "v4_negative_prompt": {"caption": {"base_caption": "lowres, 1.1::bad quality::",
                                       "char_captions": [{"char_caption": ""}]}},
}

SWARM = {
    "sui_image_params": {
        "prompt": "1girl, office lady", "negativeprompt": "pov, feet",
        "model": "realDream_animaV2", "seed": 1755497345, "steps": 16, "cfgscale": 2.0,
        "width": 1248, "height": 1824, "sampler": "er_sde", "scheduler": "beta_1_1",
        "initimagecreativity": 0.6, "swarm_version": "0.9.8.1",
        "loras": ["styleA", "turbo-v0.2"], "loraweights": ["0.4", "0.7"]},
    "sui_extra_data": {"initimage_filename": "raw/x.png"},
    "sui_models": [
        {"name": "realDream_animaV2.safetensors", "param": "model", "hash": "0x84fc41ea0821fa0a"},
        {"name": "styleA.safetensors", "param": "loras", "hash": "0xe8055d97ec312203"},
        {"name": "turbo-v0.2.safetensors", "param": "loras", "hash": "0x5c390318d2261"}],
}

# KSampler -> ControlNetApplyAdvanced (positive on slot 0, negative on slot 1).
COMFY = {
    "9": {"class_type": "KSampler", "inputs": {
        "seed": 42, "steps": 25, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde_gpu",
        "scheduler": "simple", "denoise": 1.0, "model": ["4", 0],
        "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["5", 0]}},
    "7": {"class_type": "ControlNetApplyAdvanced",
          "inputs": {"strength": 1.0, "positive": ["1", 0], "negative": ["2", 0]}},
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat, masterpiece"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 896, "height": 1152}},
    "4": {"class_type": "LoraLoader", "inputs": {
        "lora_name": "loras/style_v1.safetensors", "strength_model": 0.8,
        "model": ["3", 0], "clip": ["3", 1]}},
    "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "anima_v2.safetensors"}},
}


def main():
    m = load()

    # --- NovelAI emphasis, exactly as the WebUI's prompt parser wants it -----
    assert m.nai_prompt_to_a1111("2::gigantic, breasts::") == "(gigantic, breasts:2)"
    assert m.nai_prompt_to_a1111("1.5::tag::, plain") == "(tag:1.5), plain"
    assert m.nai_prompt_to_a1111("{{tall}}") == "(tall:1.1025)"
    assert m.nai_prompt_to_a1111("[small]") == "(small:0.9524)"
    assert m.nai_prompt_to_a1111("no markers here") == "no markers here"
    print("ok  NovelAI emphasis -> A1111 weights")

    # --- NovelAI image ------------------------------------------------------
    img = png((832, 1216), Software="NovelAI", Description="2::gigantic, breasts::",
              Source="NovelAI Diffusion V4.5 B9F340FD", Comment=json.dumps(NAI_COMMENT))
    text, items = m.read_info_from_image_comfyui(img)
    fields = check_pastable(text, "novelai")
    assert "zhu yuan, police uniform" in text and "jane doe, rat tail" in text, text
    assert "(gigantic, breasts:2)" in text and "::" not in text, text
    assert text.splitlines()[-2].startswith("Negative prompt: lowres, (bad quality:1.1)"), text
    assert fields["Sampler"] == "Euler a" and fields["Schedule type"] == "Karras", fields
    assert fields["Rescale CFG"] == "0.4" and fields["Size"] == "832x1216", fields
    assert fields["Model"] == "NovelAI Diffusion V4.5 B9F340FD", fields
    assert not {"Comment", "Description", "Software", "Source"} & set(items), items
    print("ok  NovelAI: character captions, weights, full parameters")

    # ...unless sdwebui-nai-api is installed, which reads NovelAI itself and wants
    # the NAI dialect intact. Chunks are handed back untouched; a stealth-only
    # image gets its raw JSON back, which is what that extension parses.
    sys.modules["nai_api_gen"] = types.ModuleType("nai_api_gen")
    try:
        untouched = m._previous_read_info_from_image(img)
        assert m.read_info_from_image_comfyui(img) == untouched, "NovelAI chunks were rewritten"
        raw = json.dumps({"Software": "NovelAI", "Comment": json.dumps(NAI_COMMENT)})
        text, _ = m.read_info_from_image_comfyui(stealth_png(raw))
        assert text == raw, text
    finally:
        del sys.modules["nai_api_gen"]
    text, _ = m.read_info_from_image_comfyui(img)
    assert "zhu yuan" in text, "conversion did not come back"
    print("ok  NovelAI: left alone when sdwebui-nai-api is installed")

    # --- SwarmUI ------------------------------------------------------------
    img = png((1248, 1824), parameters=json.dumps(SWARM))
    text, items = m.read_info_from_image_comfyui(img)
    fields = check_pastable(text, "swarmui")
    assert text.startswith("1girl, office lady <lora:styleA:0.4> <lora:turbo-v0.2:0.7>"), text
    assert fields["Sampler"] == "ER SDE" and fields["Schedule type"] == "Beta", fields
    assert fields["Model"] == "realDream_animaV2" and fields["Seed"] == "1755497345", fields
    assert fields["Denoising strength"] == "0.6" and fields["Size"] == "1248x1824", fields
    assert "sui_image_params" not in text, "raw JSON still shown"
    assert fields["Model hash"] == "84fc41ea08", fields
    assert fields["Lora hashes"] == '"styleA: e8055d97ec, turbo-v0.2: 5c390318d2"', fields
    print("ok  SwarmUI: prompt, loras, model, sampler, hashes")

    # --- resource lookup panel ---------------------------------------------
    lookup = load_lookup()
    res = lookup.parse_resources(text)
    assert res == [("Checkpoint", "realDream_animaV2", "84fc41ea08"),
                   ("LoRA", "styleA", "e8055d97ec"),
                   ("LoRA", "turbo-v0.2", "5c390318d2")], res
    lookup.shared.opts.data["pnginfo_lookup_civitai"] = False  # keep the check offline
    report = lookup.build_report(text)
    assert "0/3 installed" in report, report
    assert report.count("missing") == 3 and "civitai.com/search" in report, report
    assert "styleA" in report and "realDream_animaV2" in report, report
    # an installed one is reported as such
    lookup.find_lora = lambda name: types.SimpleNamespace(filename="C:/x/styleA.safetensors") if name == "styleA" else None
    assert "1/3 installed" in lookup.build_report(text)
    assert not lookup.build_report("a prompt with no models at all")
    print("ok  lookup panel: installed / missing / links")

    # --- ComfyUI ------------------------------------------------------------
    img = png((896, 1152), prompt=json.dumps(COMFY))
    text, items = m.read_info_from_image_comfyui(img)
    fields = check_pastable(text, "comfyui")
    assert text.startswith("a cat, masterpiece <lora:style_v1:0.8>"), text
    # the negative must come from its own branch of the ControlNet node
    assert "Negative prompt: worst quality" in text, text
    assert fields["Sampler"] == "DPM++ 2M SDE" and fields["Schedule type"] == "Simple", fields
    assert fields["Model"] == "anima_v2" and fields["Size"] == "896x1152", fields
    assert "prompt" not in items, items
    print("ok  ComfyUI: prompts, lora, model, size")

    # --- Fooocus / generic JSON --------------------------------------------
    img = png((896, 1152), parameters=json.dumps({
        "prompt": "a dog", "negative_prompt": "blurry", "steps": 30, "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras", "guidance_scale": 7.0, "seed": 7, "base_model": "juggernaut.safetensors",
        "resolution": "(896, 1152)", "loras": [["detail.safetensors", 0.5]]}))
    text, _ = m.read_info_from_image_comfyui(img)
    fields = check_pastable(text, "fooocus")
    assert text.startswith("a dog <lora:detail:0.5>") and "Negative prompt: blurry" in text, text
    assert fields["Model"] == "juggernaut" and fields["Size"] == "896x1152", fields
    print("ok  Fooocus / generic JSON")

    # --- plain A1111 text must be left completely alone ---------------------
    original = "a cat\nNegative prompt: bad\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 1"
    img = png(parameters=original)
    text, _ = m.read_info_from_image_comfyui(img)
    assert text == original, text
    # ...and an image with no metadata at all stays empty
    text, _ = m.read_info_from_image_comfyui(png())
    assert not text, text
    print("ok  A1111 passthrough, empty images")


    # --- InvokeAI ------------------------------------------------------------
    img = png(invokeai_metadata=json.dumps({
        "positive_prompt": "a knight in a field", "negative_prompt": "blurry",
        "steps": 30, "scheduler": "euler_a", "cfg_scale": 7.5, "seed": 99,
        "width": 768, "height": 1024, "model": {"model_name": "dreamshaper"}}))
    text, _ = m.read_info_from_image_comfyui(img)
    fields = check_pastable(text, "invokeai")
    assert text.startswith("a knight in a field") and "Negative prompt: blurry" in text, text
    assert fields["Seed"] == "99" and fields["Size"] == "768x1024", fields
    assert fields["Model"] == "dreamshaper", fields

    # legacy sd-metadata, whose prompt is a weighted list
    img = png(**{"sd-metadata": json.dumps({
        "model_weights": "sd-v1-5.ckpt",
        "image": {"prompt": [{"prompt": "a castle", "weight": 1.0}], "steps": 20,
                  "sampler": "k_euler", "cfg_scale": 7, "seed": 5,
                  "width": 512, "height": 512}})})
    text, _ = m.read_info_from_image_comfyui(img)
    assert text.startswith("a castle") and "Seed: 5" in text, text
    print("ok  InvokeAI: invokeai_metadata and legacy sd-metadata")

    # --- stealth payloads, with no PNG chunks at all --------------------------
    # NovelAI hides its whole chunk set as JSON in the alpha channel
    hidden = json.dumps({"Software": "NovelAI", "Source": "NovelAI Diffusion V5 0ADF9AB7",
                         "Comment": json.dumps(NAI_COMMENT)})
    text, _ = m.read_info_from_image_comfyui(stealth_png(hidden))
    fields = check_pastable(text, "stealth novelai")
    assert "zhu yuan, police uniform" in text, ("character captions lost", text)
    assert "(gigantic, breasts:2)" in text, ("emphasis not converted", text)
    assert fields["Seed"] == "12345" and fields["Size"] == "832x1216", fields

    # uncompressed alpha variant
    text, _ = m.read_info_from_image_comfyui(stealth_png(hidden, compress=False))
    assert "zhu yuan" in text, text

    # sd-webui hides plain A1111 text in the RGB low bits; pass it through as-is
    plain = "a fox\nNegative prompt: bad\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 3"
    for compress in (True, False):
        text, _ = m.read_info_from_image_comfyui(stealth_png(plain, "rgb", compress))
        assert text == plain, (compress, text)

    # a plain image must not be mistaken for a stealth one
    assert m.read_stealth_payload(png()) is None
    assert m.read_stealth_payload(Image.new("RGBA", (8, 8), (1, 2, 3, 255))) is None
    print("ok  stealth alpha/rgb payloads, compressed and not")


    # --- writing: round-trip through our own writer and reader ---------------
    params_text = ("a robot" + chr(92) + "n" + "Negative prompt: blurry" + chr(92) + "n"
                   + "Steps: 25, Sampler: Euler a, CFG scale: 6, Seed: 42, Size: 512x512")
    for mode in ("alpha", "rgb"):
        for compress in (True, False):
            src = Image.new("RGBA" if mode == "alpha" else "RGB", (128, 128), (30, 60, 90, 255))
            out = m.write_stealth_payload(src.copy(), params_text, mode, compress)
            assert out is not None, (mode, compress, "payload did not fit")
            assert m.read_stealth_payload(out) == params_text, (mode, compress)
            # a hidden payload must not visibly change the image
            worst = max(abs(a_ - b_) for pa, pb in zip(src.convert("RGBA").getdata(),
                                                       out.convert("RGBA").getdata())
                        for a_, b_ in zip(pa, pb))
            assert worst <= 1, (mode, compress, "changed pixels by", worst)

    # too small to hold it -> refuses rather than writing a truncated payload
    assert m.write_stealth_payload(Image.new("RGBA", (4, 4)), params_text * 20) is None

    # the save hook: writes only for PNG, only when enabled
    def saved(filename, enabled=True, text=params_text):
        img = Image.new("RGBA", (128, 128), (10, 20, 30, 255))
        m.shared.opts.data["ai_pnginfo_write"] = enabled
        p_ = m.ImageSaveParams(img, None, filename, {"parameters": text} if text else {})
        m.add_stealth_pnginfo(p_)
        return m.read_stealth_payload(p_.image)

    assert saved("out.png") == params_text
    assert saved("out.jpg") is None, "must not try to hide data in a lossy format"
    assert saved("out.png", enabled=False) is None
    assert saved("out.png", text=None) is None

    # and it stands down if the standalone stealth extension is installed
    m.shared.opts.data_labels = {"stealth_pnginfo": object()}
    assert saved("out.png") is None, "two writers would overwrite each other"
    m.shared.opts.data_labels = {}
    print("ok  writing: round-trip, all variants, save hook guards")


if __name__ == "__main__":
    sys.exit(main())
