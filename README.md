# sd_webui_ai_pnginfo

Makes **any AI image** readable in the WebUI **PNG Info** tab — and optionally
hides your own parameters in saved images so they survive sites that strip them.

One extension, nothing else needed. It replaces `sd_webui_stealth_pnginfo`
rather than sitting alongside it.

The WebUI only understands flat A1111 `parameters` text. Everything else either
shows nothing or dumps a wall of raw JSON, and "Send to txt2img" comes up empty.
This extension converts them all to clean A1111 infotext, so the info panel reads
properly and every paste button works.

## Supported

| Source | Read from |
|---|---|
| **ComfyUI** | `prompt` node-graph chunk |
| **SwarmUI** | `sui_image_params` JSON |
| **NovelAI** | `Comment` chunk, or the hidden stealth payload |
| **InvokeAI** | `invokeai_metadata`, `sd-metadata`, legacy `Dream` |
| **Fooocus** and other flat-JSON tools | prompt-ish JSON keys |
| **A1111 / Forge** | left exactly as-is |

Recovered where present: positive/negative prompt, model, LoRAs (as
`<lora:name:weight>`), sampler + schedule type (mapped to neoforge names), steps,
CFG scale, seed, size, denoising strength, Rescale CFG, Flux distilled CFG.

## NovelAI

- **V4 character prompts are kept.** The flat `Description` field NovelAI writes
  only holds the base caption, which is why the per-character `CHAR:` text is
  missing everywhere else. This reads `v4_prompt` and appends every character
  caption.
- **Emphasis is translated to A1111 weights**, so the prompt actually means the
  same thing when you send it to txt2img:
  `2::gigantic, breasts::` → `(gigantic, breasts:2)`, `{{tag}}` → `(tag:1.1025)`,
  `[tag]` → `(tag:0.9524)`.

Only prompts that came out of NovelAI metadata are rewritten — an A1111 prompt
you typed yourself is never touched.

## ComfyUI

Handles `KSampler`, `KSamplerAdvanced`, `SamplerCustom`, `SamplerCustomAdvanced`
(Flux) and custom sampler wrappers such as WanVideo. Follows conditioning through
pass-through nodes (`ControlNetApplyAdvanced` and friends) by output slot, so the
negative prompt no longer resolves to the positive one; resolves `steps`/`cfg` fed
by constant nodes; collects LoRAs from `LoraLoader`, LoraManager, rgthree Power
Lora Loader and WanVideo. In a two-pass graph the full-denoise sampler is
preferred, so you get the base generation parameters rather than the upscale pass.

> Only the `prompt` (API-format) chunk is parsed. Images carrying only a
> `workflow` chunk can't have their parameters reliably reconstructed and are
> left untouched.

## Resource lookup

Above the raw metadata, the PNG Info tab now lists every checkpoint and
`<lora:...>` the image used, and whether you have it:

```
Resources — 1/3 installed
  Checkpoint  nope                                 ✗ missing    search Civitai · HuggingFace
  LoRA        AnissaSeries_AnimaPreview3_byKonan   ✓ installed  AnissaSeries_AnimaPreview3_byKonan.safetensors
  LoRA        not_a_real_lora                      ✗ missing    search Civitai · HuggingFace
```

LoRAs resolve through the `networks` registry, falling back to a walk of
`--lora-dir` (following symlinks) so a shared LoRA folder still matches.
Missing entries get Civitai and HuggingFace search links, plus the file hash when
the metadata carried one — SwarmUI records a full SHA256 per model, which this
emits as A1111 `Model hash` / `Lora hashes` fields.

When a hash is known, the **Civitai by-hash API** identifies the file exactly, so
the row becomes the real model name linking to its page plus a direct download
link, instead of a name search. Both hash forms Civitai accepts work — AutoV2
(10 chars) and full SHA256 (64) — and 12-char A1111 LoRA hashes are skipped
because those are AddNet hashes of the tensor data, which Civitai does not index.

Lookups run in parallel, are cached per hash for the session, and time out after
5s. Any failure — offline, rate limited, region blocked (that API has been seen
answering `451 REGION_BLOCKED` on some connections), or a file simply not hosted
there — falls back to the search links. Turn it off in settings to stay offline.

## Hiding metadata in your own saves

Off by default. Turn on **Settings → AI PNGinfo → "Hide the parameters in saved
PNGs too"** and every PNG you save carries its A1111 parameters in the image's low
bits as well as its text chunks, so the settings survive re-upload.

| Setting | Meaning |
|---|---|
| Where to hide it | `alpha` holds one bit per pixel; `rgb` holds three and survives an alpha strip |
| Compress | gzip the payload, so more fits |

PNG only — JPEG and WebP compression destroys the low bits, so the hook skips
them rather than writing something unreadable. Hiding data changes each channel
by at most 1/255, which is invisible.

If `sd_webui_stealth_pnginfo` is still installed, this extension detects it and
does not write, since two writers would overwrite each other's payload. Remove
that extension to use this one's writer.

## Metadata-stripped images

Many sites strip PNG text chunks on upload. NovelAI works around this by hiding
its metadata in the image's low bits, and sd-webui's stealth pnginfo extension
does the same for A1111 text.

This extension reads that hidden payload itself, so a stripped NovelAI image still
converts with nothing else installed. All four variants are handled: alpha and RGB
channels, compressed and not. It is only attempted when the PNG chunks carry
nothing, so normal images cost no extra work.

Verified against real NovelAI V5 output with every chunk removed - prompt,
character captions, sampler, schedule, CFG, seed and size all come back.

## Notes

Toggle under **Settings → AI PNGinfo**. Restart the WebUI after installing.

This does everything `sd_webui_stealth_pnginfo` did — reading hidden payloads,
writing them, keeping the alpha channel out of img2img and upscalers, and making
the PNG Info drop target accept RGBA — plus the format conversion, in one place.
You do not need both. The converter is still published as
`shared.ai_pnginfo_convert` for anything else that wants it.

Image formats follow whatever the source tool writes. ComfyUI and NovelAI only
write PNG; A1111 JPEG/WebP metadata lives in EXIF and is already read by the host,
then passes through here untouched. The hidden-payload reader needs a lossless
image, so it works on PNG and lossless WebP but never on JPEG.

Run the self-check with:

```
python extensions/sd_webui_ai_pnginfo/test_pnginfo.py
```
