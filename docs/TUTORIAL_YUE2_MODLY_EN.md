# Practical YuE2 Tutorial for Modly

This guide explains the YuE2 nodes, how to connect them in Modly, and how to build a basic cover workflow.

Official references:

- [YuE2 Music Skill](https://github.com/multimodal-art-projection/YuE/tree/main/skills/yue2-music)
- [Generation and covers](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md)
- [Models, setup and audio-to-score](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/models-and-setup.md)
- [ABC editing](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/abc-editing.md)

## Before starting

1. Install the extension through **Modly → Extensions → Install from GitHub**.
2. Run **Repair/setup** and download the models through Modly's Models UI.
3. Run **Installation Diagnostics** first.
4. Use `device=cuda` on a compatible GPU. Full-song generation can take time and requires substantial VRAM.

Results are stored under `Workflows/YuE2/` in the workspace. Text-output nodes usually return a JSON **bundle descriptor**, not raw ABC. Use **Inspect Bundle** with `extract=abc` to obtain editable ABC text.

## What each node does

| Node | Purpose |
|---|---|
| **Generate Song** | Generates a song from lyrics and style. `cot=full` creates melody and harmony; `cot=melody` creates a melody without chord symbols; `cot=off` skips symbolic planning. |
| **Plan Score** | Creates and saves the ABC plan and native artifacts without rendering the final song. Use it before editing a composition. |
| **Render ABC / Cover / Edit** | Receives ABC through its text input and renders a new song using the lyrics and style parameters. Use `cot=melody` for a melodic cover and `cot=full` for reharmonization. |
| **Render Exact Saved Plan** | Resumes a saved plan without re-tokenizing its ABC. Do not use it after editing the ABC. |
| **Inspect Bundle / Agent Package** | Extracts `abc`, `request`, `metadata`, `descriptor`, `verify`, or an editing package from a bundle. |
| **Audio to Score / Artifacts** | Inspects audio or a YuE2 bundle that already exists. It does not transcribe an arbitrary external song. |
| **Import Native Saved Artifacts** | Imports a native artifact directory containing files such as `plan.json` or `latent.npy`. |
| **Plan to Semantic Tokens** | Continues from a saved plan and generates semantic tokens. |
| **Semantic Tokens to Latents** | Continues from semantic tokens and generates acoustic latents. |
| **Decode Audio Latents** | Converts saved latents to WAV/FLAC with the VAE decoder. |
| **Encode Audio** | Encodes a 48 kHz stereo WAV/FLAC into VAE latents. It is not transcription and does not create ABC. |
| **Batch Requests / Resume** | Runs JSON/JSONL requests sequentially and reuses completed batch items. |
| **Installation Diagnostics** | Checks Python, dependencies, models, and storage without generating music. |

## Basic editable-song workflow

```text
Lyrics/Text input
        │
        ▼
Generate Song (text → audio)
        │
        └── audio output
```

Recommended settings:

- `input_mode=lyrics`
- `cot=full`
- `style`: genre, instruments, language, vocal character, and approximate tempo
- `lyrics`: sectioned lyrics such as `[Verse]`, `[Chorus]`, and `[Bridge]`
- fixed `seed` when comparing versions

To keep the composition editable:

```text
Lyrics → Plan Score → Inspect Bundle (abc) → Render ABC / Cover / Edit → Audio
```

Never edit `plan.json`, `prefix.npy`, or `abc_tokens.npy` inside an exact saved plan. Copy `score.abc`, edit the copy, and submit it as a new Render ABC request.

## Basic cover workflow from melody ABC

This is the cover workflow that can be performed directly in this extension:

```text
Melody-only ABC ───────────────┐
                               ▼
                         Render ABC / Cover / Edit ──► Cover audio
                               ▲
                         lyrics + style
```

### Modly steps

1. Add a **Text/File input** node and load a melody-only `.abc` file.
2. Add **Render ABC / Cover / Edit**.
3. Connect the text output to the YuE2 text input.
4. Configure the YuE2 node:
   - `cot=melody`
   - `lyrics` with the lyrics to sing, or `lyrics_file` with a `.txt` file
   - `style` with the target genre, instruments, language, vocal character, and tempo
   - a fixed `seed` when comparing styles
5. Connect the audio output to a player or file-output node.
6. Run the workflow and check duration, lyrics, melody, transitions, and ending.

### What `cot=melody` means

`cot=melody` asks YuE2 to follow the supplied symbolic melody while freely generating a new accompaniment and vocal performance. It does not preserve the original singer, timbre, waveform, or recording.

The ABC should be primarily melodic and should not contain chord symbols when a new accompaniment is desired. Meter, tempo, key, note durations, and rests are part of the musical condition.

## Covering an existing audio recording

YuE2 does **not** accept a WAV as a direct cover reference. The **Encode Audio** node is a VAE encoder, not a melody transcriber.

The official route is:

```text
Source audio
      │
      ▼
SheetSage2 / external transcription
      │
      ▼
Review and correct score.abc
      │
      ▼
Remove chord symbols → melody-only.abc
      │
      ▼
Modly: Render ABC / Cover / Edit (cot=melody)
      │
      ▼
New cover with target style and lyrics
```

1. Transcribe the source audio with SheetSage2 using its full-melody task.
2. Review missed notes, timing, rests, meter, and key before blaming YuE2 for a bad result.
3. Export a chord-free melody ABC.
4. Load that `.abc` file into the Modly text input.
5. Provide the new lyrics and target style in **Render ABC / Cover / Edit**.
6. Generate and listen for melody adherence, pronunciation, transitions, and the ending.

This extension does not install SheetSage2 automatically because it has different dependencies and model weights. Keeping transcription and regeneration separate makes both stages inspectable and reproducible.

## Comparing several cover styles

Keep the same ABC, lyrics, and seed, and change only `style`:

```text
melody-only.abc
      ├── Render ABC: cinematic pop
      ├── Render ABC: acoustic folk
      └── Render ABC: synthwave
```

The melody condition is shared, but the performances will not be identical because YuE2 regenerates the complete song.

## Editing harmony or structure

Use a melody-and-harmony ABC with `cot=full`:

```text
Plan Score
   ▼
Inspect Bundle (abc)
   ▼
Edit a copy of score.abc
   ▼
Render ABC / Cover / Edit (cot=full)
   ▼
New complete version
```

Check that notes, durations, meter, and tempo were not changed accidentally. ABC editing creates a new complete recording; it is not waveform inpainting and does not preserve untouched audio samples.

## Common problems

- **The node receives audio but expects text**: use transcribed ABC for a cover, not the WAV.
- **Lyrics are ignored**: Render ABC uses its text input for ABC; lyrics belong in `lyrics` or `lyrics_file`.
- **A JSON descriptor appears instead of ABC**: connect it to **Inspect Bundle** and select `extract=abc`.
- **The cover loses the melody**: review the ABC and use `cot=melody`; `cot=full` may re-plan harmony.
- **An exact plan was edited**: copy `score.abc` and use Render ABC instead.
- **The song is too short**: increase token limits within the available VRAM and inspect `result.json` for truncation.

## Real-test checklist

- [ ] Preserve the source audio.
- [ ] Save and review the raw transcription.
- [ ] Save a chord-free melody ABC.
- [ ] Record lyrics, style, seed, and decoder settings.
- [ ] Generate the cover with `cot=melody`.
- [ ] Check duration and truncation.
- [ ] Listen to melody, lyrics, transitions, and ending.
- [ ] Keep ABC, request, metadata, and audio together.
