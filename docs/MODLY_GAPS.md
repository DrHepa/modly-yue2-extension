# Modly compatibility and minimal host changes

Audited baseline: `lightningpixel/modly` **v0.4.2**. Relevant current `main` files were also read. No core repository has been modified and no issue or pull request has been submitted by this delivery.

## What already works at the contract level

The release declares process artifacts `image`, `text`, `mesh`, **audio**; process nodes can return `{filePath}` or `{text}`. It launches an extension's `venv` Python and declared entry using a one-line JSON payload. `select`, `int`, `float`, `string` and `file-select` UI parameter types exist. Text/file picker intents and multiple text input slots are present. The wrapper uses only existing types, so generating audio does not require inventing a `type:model` audio runner or modifying mesh inference.

## 1. Active Python cancellation — necessary to promise functional UI Cancel

**Source:** `electron/main/process-runner.ts`, `PythonProcessRunner.run()` and `terminate()`. The inspected release spawns `proc` as a local variable, while `terminate(): void {}` is empty. Registry termination calls that no-op method.

**Current impact:** the user may cancel a workflow while GPU inference continues. The wrapper can cooperate with an actual signal or a local `CANCEL` file, but cannot make a host no-op kill the child.

**Minimal robust host change:** retain active `ChildProcess` handles in the runner (prefer a set if concurrent runs remain possible), terminate them on cancel, enforce a grace period followed by forced termination, settle the associated promises once, clear state and remove listeners after exit/error. On Windows terminate the process tree; on Unix arrange a separate process group and terminate that group, because optional inference backends may spawn descendants. Never signal the host's own process group. Ensure registry removal does not orphan running children.

**Acceptance tests:** cancel during imports, ABC planning, semantic generation, NAR, VAE and a batch; test stop-before-ready, process already exited, repeated cancel, Windows descendants, error/close races, and application shutdown. Confirm GPU memory is released and no later done event marks a cancelled node as successful. The test doubles in this extension do not prove any of those host behaviors.

## 2. Explicit models-directory context — recommended small contract improvement

**Source:** process-runner payload contains `input`, `params`, `nodeId`, `workspaceDir` and `tempDir`, but not `modelsDir`. The extension setup protocol described in the audited installer references contains Python/platform/GPU fields, without guaranteed storage fields.

**Current workaround:** this extension resolves explicit paths/environment, host-bound `settings.json` and a tied setup record, in that order. It never waits for `/settings/paths` over HTTP and never guesses a storage folder from its name alone.

**Minimal host change:** resolve storage once from host settings and pass `modelsDir` (and preferably `extensionsDir` / `extensionDir`) into every setup invocation and process runtime context. Keep backward compatibility. Apply the same path logic to install, update, repair and any backend setup route. Pass a current path after the user moves storage, rather than a cached path from runner creation.

**Acceptance tests:** custom Windows dependencies drive, relocated models folder, portable profile, paths with spaces/non-ASCII, local-linked extension, two installations, Python 3.12 private host and repair after migration. No destructive directory migration is necessary in this proposal.

**Priority:** not a blocker for ordinary bound installations using this wrapper, but preferable to every process extension maintaining its own settings resolver.

## 3. Bounded live stderr forwarding — recommended for progress and diagnosis

**Source:** `PythonProcessRunner.run()` accumulates `stderrBuf` and mainly uses it at process close. Native YuE2 progress goes to stderr. The wrapper emits independent JSON stage/token logs, but the native NAR step stream is not guaranteed to be visible while running.

**Minimal host change:** forward decoded stderr lines to process logs as they arrive, handle CR-style progress lines and bound the retained error tail instead of accumulating an entire long batch indefinitely. Do not interpret arbitrary stderr text as successful protocol output. Retain the existing structured stdout channel.

**Acceptance tests:** UTF-8 split chunks, carriage-return progress, very long output, no newline, child crash, cancel while writing and secrets redaction where appropriate. This improves many Python process extensions, not just YuE2.

## 4. Rich multi-artifact output — optional, not required for current nodes

The inspected process contract has one typed output per node and `ProcessResult` only `filePath`/`text`. It does not define named simultaneously typed ports for audio + score + latents + metadata.

This wrapper stores durable files and sends a text bundle descriptor between stages. An audio-extraction node retrieves the score/descriptor from an existing song's folder. A future backward-compatible `artifacts[]` result and viewer could improve discoverability; a full typed multi-output workflow feature is larger than a minimal compatibility patch and should not block basic integration.

## 5. ABC / lyric editing and input selection — optional usability improvements

`pickerIntent` in the inspected release includes folder/image/mesh/text, but not audio or a dedicated ABC editor. Text controls are not a specialized multiline musical-score editor. The wrapper accepts direct text or explicit UTF-8 files; its fallback audio path accepts a native absolute path, and an audio connection is preferable.

A text editor or generic file-picker filter would improve this workflow. Do not claim unsupported manifest fields implement that UI today.

## Not host gaps

SheetSage2 transcription, ASR, an external editing agent, GPU BF16/FP8 support and platform-specific vLLM wheels are **not deficiencies in Modly's audio artifact type**. They require auxiliary integrations or compatible native runtimes/hardware. Changing Modly's manifest parser cannot make those components materialize.
