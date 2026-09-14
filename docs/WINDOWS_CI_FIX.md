# Windows path-identity CI fix — 2026-09-14

## Observed failure

The initial workflow run [34834047717](https://github.com/DrHepa/modly-yue2-extension/actions/runs/34834047717), for commit `36f8a7a2d2677923a847e6c196cf1db81d0b3144`, passed all four Linux x86_64/ARM64 jobs and failed both Windows jobs (Python 3.11.9 and 3.12.10).

Both Windows logs report 52 tests, four failures and one error. Dependency installation and source compilation succeeded. The failing tests compared the temporary profile path `C:/Users/RUNNER~1/...` with the resolved path `C:/Users/runneradmin/...`.

Four failures were noncanonical expected paths in storage/bundle assertions. The error also exposed a production issue: saved setup state compared `extension_root` as a raw string against a resolved root, rejecting equivalent Windows path spellings with `MODELS_DIR_MISSING`.

## Changes

- Canonicalize the saved absolute `extension_root` before comparing Path objects. Invalid/unresolvable saved paths do not bind to this host. The existing models-directory separation checks and settings precedence remain intact.
- Resolve expected paths in the five existing storage/bundle assertions without removing the original raw temporary-path inputs.
- Add seven regression tests: equivalent absolute spellings, raw tempfile paths, malformed/relative roots, unrelated directories, model-directory overlap, native Windows case/separator variants, and real 8.3 aliases obtained with `GetShortPathNameW`.

The native short-name test skips only when not on Windows or when the volume does not expose a distinct 8.3 alias. Unexpected Win32 API errors fail the test. Artifact symlink/reparse/traversal checks are unchanged. No dependencies, checkpoint locations, model revisions, node capabilities or CI matrix entries were changed.

## Validation at patch preparation

On Linux x86_64 with CPython 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5 and SoundFile 0.13.1:

- The new equivalent-path regression reproduced the original rejection before applying the production fix.
- The full suite completed: **59 discovered, 57 passed, 2 Windows-only skips, no failures/errors**.
- Source compilation, Python 3.11 AST syntax parsing and `git diff --check` passed.

These local checks are not native Windows or CUDA validation. The unchanged six-job GitHub Actions matrix validates the patch on Python 3.11/3.12 for Windows x86_64, Linux x86_64 and Linux ARM64. Consult the run attached to the fix commit for its final outcome; this note does not predeclare that outcome. Full-model inference and target-machine installation remain separate acceptance tasks.
