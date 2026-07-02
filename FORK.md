# FIE Fork of LTX-2

This is the [fie.us](https://fie.us) fork of [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2),
maintained as a **light patch stack** on top of upstream snapshots. The fork exists primarily to
support high-quality / professional I/O (high bit depth, full-chroma, tensor-native conditioning)
that upstream does not expose.

## Branch model

| Branch | Role | Rules |
|---|---|---|
| `main` | Pristine mirror of upstream `main` | Fast-forward only; never commit here |
| `fork/*`, `feat/*` | The patch stack — one small topic branch per change | Rebased onto `main` after each upstream snapshot |
| `fie-main` | **Default branch.** Generated integration branch = `main` + all patches merged | Rebuilt (force-pushed) after each snapshot; never do original work here |

Upstream publishes as periodic "Public sync" snapshot merges, not live history, so pipeline
internals can change wholesale between snapshots. The patch stack survives this by design:
patches live in the utility/I-O layer as additive changes, avoiding pipeline `__call__` bodies
(the highest-churn surface).

## Patch stack

Each active patch is tracked by an open issue on this fork (label `active-patch`), with the
patch branch linked under the issue's **Development** field (a ref pointer, so it survives
rebases and force-pushes). The issue is closed when the patch is retired (absorbed upstream,
dropped, or superseded). We deliberately do not use PRs: patches are perpetually rebased and
re-merged, which the PR merged/closed lifecycle misrepresents. Never open a PR from a linked
branch — merging it would auto-close the tracking issue.

| Branch | Issue | What it does |
|---|---|---|
| `fork/docs` | — | This file + README banner (bottom of stack) |
| `feat/tensor-conditioning-inputs` | [#1](https://github.com/leith-bartrich/LTX-2/issues/1) | Conditioning inputs accept pre-decoded tensors everywhere a path is accepted, bypassing the 8-bit file decode (see `packages/ltx-pipelines/README.md` § Pre-Decoded Tensor Inputs) |

## Versioning & using as a dependency

Releases are annotated tags on `fie-main`: `fie-v1`, `fie-v2`, …
Tags are immutable and keep their commits alive even after `fie-main` is rebuilt, so pins stay
valid forever. Each release below records its upstream base and included patches.

Pin with uv:

```toml
[tool.uv.sources]
ltx-core = { git = "https://github.com/leith-bartrich/LTX-2.git", tag = "fie-v1", subdirectory = "packages/ltx-core" }
ltx-pipelines = { git = "https://github.com/leith-bartrich/LTX-2.git", tag = "fie-v1", subdirectory = "packages/ltx-pipelines" }
```

or pip:

```text
ltx-pipelines @ git+https://github.com/leith-bartrich/LTX-2.git@fie-v1#subdirectory=packages/ltx-pipelines
```

> Always pin `ltx-core` and `ltx-pipelines` to the **same tag** — they are developed in lockstep
> in this workspace.

### Releases

| Tag | Upstream base | Patches |
|---|---|---|
| _(none yet)_ | | |

## Maintenance workflow (per upstream snapshot)

```bash
# 1. Update the pristine mirror
git fetch origin                       # origin = Lightricks/LTX-2
git push fork origin/main:main         # fork = this repository

# 2. Rebase each patch branch
git rebase origin/main fork/docs                        && git push -f fork fork/docs
git rebase origin/main feat/tensor-conditioning-inputs  && git push -f fork feat/tensor-conditioning-inputs

# 3. Rebuild the integration branch
git checkout -B fie-main origin/main
git merge --no-ff fork/docs
git merge --no-ff feat/tensor-conditioning-inputs
git push -f fork fie-main

# 4. (When releasing) tag the verified rebuild
git tag -a fie-vN -m "fie-vN: upstream <sha> + patch stack" && git push fork fie-vN
```

If a rebase conflicts, fix it in the topic branch — never patch over it in `fie-main`.
New work started on `fie-main` must be extracted into a topic branch before the next rebuild.
