# Contributing

Thank you for improving VBVR-RL. Changes should preserve reproducibility across
training, checkpoint conversion, and VBVR-Pro evaluation.

## Development Setup

Create the locked Python 3.12 environment from the repository root:

```bash
uv sync --frozen
uv lock --check
uv sync --frozen --check
```

Use direct `.venv/bin/python`, `.venv/bin/torchrun`, and `.venv/bin/ruff`
commands for project work. Fish is a host prerequisite for the launchers.

## Before Opening a Pull Request

Run the project suite and the same Ruff checks as CI:

```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check --output-format=github .
.venv/bin/ruff format --check .
```

If you change a Fish launcher, also run `fish -n` on it. If you change a YAML
config, construct its matching Pydantic config and perform the cheapest
relevant runtime preflight. Training changes should include a bounded test or
smoke that demonstrates the intended optimizer behavior.

## Change Discipline

- Keep `README.md`, `docs/`, runnable configs, launchers, and `AGENTS.md` in
  sync when a data, checkpoint, training, or evaluation contract changes.
- Add focused tests for bug fixes and behavior changes.
- Preserve explicit provenance for model, dataset, sampler, media preparation,
  evaluator source, and scorer runtime.
- Keep generated artifacts under ignored paths such as `storage/`, `logs/`,
  `wandb/`, and `tmp/`.
- Do not commit model weights, datasets, videos, checkpoints, credentials,
  tokens, private endpoints, scheduler settings, or machine-specific paths.
- Do not vendor VBVR-EvalKit. Rule scoring must use an explicit external path
  and exact source fingerprint.
- Treat reference configs as executable documentation. Avoid speculative or
  unvalidated options in files presented as runnable.

## Documentation

Public guides should use repository-relative paths and generic distributed
terminology. Do not commit experiment diaries, presentation material, email
drafts, generated media, private planning notes, or machine-specific runbooks.
Promote only durable, reusable conclusions into the relevant public guide.

Use descriptive Markdown links for repository files and verify that local
links resolve. Document external prerequisites and licenses without implying
that they are bundled.

## Pull Request Checklist

- [ ] The change has a focused description and test plan.
- [ ] Project tests pass.
- [ ] Ruff lint and format checks pass.
- [ ] Changed launchers and configs pass syntax/config validation.
- [ ] User-facing docs match the implemented behavior.
- [ ] No secrets, private paths, external artifacts, or generated outputs are
      included.
- [ ] Evaluation or reward changes use a new provenance contract when they can
      affect scores.

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
