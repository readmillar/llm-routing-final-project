# Runbook

## Default Submission Path

Install dependencies once:

```bash
make install
```

Acceptance commands for the default submission path:

```bash
make test
make run-smoke
make run-final
make audit
make plots
make test
```

The default path includes A0, A1, A2, A3, solver diagnostics, solution audit,
report comparison tables, and required figures.

`make run-smoke` writes artifacts to `outputs_smoke.tmp` first and promotes that
directory to `outputs_smoke` only after the smoke experiment succeeds.

`make run-final` writes artifacts to `outputs_final.tmp` first and promotes that
directory to `outputs_final` only after the final experiment succeeds. If the
experiment command fails, the existing `outputs_final` directory is left in
place.

Final artifacts are written to `outputs_final/tables`, `outputs_final/figures`,
and `outputs_final/solutions`.

## Extension Flags

The following extension flags are present in `config/final.yaml` and
`config/smoke.yaml`, and are disabled by default:

- `features.empirical_recovery`
- `features.a4_cvar`
- `features.three_stage`
- `features.lexicographic_a3`
- `features.provider_storage_constraints`
- `features.stress_tests`

Extension outputs should only be included in the final report when they are
stable, audited, and easy to explain.
