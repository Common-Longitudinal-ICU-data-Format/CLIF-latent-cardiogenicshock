#!/bin/bash
#
# Preflight: resolve every path in paths.yaml and check the model directory
# is intact, before queueing anything. Safe on a login node -- it reads
# config and lists directories, and opens no CLIF table.
#
#     bash hpc/00_preflight.sh
#
# The check that matters is the model glob. The weights directories are
# named with a U+25A1 WHITE SQUARE between "GEM" and the dataset -- GEM□-all,
# not GEM-all -- and `home_glob` matches it with a single-character wildcard,
# `GEM?-all`. A copy that dropped the character, expanded it, or replaced it
# with a multi-character escape leaves a glob that matches nothing or matches
# twice, and the failure surfaces much later as a confusing path error.

set -uo pipefail
cd "$(dirname "$0")/.."

# Deliberately does NOT source _common.sh: that does a module purge/load,
# which is the wrong thing to do to a login shell you are only using to
# check config. Activate the venv if it is where _common.sh expects it,
# otherwise fall back to whatever python is on PATH.
VENV_HOME="${VENV_HOME:-$HOME/venvs/clif-cs}"
if [ -f "$VENV_HOME/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_HOME/bin/activate"
elif [ -f "$VENV_HOME/Scripts/activate" ]; then
    source "$VENV_HOME/Scripts/activate"
else
    echo "note: no venv at $VENV_HOME; using $(command -v python)"
fi

python - <<'PY'
import sys, pathlib
sys.path.insert(0, "code")

ok = True

def report(label, value, good, note=""):
    global ok
    mark = "ok  " if good else "FAIL"
    if not good:
        ok = False
    print(f"  [{mark}] {label:<16} {value}")
    if note:
        print(f"         {note}")

try:
    from clifcs import load_paths
    paths = load_paths()
except Exception as exc:
    print(f"  [FAIL] paths.yaml did not load: {exc}")
    raise SystemExit(1)

print("sites")
for site in paths.sites:
    d = pathlib.Path(paths.raw_site_data[site])
    tables = sorted(p.stem for p in d.glob("clif_*.parquet")) if d.is_dir() else []
    report(site, d, d.is_dir(), f"{len(tables)} clif_* tables"
           if tables else "no clif_*.parquet found -- wrong directory?")

print("model")
try:
    home = paths.model_home
    config = pathlib.Path(home) / "config.json"
    weights = pathlib.Path(home) / "model.safetensors"
    report("checkpoint", home, pathlib.Path(home).is_dir())
    report("config.json", config.name, config.is_file())
    report("model.safetensors", weights.name, weights.is_file(),
           f"{weights.stat().st_size / 2**20:.0f} MiB" if weights.is_file() else "")
except Exception as exc:
    report("checkpoint", "unresolved", False, str(exc))
    # the glob is the usual culprit; show what is actually on disk
    import re
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("code/config/paths.yaml")
    pattern = str(cfg.models[cfg.default_model].home_glob)
    root = pathlib.Path(pattern.split("GEM?")[0])
    if root.is_dir():
        print(f"         directories under {root}:")
        for entry in sorted(root.iterdir()):
            cps = " ".join(f"U+{ord(c):04X}" for c in entry.name if ord(c) > 127)
            print(f"           {entry.name!r}  {cps or '(all ASCII)'}")
        print("         the glob needs EXACTLY ONE character where ? is.")

print("tokenizer")
tok = pathlib.Path(paths.tokenizer_yaml)
report("tokenizer.yaml", tok, tok.is_file())

print("writable locations")
for label, target in (("output_home", paths.output_home),
                      ("scratch_home", paths.scratch_home)):
    target = pathlib.Path(target)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write-probe"
        probe.touch()
        probe.unlink()
        writable = True
        note = ""
    except Exception as exc:
        writable = False
        note = str(exc)
    report(label, target, writable, note)

if pathlib.Path(paths.output_home).resolve().is_relative_to(
    pathlib.Path(paths.scratch_home).resolve()
) if pathlib.Path(paths.scratch_home).exists() else False:
    print("  [WARN] output_home is inside scratch_home. .id_salt lives under")
    print("         output_home and is unrecoverable if scratch is purged.")

print()
print("PASS" if ok else "FAIL -- fix the above before submitting a job")
raise SystemExit(0 if ok else 1)
PY
