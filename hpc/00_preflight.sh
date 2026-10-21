#!/bin/bash
#
# Preflight: resolve every path in paths.yaml and check this machine has what
# the run needs, before starting anything. Safe on a login node -- it reads
# config and lists directories, and opens no CLIF table. Worth running whether
# or not you have Slurm; it is the cheapest way to catch a misconfiguration.
#
#     bash hpc/00_preflight.sh
#
# For TIER 1 (01_site_tables.py) the checks that matter are the CLIF tables and
# `tokenizer_yaml`. The model glob is reported but cannot fail the preflight:
# tier 1 never loads the weights.
#
# For TIER 2 the model glob is the one people get wrong. The weights
# directories are named with a U+25A1 WHITE SQUARE between "GEM" and the
# dataset -- GEM□-all, not GEM-all -- and `home_glob` matches it with a
# single-character wildcard, `GEM?-all`. A copy that dropped the character,
# expanded it, or replaced it with a multi-character escape leaves a glob that
# matches nothing or matches twice, and the failure surfaces much later as a
# confusing path error.

set -uo pipefail
# Run directly (`bash hpc/00_preflight.sh`), never submitted, so unlike
# the .sbatch files $0 really is this file and dirname is safe here.
cd "$(dirname "$0")/.."

# Deliberately does NOT source _common.sh: that does a module purge/load,
# which is the wrong thing to do to a login shell you are only using to
# check config. Activate the venv if it is where _common.sh expects it,
# otherwise fall back to whatever python is on PATH.
# Keep this in step with hpc/_common.sh. It is duplicated rather than
# sourced because _common.sh does a module purge, which is the wrong
# thing to do to a login shell.
VENV_HOME="${VENV_HOME:-/gpfs/data/pearson-lab/PERSONAL/hhieromnimon/venvs/clif-cs}"
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


def note_only(label, value, note=""):
    """report something that is informational -- does not set the exit code"""
    print(f"  [--  ] {label:<16} {value}")
    if note:
        print(f"         {note}")

try:
    from clifcs import load_paths
    paths = load_paths()
except Exception as exc:
    print(f"  [FAIL] paths.yaml did not load: {exc}")
    raise SystemExit(1)

# Every table the pipeline reads as INPUT. The _processed, _converted and
# clif_sofa tables are derived into scratch by the `derive` and `collate`
# stages and are deliberately not listed here.
#
# Counting clif_*.parquet is not the check: UCMC and NU both hold 19 tables
# and they are not the same 19 -- the union is 21, because each site ships
# extras (bal, clinical_trials, intermittent_hemodialysis, the microbiology
# tables) that this pipeline never opens. What matters is the required set.
REQUIRED = [
    "clif_adt",
    "clif_code_status",
    "clif_crrt_therapy",
    "clif_hospital_diagnosis",
    "clif_hospitalization",
    "clif_labs",
    "clif_medication_admin_continuous",
    "clif_medication_admin_intermittent",
    "clif_patient",
    "clif_patient_assessments",
    "clif_patient_procedures",
    "clif_position",
    "clif_respiratory_support",
    "clif_vitals",
]

#: absent, these degrade the run quietly rather than stopping it, so each
#: gets its own line saying what goes wrong
DEGRADES_QUIETLY = {
    "clif_hospital_diagnosis":
        "every diagnosis-based trial criterion reads as not met; "
        "eligibility and shock type are understated, not errored",
    "clif_patient_procedures":
        "IABP, ECMO and microaxial flags all read false, so the device "
        "trials' concordance is silently wrong rather than absent",
    "clif_crrt_therapy":
        "renal support is invisible to the tokenizer",
    "clif_code_status":
        "limitation-of-treatment context is lost",
}

print("sites")
for site in paths.sites:
    d = pathlib.Path(paths.raw_site_data[site])
    if not d.is_dir():
        report(site, d, False, "directory not found")
        continue
    present = {f.stem for f in d.glob("clif_*.parquet")}
    missing = [t for t in REQUIRED if t not in present]
    report(site, d, not missing,
           f"all {len(REQUIRED)} required tables present "
           f"({len(present)} total in directory)" if not missing
           else f"MISSING {len(missing)} of {len(REQUIRED)} required")
    for table in missing:
        note = DEGRADES_QUIETLY.get(table)
        print(f"           - {table}")
        if note:
            print(f"               {note}")
    extra = sorted(present - set(REQUIRED))
    if extra:
        print(f"         also present, never read: {', '.join(extra)}")

# TIER 2 ONLY. 01_site_tables.py never loads the weights -- it reads the
# tokenizer, checked below -- so an unresolved glob here is reported but does
# not fail the preflight. A site asked only for the aggregate tables can leave
# `models.<key>.home_glob` pointing at nothing.
print("model  (tier 2 only -- 01_site_tables.py does not read this)")
try:
    home = paths.model_home
    config = pathlib.Path(home) / "config.json"
    weights = pathlib.Path(home) / "model.safetensors"
    report("checkpoint", home, pathlib.Path(home).is_dir())
    report("config.json", config.name, config.is_file())
    report("model.safetensors", weights.name, weights.is_file(),
           f"{weights.stat().st_size / 2**20:.0f} MiB" if weights.is_file() else "")
except Exception as exc:
    note_only("checkpoint", "unresolved",
              "fine if you are running tier 1 only; 02_site_embeddings.py and "
              "08_token_space.py will refuse to start until it resolves")
    print(f"         {exc}")
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

# TOKEN STAGES ONLY. Needed to tokenize and to build the token-prevalence
# tables, which are token ids and are only comparable across sites under the
# training-time mapping. Without it 01 skips `tokenize`, `winnow` and
# `tokencounts` and exports everything else, so this cannot fail the preflight
# either -- the cohort, phenotype, baseline and trial tables never see a token.
print("tokenizer  (token stages and tier 2 -- the rest of 01 runs without it)")
tok = paths.tokenizer_yaml_or_none
if tok is not None:
    report("tokenizer.yaml", tok, True)
else:
    note_only("tokenizer.yaml", paths.tokenizer_yaml_path or "not configured",
              "01 will SKIP tokenize, winnow and tokencounts, so the bundle "
              "carries no token_prevalence, no token_families and no "
              "token_levels. Everything else still exports.")
    print("         it must be the tokenizer the model was TRAINED with -- a "
          "locally fitted one maps the same clinical event to a different id")

# Both default to the working directory when unset or left as a placeholder,
# so these lines say where the run will actually write -- which is the point,
# since ./scratch under a repo on a synced drive is a working default and a bad
# idea for a large cohort.
print("writable locations  (default to ./output and ./scratch when unset)")
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
