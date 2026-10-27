# Buddy Test Report

| | |
|---|---|
| **Buddy site / institution** | *UCMC* |
| **Tester** | *Kaveri Chhikara* |
| **Date** | *2026-10-26* |

## Environment

| | |
|---|---|
| **OS** | *macOS 26.7.1* |
| **RAM** | *64 GB* |
| **Python** | *3.12.11* |


###  Suggestions
1. **Published rates can give back counts below 11.** Mortality and share columns ship whenever their denominator is at least 11, but rate × denominator recovers numerators of 1-10, including deaths among treated and untreated patients, whose group sizes are never checked. The export check only inspects count columns (`trial_aggregate.py`, `baseline.py`, `export/contract.py`).
2. **Seymour and Bhavani are refit at every site.** The script builds the sepsis model list before it fills in the shipped model paths, so only the Zweck model is applied, although the log says all three are. Labels are therefore not comparable across sites (`01_site_tables.py`, `main()`). Hanna noted this does not matter for this run request.
3. **Bhavani transfer path.** When the shipped Bhavani model is applied, the transfer path skips the eligibility rule and carry-forward the model was fitted with; fix it together with the item above (`sepsis_phenotype.py`, `attach()`).
4. **SOFA window.** SOFA is the worst value over the whole stay, with missing components scored 0, but is used as a baseline severity adjuster; consider a pre-onset window (`01_site_tables.py`, `stage_derive()`).
5. **Memory at large sites.** `derive` holds whole tables in pandas and peaked at about 17.5 GB on the test site; larger sites such as UMN and JHU will likely run out of memory. Suggest processing hospitalizations in shards (`01_site_tables.py`, `stage_derive()`).
6. **Avoidable memory use.** The trials stage re-reads the full raw tables instead of the cohort subset, procedures are collected into Python sets over the whole table, and two clifpy orchestrators are held in memory at once (`01_site_tables.py`, `cohort.py`).
7. **HPC defaults.** The HPC scripts default to one user's cluster paths and site (`hpc/_common.sh`).

### Fixes made during the buddy test (`kc-review`)
1. **Setup instructions** (`README.md`). the README now finds a 3.11+ interpreter first and names that error.
2. **Template config crashed on import** (`code/clifcs/config.py`). Copying `paths.example.yaml` and setting only `data_dir`, as instructed, crashed with "Read-only file system: '/path'"; a placeholder `scratch_home` now counts as unset, as it already did elsewhere.
3. **No adult filter** (`code/clifcs/cohort.py`, `base_cohort()`). The study is adults only, but age was never filtered. Added `min_age_years: 18` to `cohort.yaml`, applied before the first-hospitalization pick, with its own row in the patient-flow table.
4. **Lowercase category values** (`code/clifcs/clif.py` and six modules). Mortality and sex were matched against "Expired", "Male" and "Female" exactly, so a site sending lowercase values got 0% mortality and no eGFR; matching is now case-insensitive, and the run stops with the values it saw if none matches the mortality list.
5. **SOFA scores belonged to other patients** (`code/01_site_tables.py`, `derive` stage). The per-encounter summary was joined to clifpy's SOFA table by row position instead of by `encounter_block`, so every patient carried another patient's score and the assignment changed between runs. After the fix, SOFA differs by shock type and predicts in-hospital mortality as expected. The upstream coreopsis SOFA recipe has the same join.
6. **Cohort files not shipped, and stored twice** (`code/01_site_tables.py`, `code/clifcs/export/writer.py`).  Each stage now writes its tables once, straight into `upload_to_box_<SITE>/`, already suppressed and checked, and `output/cohort` keeps only patient-level parquet files; `seymour_agreement.csv`'s count column was renamed from `len` to `n` so its small cells are suppressed, and refit model files are checked against a field list.
7. **Achieved MAP window** (`code/clifcs/trials.py`). The "24 h" achieved MAP used by the MAP-target trials was computed from vitals within 6 h of onset; it now reads the full 24 h.
8. **Row-order dependence** (`code/clifcs/trials.py`). Heart rate, DBP and temperature at onset depended on input row order when two readings tied for the lowest MAP; ties now go to the earliest reading.
9. **Baseline lab source** (`code/clifcs/baseline.py`). The baseline lab rows read the cohort's nearest-in-window values instead of the documented at-onset values, because of a column-name collision in a join; the at-onset definition covers fewer patients, so the authors may prefer to change the documentation instead.
10. **Timestamps without a timezone** (`code/01_site_tables.py`, `code/clifcs/clif.py`). These were read as UTC without any check; `--check` now warns about them, and the run stops if DuckDB is not on UTC. The DuckDB timezone helper this uses had never been called and failed on duckdb 2.x, and is fixed too.

