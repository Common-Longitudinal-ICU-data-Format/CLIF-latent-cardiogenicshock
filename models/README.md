# Fitted phenotype models

The three phenotype assignments, fitted once on the derivation cohort and
**transferred** to every contributing site. A site passes them to
`01_site_aggregate.py`:

```bash
python code/01_site_aggregate.py \
    --phenotype-model models/phenotype_model.json \
    --seymour-model   models/seymour_model.json \
    --bhavani-model   models/bhavani_model.json
```

These flags are **required at a contributing site**. Without them the site
refits its own clusters, and "phenotype II", "beta" and "group A" stop meaning
the same thing at each site — which makes every pooled breakdown
uninterpretable.

## Why these are in git when nothing else derived from patients is

They contain no patient data. Each holds a feature list, the derivation
cohort's median and standard deviation per feature, the cluster centroids in
that standardised space, and the label names. Those are aggregate statistics
over tens of thousands of patients, the same disclosure class as the count
tables — and unlike the count tables there are no small cells, because a median
over 34,142 patients is not a cell.

They have to be distributed somehow, and every other route (email, Box, a
wiki) loses the link between a model file and the code version that can read
it. In git, a site that checks out a commit gets the models that commit's code
expects.

## What each file carries, and why all of it has to travel

| field | why it is load-bearing |
|---|---|
| `features` | the variable set, including which were dropped for sparsity. A site that recomputed this against its own availability would cluster in a different space. |
| `centre`, `scale` | the derivation cohort's moments. A site standardising against its own would re-centre the space and move patients between phenotypes without anything clinical changing. |
| `log_transform` | which variables were log-transformed before standardising. |
| `z_clip` | the standard-deviation cap applied afterwards. |
| `centroids`, `labels` | the clusters and their deterministic names. |

`log_transform` is the one that bit. An earlier `save_model` enumerated the
payload and silently dropped it, so a receiving site applied no transform and
compared raw AST — which runs from 1 to the thousands — against centroids
fitted on `log1p(AST)`. Agreement with the correct assignment was **36.5%**,
and it put 56% of patients in delta against a correct 10%. `apply_model` now
**refuses** a file with no `log_transform` key rather than defaulting to none,
because the default is the wrong answer and it is wrong silently.

## Regenerating them

At the derivation site only:

```bash
python code/01_site_aggregate.py --stage cohort
cp output/cohort/{phenotype,seymour,bhavani}_model.json models/
```

Then commit, so that the models and the code that reads them move together.
