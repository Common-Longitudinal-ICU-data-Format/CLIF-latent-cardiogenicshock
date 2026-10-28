# OMspell dissimilarities for handbook discovery, called from Python.
#
# WHY THIS EXISTS. Handbook discovery used a semi-global alignment written in
# this project. Benchmarked against thirteen TraMineR dissimilarities on a test
# set built from the phenotypes we expect plus four hand-written examples, that
# aligner scored worst of the fourteen: 58% nearest-neighbour purity with a
# NEGATIVE margin, meaning a low-pressor course perturbed by one spike or one
# off-pressor hour was more often nearer some other phenotype than its own clean
# sibling. OMspell scored 100% purity, the widest margin (0.54), and was one of
# only two measures that grouped all four hand-written examples.
#
# WHY OMspell SPECIFICALLY. It edits SPELLS (runs) rather than hours, and prices
# a spell's duration separately through `expcost`. Duration is therefore
# discounted rather than discarded -- between raw hourly distance, where a long
# plateau drowns out the order of events, and a full run-length collapse, where
# an eight-hour plateau and a one-hour blip are the same character. A one-hour
# artefactual spike becomes a cheap short spell instead of a mismatched hour.
#
# `expcost` is the duration-versus-order knob and lives in config/dynamics.yaml,
# like every other threshold in this project.
#
# Two modes:
#   pairwise  square matrix within one set, for clustering a sample
#   refs      between two sets, for assigning everyone to the k medoids
# Reading and writing plain CSV is deliberate: it keeps the R side stateless so
# a failed run leaves nothing behind to be mistaken for a result.

lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library",
                 paste(R.version$major, substr(R.version$minor, 1, 1), sep = "."))
if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(TraMineR))

args <- commandArgs(trailingOnly = TRUE)
opt <- list(mode = "pairwise", expcost = 0.5, indel = 1, norm = "auto",
            sm = "TRATE")
for (a in args) {
  kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
  opt[[kv[1]]] <- if (length(kv) > 1) kv[2] else TRUE
}
stopifnot(!is.null(opt$input), !is.null(opt$output), !is.null(opt$alphabet))
expcost <- as.numeric(opt$expcost)
indel <- as.numeric(opt$indel)

alphabet <- strsplit(opt$alphabet, "", fixed = TRUE)[[1]]
raw <- read.csv(opt$input, colClasses = "character")
stopifnot("sequence" %in% names(raw))

# One column per hour. Short courses pad with NA, which seqdef reads as void;
# right = "DEL" then drops the trailing voids rather than treating "this patient
# was discharged" as a state they were in.
width <- max(nchar(raw$sequence))
cells <- do.call(rbind, lapply(raw$sequence, function(s) {
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  length(chars) <- width
  chars
}))
seqs <- seqdef(cells, alphabet = alphabet, right = "DEL", xtstep = 6)

# Substitution costs measured from observed transition rates, not assumed. This
# is the same reasoning as the project's Jensen-Shannon substitution matrix:
# two states are similar if what follows them is similar.
costs <- if (identical(opt$sm, "CONSTANT")) {
  seqsubm(seqs, method = "CONSTANT", cval = 2)
} else {
  seqsubm(seqs, method = "TRATE")
}

if (identical(opt$mode, "refs")) {
  refs <- as.integer(strsplit(opt$refs, ",", fixed = TRUE)[[1]])
  rest <- setdiff(seq_len(nrow(raw)), refs)
  d <- seqdist(seqs, method = "OMspell", sm = costs, indel = indel,
               expcost = expcost, norm = opt$norm,
               refseq = list(refs, rest))
  # Return a full n x k block so Python does not have to reconstruct the row
  # order: column j is medoid refs[j], row i is sequence i of the input.
  full <- matrix(NA_real_, nrow = nrow(raw), ncol = length(refs))
  block <- as.matrix(d)                       # refs x rest
  full[rest, ] <- t(block)
  for (j in seq_along(refs)) full[refs[j], j] <- 0
  # the remaining ref-to-ref cells come from a small square among the refs
  if (length(refs) > 1) {
    among <- as.matrix(seqdist(seqs[refs, ], method = "OMspell", sm = costs,
                               indel = indel, expcost = expcost,
                               norm = opt$norm))
    full[refs, ] <- among
  }
  write.table(full, opt$output, sep = ",", row.names = FALSE,
              col.names = FALSE)
  cat("wrote", nrow(full), "x", ncol(full), "distances to", length(refs),
      "references\n")
} else {
  d <- as.matrix(seqdist(seqs, method = "OMspell", sm = costs, indel = indel,
                         expcost = expcost, norm = opt$norm))
  write.table(d, opt$output, sep = ",", row.names = FALSE, col.names = FALSE)
  cat("wrote", nrow(d), "x", ncol(d), "pairwise distances\n")
}
