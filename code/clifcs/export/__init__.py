#!/usr/bin/env python3

"""
the site boundary: what may leave, how it is checked, and how it is read back

`contract`   the allowlist of columns permitted to cross a site boundary
`suppress`   small-cell suppression for the aggregate tables
`manifest`   what a bundle says about the code, model and tokenizer that made it
`writer`     writing the two tiers of a bundle, at a contributing site
`bundles`    discovery and validation of received bundles, at the pooled site
"""

from clifcs.export import bundles, contract, manifest, suppress, writer

__all__ = ["bundles", "contract", "manifest", "suppress", "writer"]
