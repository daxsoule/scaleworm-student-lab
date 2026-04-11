# Literature

This directory holds reference literature used in this project.
**The PDFs themselves are not committed to git** (see `.gitignore` at the
project root). This README is the version-controlled record of what's in
the folder, where it came from, and what we're allowed to do with it.

## Provenance rule

Every PDF in this folder must have a row in the table below **before**
it is read or summarized for project work. If you cannot fill in every
column for a paper, the paper does not belong in this folder yet —
either find the missing information or keep the paper out.

## Subdirectory layout

| Subdirectory | What goes here | Claude usage |
|---|---|---|
| `open_access/` | CC-BY, CC-BY-NC, public domain, or otherwise unambiguously open papers | Free to summarize, paraphrase, and quote (with citation). Figures: check the specific CC variant — CC-BY-ND blocks derivative figures. |
| `library_subscription/` | Paywalled papers we can read via an institutional subscription | **Check the publisher's text/data-mining (TDM) clause before asking Claude to read.** Best for occasional, human-mediated lookups. Do not bulk-feed. |
| `author_copies/` | Papers shared directly by an author, your own papers, or authorized preprints (arXiv / EarthArXiv / ESSOAr) | Check the posting agreement — author copies are usually OK to read; arXiv preprints follow whatever license the author chose. |
| `embargoed/` | Manuscripts under review, NDA'd material, anything you should NOT hand to an AI | **Do not feed to Claude.** This folder exists as a forcing-function: if it's here, it's off-limits. |

## Inventory

| File | Citation | Source | License / access basis | Publisher TDM stance | Allowed Claude use | Date added |
|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | |

### How to fill in a row

- **File**: relative path from `literature/` (e.g., `open_access/smith_2024_eos.pdf`).
- **Citation**: full reference in your preferred style.
- **Source**: where you got it — `QC library`, `arXiv:2401.12345`, `author email 2026-04-08`, `journal website OA`, etc.
- **License / access basis**: `CC-BY 4.0`, `Elsevier subscription via QC`, `personal author copy`, `public domain (USGS)`, etc.
- **Publisher TDM stance**: `permitted`, `prohibited`, `silent`, or `unknown — do not feed to AI`. For Elsevier / Wiley / Springer Nature / ACS, default to `prohibited` unless you have specifically checked.
- **Allowed Claude use**: `full summarization OK`, `human-read only`, `do not feed`, `figures excluded`, etc.
- **Date added**: ISO date (YYYY-MM-DD).

## Scope of the no-commit rule

The `.gitignore` at the project root blocks `literature/**/*.pdf`,
`*.epub`, and `*.djvu`. This rule **only** applies to the `literature/`
tree. Generated artifacts under `outputs/` (methods PDFs, figure
montages, constitution renders) are deliberately allowed in version
control because they are reproducible from version-controlled scripts
and are not subject to external publisher licensing. The literature-vs-
outputs distinction is the whole point: external sources are governed
by license, internal artifacts are governed by the build pipeline.

## Hard rules

1. **Never commit an external PDF to `literature/`.** The `.gitignore`
   blocks the entire `literature/**/*.pdf` tree — do not override it
   with `git add -f`.
2. **Never feed `embargoed/` content to Claude.** No exceptions.
3. **Default to "do not feed" for paywalled content** until the
   publisher's TDM clause has been checked.
4. **No figure reproduction.** Even for open-access papers, do not have
   Claude regenerate figures from a copyrighted source. Cite and link
   instead, or generate a fresh figure from the underlying data.
5. **Cite everything.** Any project material derived from a paper in
   this folder must cite the paper, regardless of how heavily Claude
   was involved in drafting.
6. **AI-generation disclosure.** Per the user's global rule, any prose
   Claude writes for project use carries the standard italic disclosure
   label and Courier New / monospace styling.

## Why these rules exist

Short version:

- Storing legitimately-acquired PDFs locally is generally fine; **what
  you do with the output is where copyright and TOS issues live**.
- Several major publishers contractually prohibit feeding their content
  to AI systems, even when you have a legitimate subscription to read
  the PDF yourself.
- Anthropic does not train on Claude Code / API content by default, but
  the content still leaves your machine — so embargoed and NDA'd
  material should not go through here at all.
- AI-drafted text that paraphrases a paper too closely, or that omits
  citation, is plagiarism regardless of whether a human or a model
  wrote it. The disclosure label and citation are both required.
