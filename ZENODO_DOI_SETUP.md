# Zenodo DOI Setup

This repository is now connected to Zenodo and has a published archive record.

Current repository:

- GitHub: `https://github.com/syedofc/project-synapse`
- Current tagged release: `v0.3.2`
- Zenodo concept DOI: `10.5281/zenodo.20544669`
- Zenodo version DOI for `v0.3.2`: `10.5281/zenodo.20544670`

## Current Metadata Status

- `CITATION.cff` is present and was validated locally against CFF schema `1.2.0`.
- The repository already has named authorship, license metadata, and a public release.
- No `.zenodo.json` file is included.

That is intentional. According to Zenodo's current documentation, Zenodo will use `CITATION.cff` metadata for GitHub release archiving unless a `.zenodo.json` file is present. If both files exist, Zenodo will ignore `CITATION.cff` and use `.zenodo.json` instead.

For this repository, `CITATION.cff` is sufficient and keeps the metadata simpler to maintain across GitHub and Zenodo.

## Current State

The main Zenodo setup work is complete for `v0.3.2`.

Recommended repository-facing use:

1. use the **concept DOI** in the README badge and repository homepage,
2. use the **version DOI** when citing the exact `v0.3.2` release,
3. keep `CITATION.cff` aligned with the latest archived release.

## Future-Version Workflow

### 1. Publish A New GitHub Release

For any future tagged release:

1. create the new GitHub tag and release,
2. wait for Zenodo to ingest the release,
3. verify that a new version DOI appears under the same concept DOI,
4. update `CITATION.cff` and the README badge if needed.

### 2. If The New Release Does Not Appear

If a new tagged release does not appear in Zenodo after sync:

1. use Zenodo's GitHub integration page and `Sync now`,
2. confirm the repository is still enabled,
3. check whether the GitHub release was published after the integration remained active.

## Recommended Post-Mint Repository Edits

Once a DOI is minted, the best public-facing polish is:

1. add a badge near the top of `README.md`,
2. add a short `Cite this software` snippet with the DOI URL,
3. set the GitHub repository homepage to the Zenodo concept DOI landing page.

## Official References

- Zenodo GitHub integration overview: `https://help.zenodo.org/docs/github/`
- Enable a repository: `https://help.zenodo.org/docs/github/enable-repository/`
- CITATION.cff support: `https://help.zenodo.org/docs/github/describe-software/citation-file/`
- Zenodo JSON file behavior: `https://help.zenodo.org/docs/github/describe-software/zenodo-json/`
- Archive a release from GitHub: `https://help.zenodo.org/docs/github/archive-software/github-upload/`
