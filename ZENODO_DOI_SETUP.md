# Zenodo DOI Setup

This repository is ready for Zenodo archiving through the GitHub integration.

Current repository:

- GitHub: `https://github.com/syedofc/project-synapse`
- Current tagged release: `v0.3.0`

## Current Metadata Status

- `CITATION.cff` is present and was validated locally against CFF schema `1.2.0`.
- The repository already has named authorship, license metadata, and a public release.
- No `.zenodo.json` file is included.

That is intentional. According to Zenodo's current documentation, Zenodo will use `CITATION.cff` metadata for GitHub release archiving unless a `.zenodo.json` file is present. If both files exist, Zenodo will ignore `CITATION.cff` and use `.zenodo.json` instead.

For this repository, `CITATION.cff` is sufficient and keeps the metadata simpler to maintain across GitHub and Zenodo.

## Exact Next Steps

### 1. Connect Zenodo to GitHub

In Zenodo:

1. Open your profile menu and go to `GitHub`
2. Click `Sync now`
3. Find `syedofc/project-synapse`
4. Toggle the repository on

### 2. Trigger Archival

After the repository is enabled in Zenodo:

1. Go back to the Zenodo `GitHub` page
2. Select `project-synapse`
3. Use Zenodo's release workflow to archive a GitHub release

### 3. If `v0.3.0` Does Not Appear

Zenodo's documentation says that once connected, **new releases** are automatically ingested and archived.

Inference:

If Zenodo does not pick up the already-published `v0.3.0` release after enabling the repository, create a fresh GitHub release after the integration is active.

Recommended options:

1. wait for the next meaningful code or benchmark release, or
2. create a small archival patch release such as `v0.2.3` if you want the DOI immediately.

### 4. After the DOI Is Minted

Update the repository with:

1. a Zenodo DOI badge in `README.md`,
2. the DOI URL in the GitHub repository website field,
3. the DOI in `CITATION.cff` if you want the citation metadata to point directly to the archived software record.

## Recommended Post-Mint Repository Edits

Once you have a DOI, the best public-facing polish is:

1. add a badge near the top of `README.md`,
2. add a short `Cite this software` snippet with the DOI URL,
3. optionally set the GitHub repository homepage to the Zenodo concept DOI landing page.

## Official References

- Zenodo GitHub integration overview: `https://help.zenodo.org/docs/github/`
- Enable a repository: `https://help.zenodo.org/docs/github/enable-repository/`
- CITATION.cff support: `https://help.zenodo.org/docs/github/describe-software/citation-file/`
- Zenodo JSON file behavior: `https://help.zenodo.org/docs/github/describe-software/zenodo-json/`
- Archive a release from GitHub: `https://help.zenodo.org/docs/github/archive-software/github-upload/`
