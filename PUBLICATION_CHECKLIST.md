# Publication Checklist

Current target release: `v0.2.3`

## Repository Readiness

- [x] Public `README.md`
- [x] Version file
- [x] Changelog
- [x] Release notes
- [x] Citation metadata
- [x] Contributing guide
- [x] License file
- [x] `.gitignore`

## Verified Paths

- [x] Unit tests pass
- [x] Stable public quick-start trains successfully
- [x] Stable public quick-start evaluates successfully
- [x] Resource-aware demo trains successfully
- [x] Adaptation analysis script runs successfully
- [x] Feature extraction utility runs successfully

## Honest Limitations To Keep Public

- [x] Semantic context is still placeholder logic
- [x] Legacy multimodal configs are still experimental
- [x] Baseline suite against adapters and LoRA is not yet included
- [x] Resource-aware behavior is functional, but not yet presented as a formal benchmark study

## Recommended GitHub Metadata

Repository name:

`project-synapse`

Short description:

`Resource-conditioned hypernetworks for modular task adaptation, with reproducible public quick-starts and adaptation analysis tools.`

Suggested topics:

`hypernetworks`, `multitask-learning`, `edge-ai`, `parameter-efficient-learning`, `pytorch`, `research-software`, `continual-learning`

## Recommended Launch Presentation

- Make `README.md` the main explainer for what the project is and who it is for
- Use `REPO_DESCRIPTION.md` for the GitHub About text, topics, and pinned issue copy
- Use `GITHUB_RELEASE_DRAFT_v0.2.1.md` as the initial release body
- Keep the wording `research prototype` or `research artifact`
- Do not market the project as a production framework or a completed benchmark claim

## GitHub Tagging Guidance

- No one needs to be tagged for the repository to look professional
- Tag collaborators only if they materially contributed and want to be associated publicly
- If this is a solo release, a clean README, release notes, citation metadata, and a tagged release are enough
- If you have an advisor or lab account, acknowledgement belongs in the README or release notes, not as random GitHub mentions

## Recommended First Push Steps

```bash
git init -b main
git add .
git commit -m "Release v0.2.1 public research candidate"
```

Then:

1. Create the remote repository
2. Push `main`
3. Create an annotated git tag: `git tag -a v0.2.1 -m "Project Synapse v0.2.1"`
4. Push tags with `git push origin main --tags`
5. Create a GitHub release using the current versioned GitHub release draft
6. Pin the quick-start and project-purpose summary in the repo description or a top issue
