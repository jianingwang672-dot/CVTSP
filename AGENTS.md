# Manuscript and Overleaf synchronization

The user requested on 2026-11-13 that subsequent manuscript edits update both
the local English TeX source and this Overleaf project:
https://www.overleaf.com/project/6ad9c20b8e72c0d209b0c86e

- The active local English manuscript is
  `paper/full_20261104/revision_r1/main.tex`. Do not substitute the older
  `paper/trE_overleaf/main.tex`.
- For edits to this manuscript or its dependencies, follow
  `paper/full_20261104/revision_r1/overleaf_sync/README.md`.
- Update Overleaf in the same authorized manuscript-editing task when a working
  authenticated connection is available. If unavailable, preserve the local
  edits and explicitly report that the remote update is pending. Never claim
  successful synchronization based only on a local edit, ZIP, or browser-open
  request.
- Before remote writes, read and preserve the current Overleaf sources. Check
  for coauthor edits and resolve conflicts; do not force-push or silently
  discard remote-only changes, comments, or tracked changes.
- Keep any Overleaf Git checkout separate from the research repository. Never
  push this repository's experiment data, models, or history to Overleaf.
- Do not store authentication tokens or private share-link fragments in source
  files, repository configuration, scripts, logs, or chat messages. Use an
  appropriately secured credential helper configured by the user.

