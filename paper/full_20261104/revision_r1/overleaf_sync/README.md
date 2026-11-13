# English manuscript → Overleaf

## User request and target

On 2026-11-13 the author asked to upload the current English `main` manuscript
and to keep local TeX and Overleaf updated together in subsequent revision
tasks. This is a standing workflow preference, not a background watcher.

- Local authoritative source: `../main.tex`.
- Overleaf project: https://www.overleaf.com/project/6ad9c20b8e72c0d209b0c86e
- Git URL: `https://git@git.overleaf.com/6ad9c20b8e72c0d209b0c86e`.
- Intended Overleaf main document: `main.tex`, compiled with pdfLaTeX/BibTeX.
- Upload only the English project. The bilingual review copy, raw experimental
  data, checkpoints, audit folders, and the research Git history are excluded.

## Connection status: NOT UPLOADED

Checked 2026-11-13:

1. No Overleaf plugin was returned by the available plugin search.
2. No connected document-control session or interactive browser-control tool
   was available. The browser-open request was only queued; it does not prove
   project access, upload, or save.
3. The Overleaf Git host was reachable outside the network-restricted sandbox,
   but `git ls-remote` stopped because no Git authentication identity was
   available. No project contents or remote refs were retrieved. Project write
   permissions, Git entitlement, comments, and remote file conflicts remain
   unverified. No remote file has been changed.

The user must configure a working authenticated connection before the initial
upload or automatic updates can complete. Never request that the user paste
an account password, browser cookie, or Git token into chat. Use a secure local
credential helper or an authenticated, supported browser connection instead.
Overleaf's official Git documentation is at:
https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens

## Minimal source bundle

`prepare_bundle.py` packages these five files from the parent directory without
changing their contents:

- `main.tex`
- `fig_method.tex`
- `refs.bib`
- `elsarticle.cls`
- `elsarticle-harv.bst`

Run with Python 3.11:

```
/home/marmot/anaconda3/envs/py311/bin/python paper/full_20261104/revision_r1/overleaf_sync/prepare_bundle.py
```

Outputs here are `main_overleaf_20261113.zip` and `bundle_manifest.json`. The
manifest records SHA-256 values for the packaged sources and ZIP, and retains
the `not_uploaded` status. Packaging does not establish remote synchronization.
The initial bundle was extracted into a fresh temporary directory and compiled
successfully with pdfLaTeX/BibTeX; all cross-references resolved. The existing
five BibTeX empty-pages warnings remain. This is a local build, not an Overleaf
compile or upload.
For manual upload to the existing project, extract the ZIP and upload its five
files to the project root, after preserving the existing remote sources. Do not
assume uploading a ZIP inside an existing project will extract its contents.

## Workflow after connection

1. Read/download or fetch the existing Overleaf project into a separate local
   checkout; preserve its current state before writing. Do not attach an
   Overleaf remote to the research repository and push its history.
2. Compare remote files with the local manuscript and any last-synchronized
   baseline. Preserve coauthor changes. Stop for author direction on conflicts
   that cannot be safely merged. Check for comments or tracked changes before
   using Git, since a Git push may displace or remove them.
3. Apply only the authorized manuscript changes and necessary dependencies.
   Keep the local bilingual review copy consistent under the existing workflow;
   do not upload it unless separately requested.
4. Compile locally. Upload/push only the English project's changed source
   files. Never force-push or delete unrelated Overleaf files.
5. Read back the remote source or fetch after the write and compare the
   intended file contents/hashes. Record the resulting remote revision and
   source hashes. Compile remotely when the connection supports it; distinguish
   remote compilation from a successful local build.
6. Tell the user which destinations updated. An authentication/network failure
   leaves the remote update pending, even if the local PDF compiles.

