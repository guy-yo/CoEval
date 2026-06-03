# CoEval Paper (HTML build)

This directory contains a self-contained academic paper, `index.html`, for the CoEval
LLM-evaluation framework, formatted for the journal TMLR. It is a single HTML file that
loads KaTeX from a CDN to render the math; no build step is required.

## Viewing locally

Just open `index.html` in any modern browser. An internet connection is needed the first
time so the KaTeX CSS/JS can load from the jsdelivr CDN.

## Publishing on GitHub Pages

1. Push this repository to GitHub (the `docs/paper/` directory must be committed).
2. In the repository, go to **Settings -> Pages**.
3. Under **Build and deployment -> Source**, choose **Deploy from a branch**.
4. Set the branch to **`master`** and the folder to **`/docs`**, then click **Save**.
5. GitHub builds the site. After a minute or two, your Pages URL appears at the top of the
   Pages settings (typically `https://<user>.github.io/<repo>/`).

Because the paper lives in `docs/paper/`, the published paper is served at:

```
<pages-url>/paper/
```

For this repository that resolves to:

```
https://apartsinprojects.github.io/CoEval/paper/
```

(`index.html` is served automatically for the `/paper/` path.)

## Results

Every number, table, figure, and confidence interval in `index.html` is backed by a
committed artifact under `Runs/**/reports/*.json`; there are no placeholders. The
empirical sections are:

- **5.1** ground-truth correlation (Spearman ρ = 0.86, 95% CI [0.77, 0.94]);
- **5.2** ensemble reliability, judge-choice regret, the self-validating panel, and the
  **doubly-robust ranking** (recovers the true ordering of 13 models at Spearman 0.95,
  rogue judge weight 0.00);
- **5.3** verbosity-bias cancellation; **5.4** cross-family self-preference;
- **5.5** contamination resistance; **5.6** cost;
- **5.7** domain case studies on three custom verticals;
- **5.8** rankings are domain-specific (three different models top four generated
  domains; the pooled-best is domain-best in only 1 of 4).

## Regenerating the Word versions

After any edit to `index.html`, rebuild both `.docx` downloads with the `html2doc` skill
(single-column `CoEval.docx` via the `camera-ready-generic` profile and two-column
`CoEval_twocolumn.docx` via the `two-column` profile). Audit for zero em-dashes, balanced
`$` delimiters, and zero stray `$` in the rebuilt documents before committing.
