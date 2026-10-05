# paper/: Progress report (LaTeX)

This folder contains the thesis progress report. The report is in English and uses ASD-STE100 (Simplified Technical English). Edit the files locally, build the PDF, then pack the folder into a `.zip` file and upload it to Overleaf for supervisor review.

---

## 1. Folder layout

```
paper/
├── main.tex              # main file: preamble, title, abstract, \input of each section
├── sections/             # CONTENT: one file for each section; edit the text here
│   ├── 01_intro.tex
│   ├── 02_literature.tex
│   ├── 03_pipeline.tex
│   ├── 04_setup.tex
│   ├── 05_results.tex
│   ├── 06_limits.tex
│   ├── 07_next.tex       # Next steps + Conclusion
│   └── A_appendix.tex
├── refs.bib              # bibliography (add entries only with fetch_bibtex.sh, see section 5)
├── results/              # SOURCE DATA: copies of logs/CSVs from data/runs/, data/irra_rtsp/, ...
├── scripts/gen_tables.py # makes the tables and numbers from results/ into generated/
├── generated/            # GENERATED: do not edit by hand (macros.tex, tab_*.tex, *.csv for plots)
├── build/                # build output (PDF, .aux, .log); do not upload, do not commit
└── README.md             # this file
```

Rule: **change text in `sections/`**, **change data in `results/`** and run the script. Do not edit `generated/` or `build/` by hand.

---

## 2. Build the PDF locally

```bash
cd ~/code/master-thesis/paper
python3 scripts/gen_tables.py              # only necessary when results/ changes
latexmk -pdf -outdir=build main.tex        # runs all passes (pdflatex + bibtex + pdflatex x2)
```

The PDF is `build/main.pdf`.

- **In VS Code (LaTeX Workshop):** open `main.tex` and push Build (Ctrl+Alt+B). `.vscode/settings.json` sets the build to latexmk, with the output in `paper/build/`.
- **Do not** run one single `pdflatex main.tex` or `xelatex main.tex` pass. This skips bibtex, so citations and references show `??`, and the `.aux/.log/.pdf` files go directly into `paper/`.
- To remove the build files: `latexmk -C -outdir=build` or `rm -rf build`.

---

## 3. Edit the LaTeX content

### 3.1 Edit the text
Open the related file in `sections/` and edit it directly:

| To change | File |
|---|---|
| Title, author names, abstract | `main.tex` |
| Problem, gap, research questions, work done | `sections/01_intro.tex` |
| Literature (PEFT vs. full fine-tuning, SDM) | `sections/02_literature.tex` |
| Pipeline `clip_lora_fedavg` | `sections/03_pipeline.tex` |
| Data partition, runs | `sections/04_setup.tex` |
| Results: C1-true, B5, client loss | `sections/05_results.tex` |
| Limitations, claims and evidence | `sections/06_limits.tex` |
| Next steps, conclusion | `sections/07_next.tex` |

**Open item:** write the names in `main.tex`, line `\author{\todo{student name} ... \todo{supervisor name}}`.

### 3.2 Add a new section
1. Make `sections/08_xxx.tex`. Start it with `\section{...}\label{sec:xxx}`.
2. Add `\input{sections/08_xxx}` to `main.tex` at the correct position.

### 3.3 Available macros (defined in `main.tex`)

| Macro | Use |
|---|---|
| `\result{...}` | blue "Result." box for a main conclusion |
| `\caution{...}` | orange "Caution." box for a limitation |
| `\fit{...}` | makes a wide table or figure smaller to fit the text width |
| `\todo{...}` | red note `[TODO: ...]` |
| `\pp` | unit "pp" (percentage points), for example `1.55\pp` |
| `\numXxx` | generated number from `generated/macros.tex`, for example `\numCPlatR` = converged R@1 of C1-true |

Cross-references: `\cref{sec:results}`, `\cref{tab:runs}`, `\cref{fig:curves}`. Citations: `\citep{<key>}`, with `<key>` from `refs.bib`.

### 3.4 Writing rules (ASD-STE100)
- Short sentences: maximum approximately 20 words (procedures) and 25 words (descriptions). One sentence, one idea.
- Active voice, present tense. Use the same term for the same thing (always *client*, *round*, *centralized run*, *FL run*).
- Do not write a claim that is stronger than the evidence. Each new claim must have a row in the table `tab:claims` (`06_limits.tex`).

### 3.5 Common errors

| Error | Cause / correction |
|---|---|
| `??` in the PDF | not all build passes ran → use `latexmk` (section 2) |
| `<` or `>` shows incorrectly | in text, write `$<$` and `$>$` |
| `%`, `&`, `_`, `#` cause errors | escape them: `\%`, `\&`, `\_`, `\#` |
| Error with a macro name that has digits (`\numR30`) | LaTeX command names contain only letters → `\numRThirty` |
| Errors with Vietnamese characters | the report uses pdfLaTeX and English; do not type Vietnamese diacritics in `.tex` files |
| Table wider than the text (Overfull hbox) | put the tabular in `\fit{...}` |

---

## 4. Update the numbers (after a new run)

All experimental numbers come from `results/`. **Do not type numbers by hand into `.tex` files.**

1. Copy the new log into `results/`, for example:
   ```bash
   cp ../data/runs/<run_dir>/log.csv results/<RunName>_log.csv
   ```
2. For a new run (not only a replaced file): change `scripts/gen_tables.py` so that it reads the file and writes new macros or tables into `generated/`.
3. Run `python3 scripts/gen_tables.py`, then build again (section 2).
4. Make sure that the related text in `sections/05_results.tex` is still correct for the new numbers.

Note: `gen_tables.py` reads `../data/runs/B5_camera_lora4_v2_20260709/args.json` for the parameter counts. Thus, the script runs only inside the `master-thesis` repository, not on Overleaf. For this reason, the zip file must always contain `generated/`.

---

## 5. Add a reference

Add references only with the script (from a DOI or an arXiv ID). Do not type BibTeX from memory:

```bash
cd ~/code/master-thesis/paper
bash ../.claude/skills/latex-document/scripts/fetch_bibtex.sh 2106.09685 --append --output refs.bib
grep "^@" refs.bib | tail -1      # shows the new key; use it in \citep{...}
```

Make sure that the title of the new entry is the paper that you want to cite.

---

## 6. Upload to Overleaf (with a zip file)

### 6.1 First upload
```bash
cd ~/code/master-thesis/paper && rm -f ../paper.zip && zip -r ../paper.zip . -x "build/*"
```
The zip file is `~/code/master-thesis/paper.zip`. The `rm -f` removes the old zip first: `zip -r` into an existing file adds to it, so deleted files stay in the archive.

On Overleaf:
1. **New Project → Upload Project** → select `paper.zip`.
2. **Menu → Compiler: pdfLaTeX**, **Main document: `main.tex`** → Recompile.
3. **Share** → invite the supervisors by email (*Can edit*, or *Can review* if your plan has it). Turn on **Review → Track changes** to see who changed what.

### 6.2 Send local changes to Overleaf
A new zip upload makes a **new project**. The new project does not have the comments and the history of the old project. Thus, for a shared project:
- For a small number of changed files: on Overleaf, click **Upload** (icon above the file tree) and select each changed file. A file with the same name is replaced.
- For changed numbers: also upload the changed files in `generated/`.
- Before you replace a file, make sure that nobody changed it on Overleaf. If somebody did, you lose their changes.

### 6.3 Get supervisor changes into the local folder
1. On Overleaf: **Menu → Download → Source** to download a zip.
2. Locally:
   ```bash
   cd ~/code/master-thesis/paper
   unzip -o ~/Downloads/<project-name>.zip -d /tmp/overleaf_src
   diff -ru --exclude=build . /tmp/overleaf_src | less     # see what the supervisors changed
   rsync -a /tmp/overleaf_src/ ./                          # accept the changes
   ```
3. Build again (section 2) to check the result.

**Overleaf comments are not in the zip.** Read and answer the comments on Overleaf. To process them locally (or with an AI assistant), copy them into a file (for example `paper/REVIEW_NOTES.md`), or ask the supervisors to write their notes directly in the `.tex` files.

With a paid Overleaf plan, you can use Git (`git pull` / `git push`) instead of zip files.

---

## 7. Edit with an AI assistant

You can use any AI coding assistant that can read and edit the files in this repository (in the terminal or in the editor).

### 7.1 Give the assistant the context
Tell the assistant to read these files before it makes changes:
- `paper/README.md` (this file): folder layout, build command, writing rules.
- The project instruction file in the repository root (for example `CLAUDE.md` or `AGENTS.md`), if your assistant does not read it automatically.

### 7.2 Rules for the assistant
Include these rules in the request, or keep them in the project instruction file:
1. Do not invent numbers. All experimental numbers come from `results/` through `scripts/gen_tables.py`.
2. Each `\cite`/`\citep` key must exist in `refs.bib`. Add entries only with `fetch_bibtex.sh` (section 5).
3. The code in `experiments/clip_lora_fedavg/` is the reference for the method description. Check the text against the code.
4. Follow the writing rules in section 3.4 (ASD-STE100).
5. Do not edit `generated/` or `build/` by hand.
6. After each change, build with `latexmk -pdf -outdir=build main.tex` and make sure that there are no errors and no `??`.

### 7.3 Example requests
- "Change section 5.3 in `paper/sections/05_results.tex` as the notes in `paper/REVIEW_NOTES.md` tell you."
- "Add the rank-sweep run to `paper/results/`, update `gen_tables.py`, and add a results table."
- "Make the abstract in `paper/main.tex` shorter. Keep ASD-STE100."

### 7.4 After the assistant changes the files
1. Read the diff (`git diff paper/`). Make sure that all numbers and claims are correct.
2. Build again (section 2) and read the PDF.
3. Upload the changed files to Overleaf (section 6.2).
