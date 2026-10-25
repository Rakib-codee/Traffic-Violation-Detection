# Privacy and ethics note

## Purpose and scope

This system flags *candidate events for human review*. It is not an enforcement
tool: no automated fines, penalties, or notifications are produced, and no
decision about a person or vehicle is made without a human reviewer.

## Data minimization in stored outputs

- Plates and faces are blurred in stored evidence (clips and keyframes) by
  default. The pipeline's `--no-blur` flag exists only for debugging and must
  not be used for stored review data.
- Evidence clips are short windows (default 7 s) around a candidate event, not
  continuous footage.
- The optional ANPR module reads plates only to support review; plate text is
  not required for any core violation rule and is never committed to this
  repository.

## Raw footage

Raw video of people is never committed to this repository (`data/` and
`results/` are gitignored). Sample clips come from public research datasets or
are synthetic; see `docs/DATA.md` for sources and licenses. Any own footage
must come from permitted sources with documented permission.

## Recommended operating policy

These are recommendations for anyone deploying this code, not properties of
the code itself:

- Access control: restrict raw (unblurred) evidence to authorized reviewers.
- Retention: define and document a retention period for evidence and event
  metadata; delete on schedule.
- Notice: comply with local law on video surveillance signage and data
  protection (e.g. GDPR-style duties where applicable).
- Human review: treat every event as a lead, not a verdict. False positives
  occur; reviewers must confirm against the evidence clip.
- Bias/performance: detection accuracy varies with scene, camera angle, and
  lighting (see `docs/ROBUSTNESS.md`); calibrate per camera and re-validate
  before relying on outputs.

## License compliance

Datasets used for training and evaluation are listed with their licenses in
`docs/DATA.md`. Verify license terms (including region restrictions for plate
datasets) before redistributing derived models or data.
