# Portfolio / CV text

Copy-paste ready project descriptions. Trim to fit the medium.

## One-liner (CV project line)

**Traffic Violation Detection** — end-to-end computer-vision pipeline that
detects, tracks and flags traffic-rule violations in video with reviewable
evidence (Python, YOLOv8, ByteTrack, FastAPI, Streamlit, Docker).

## CV bullets (pick 3–4)

- Built an end-to-end pipeline (detection → tracking → scene calibration →
  rule engine → evidence dashboard) flagging red-light and wrong-way
  violations as candidates for human review, with plates/faces blurred by
  default for privacy.
- Fine-tuned YOLOv8 on UA-DETRAC to mAP@0.5 0.970 / mAP@0.5:0.95 0.841;
  verified robustness under night/rain/glare degradations (≤ 0.03 mAP drop).
- Designed a constructed-case evaluation methodology for rare events
  (reversed-playback wrong-way, synthetic signal overlay) with ground truth
  by construction; it caught two real failure modes before field use.
- Novel component: lightweight online tracklet stitching that lifts
  violation-event recall from 0.33 to 0.59 (and up to 0.96) under 10–30%
  tracker misses at zero precision cost, validated by a seeded dropout study.
- Engineering: 102 pytest tests, Dockerized fresh-clone demo, seeded and
  pinned reproducible runs, results regenerated from real executions only.

## LinkedIn / portfolio paragraph

Built a complete traffic-violation detection system from video: a YOLOv8
detector fine-tuned on UA-DETRAC (mAP@0.5 0.970), ByteTrack tracking with
scene-aware rules (stop line, lanes, signal state) that flag red-light and
wrong-way violations and produce blurred, review-ready evidence clips behind
a FastAPI + Streamlit dashboard. The part I am proudest of is the evaluation
engineering: with no labeled violation footage available, I built constructed
test cases (reversed playback, synthetic signal overlay) whose ground truth
holds by construction — they caught two real bugs (an occlusion-driven false
positive and a mislabeled "duplicate" that was actually a second violator).
I then designed a lightweight track-stitching module that keeps violation
recall at 0.96 under 10% tracker misses (vs 0.59 without), measured with a
seeded dropout study. Everything is reproducible: `make demo` or one Docker
command from a fresh clone. Code: github.com/Rakib-codee/Traffic-Violation-Detection

## Elevator pitch (30 s, spoken)

"Traffic violations are rare events buried in hours of video. I built a
pipeline that detects and tracks vehicles, understands each camera's scene,
and flags red-light and wrong-way violations as reviewable candidates — never
automated penalties. The detector hits 0.97 mAP, and because no labeled
violation footage exists, I created constructed test cases with built-in
ground truth, which caught real bugs before deployment. My main novel
contribution is a lightweight track-stitching layer that roughly doubles
violation recall under tracker misses, at zero precision cost. It runs from a
fresh clone with one command."
