# LinkedIn post draft

Post the video `results/showcase/demo.mp4` as a **native upload** (attach it
directly to the post — not a YouTube/GitHub link). Put the GitHub link in the
**first comment**, not in the post body (external links in the body reduce
reach). Suggested first comment:

> Code, docs and a one-command demo (make demo / Docker):
> https://github.com/Rakib-codee/Traffic-Violation-Detection

---

## Post text (English)

🚦 Traffic violations are rare events buried in hours of footage — so I built
a system that finds them and hands a human reviewer the evidence.

Over the past weeks I built an end-to-end traffic violation detection
pipeline in Python: YOLOv8 detection → ByteTrack tracking → per-camera scene
calibration (stop line, lanes, signal state) → a rule engine that flags
red-light and wrong-way violations → blurred evidence clips behind a
FastAPI + Streamlit review dashboard. Every output is a candidate for human
review — never an automated penalty.

Some numbers (all from real, seeded runs):

• Vehicle detector: 0.970 mAP@0.5 on UA-DETRAC (0.841 mAP@0.5:0.95)
• Event-level rules on real video: P/R/F1 = 1.000 on a hand-verified
  constructed test set
• Robust under night/rain/glare degradations (≤ 0.03 mAP drop)

The part I learned the most from: there is no labeled "violation" footage to
evaluate against, so I built constructed test cases with ground truth by
construction — reversed playback makes every vehicle wrong-way, a synthetic
signal overlay creates controlled red-light scenarios. That evaluation caught
two real bugs before any field use, including an occlusion-driven false
positive I then fixed.

I also designed a lightweight track-stitching module: when the tracker loses
vehicles (occlusion, ID switches), it re-links the fragments — lifting
violation recall from 0.33 to 0.96 under 10% tracker misses at zero precision
cost, measured with a seeded dropout study.

Everything is reproducible from a fresh clone with one command (make demo or
a single Docker run). Privacy by default: plates/faces are blurred in stored
evidence.

Link to the code in the comments 👇 Feedback welcome!

#ComputerVision #MachineLearning #Python #OpenCV #DeepLearning

---

## Posting checklist

- [ ] Attach `results/showcase/demo.mp4` directly to the post (native video)
- [ ] Paste the post text above
- [ ] Publish on a weekday morning (9–11 am)
- [ ] Immediately add the first comment with the GitHub link
- [ ] Add the repo to your profile's **Featured** section
- [ ] On GitHub: pin the repo, add topics (computer-vision, yolov8,
      object-detection, traffic-monitoring, bytetrack)
- [ ] Reply to every comment in the first 2 hours (engagement signal)
