# fgc-vision

Detects and tracks robots in FIRST Global Challenge field livestreams, and measures whether
per-robot, per-team stats (climb time, path, game pieces) can be pulled out of them.

**Status: feasibility spike, final verdict (2026-10-04).** Robot detection works.
Telling *which team* a tracked robot is does **not** work on the 2025 field streams, and
neither do the per-team metrics that depend on it. Three different approaches (chaining
tracks, classifying fragments against start galleries, a trained re-ID embedding) all hit
the same wall: at stream resolution most REV-kit robots look alike, to the models and to
me. On the one match with hand ground truth, robots are verifiably identified for **17 %**
of match time. Nothing here should feed per-team numbers into fgc-scout.

One question was answered without vision: the official per-robot fields
`*RobotOne/Two/Three*` are stations x1/x2/x3 (181/181 teams, see below).

## Verdict per metric

| Metric | Verdict | Measured |
|---|---|---|
| Robot detection | works | YOLO11s F1 0.79 at 14 ms/frame; OWLv2 F1 0.84 at 0.51 s/frame |
| Game-piece detection (2025 balls) | works with caveats | F1 0.73–0.78, 22 labelled balls; 2026 needs retraining |
| Match-time sync | works with caveats | overlay timer read 100 %, but 4.5 s off the video in 1 of 3 matches |
| Robot identity → team | **doesn't work** | 31–46 % wrong joins; 17 % of match time verifiably identified |
| Shots per robot | doesn't work | needs identity |
| Climb speed per team | doesn't work | needs identity; end state is a crowded pile under the canopy |
| Path / speed / idle per team | doesn't work | needs identity (per-anonymous-robot is possible) |
| Climb result per team | not needed | official per-robot data, mapping verified |
| RobotOne = station x1 | **verified** | 181/181 teams vs ranking totals; best alternative 11/181 |

It consumes [fgc-matchwatch](https://github.com/BogdanStamenovic/fgc-matchwatch) for stream
discovery and match placement and does not reimplement them.

## How it works

```
fgc-matchwatch align ──► fetch 180 s of the field stream (1080p60 H.264, yt-dlp sections)
        ──► overlay timer OCR (tesseract) ──► match t=0 to ±0.1 s
        ──► YOLO11s fine-tuned on OWLv2 pseudo-labels, 10 Hz ──► BoT-SORT / ByteTrack
        ──► field mask + static-furniture filter ──► 6-slot stitcher (start anchor = side of field)
        ──► end-game climb signature ──► compare with official per-robot end values
```

| Module | Does |
|---|---|
| `sources.py` | match key → stream + offset (via `fgc-matchwatch align`), section download |
| `overlay.py` | reads the scorebug timer/score; fits match start frame |
| `track.py` | detector + tracker over a clip, JSON per frame |
| `identity.py` | tracklets → per-robot chains; climb detection; official end values / stations |
| `bench/` | the experiments: detector comparison, pseudo-labelling, labelling views, join montages |
| `labels/` | hand labels (points) used for every precision/recall number below |

## What was measured (2025 footage, 8 match clips, fields 1/2/3/5, days 1 and 3)

### Cameras

| Stream | Camera | Usable for tracking |
|---|---|---|
| Per-field streams, fields 1–4 | fixed wide shot, identical framing from 1100 s to 13000 s into the day | yes |
| Field 5 (main stage) per-field stream | fixed wide shot, farther back and wider | yes |
| Main program feed (`… Day N`) | directed broadcast: of 53 samples every 3 s across one match, ~15 are wide field views, the rest booth, crane, robot and drive-team close-ups | no |

Each field's camera pose differs, so the field mask is per camera (one polygon covers
fields 1 and 3 today). Every field stream carries the same scorebug: timer, live score,
"Match N | Field M", teams in station order x1/x2/x3 top-to-bottom (checked on t2-1). The
"Field M" text is not reliable (a field-1 frame said "Field 3"); the API's field is.

### Match-time sync

Tesseract on the timer reads 149/149, 108/108 and 131/131 one-per-second samples in three
clips, and the 2:30→2:29 flip is located to 0.1 s. **But the overlay clock is not always
the video clock.** Checked by eye against the moment robots leave the rail (1 s steps):

| match | robots move (clip s) | overlay t0 | matchwatch (audio) |
|---|---|---|---|
| t2-308 | 15.5 | 15.0 | 15.0 |
| t2-361 | 15.5 | 15.9 | 15.0 |
| t2-1 | 11.5 | 16.0 (4.5 s late) | 10.2 |

So the overlay is right to ~0.5 s in 2 of 3 matches and 4.5 s late in Match 1 of the
event. (An earlier version of this README said the overlay fixes matchwatch on t2-1; that
was backwards.) Any timing metric has to be anchored on the video itself, e.g. on robots
leaving their start spots.

### Robot detection (28 hand-labelled robot centres, 7 frames, 5 clips; none in training)

A detection is correct if its box contains an unmatched labelled centre (Hungarian).
Robots fully hidden behind towers are not labelled, so recall is over *visible* robots.
All models get the same post-filter (a box ≥60 % inside a higher-scoring box is dropped).
The threshold is picked per model on these same frames, so the numbers are optimistic,
and 28 points give a ±~0.1 error bar on each rate.

| Model | P | R | F1 | GPU s/frame | CPU s/frame | VRAM |
|---|---|---|---|---|---|---|
| OWLv2 base ensemble (2 tiles), zero-shot | 0.83 | 0.86 | **0.84** | 0.51 | 5.84 | 0.8 GiB |
| **YOLO11s fine-tuned** on OWLv2 pseudo-labels | 0.77 | 0.82 | 0.79 | **0.014** | 0.18 | 0.2 GiB |
| YOLO11n fine-tuned, same data | 0.69 | 0.79 | 0.73 | 0.010 | 0.07 | 0.1 GiB |
| Grounding DINO tiny, zero-shot ("robot.") | 0.59 | 0.71 | 0.65 | 0.26 (fp16) | – | 2.2 GiB |
| YOLO-World v2 s / l, zero-shot | 0 | 0 | 0 | 0.016 / 0.048 | – | 1.0 GiB |

YOLO-World never outputs "robot" on these frames; 9 prompts tried ("machine", "vehicle",
"toy car", "metal frame", "wheeled robot", …), best was "box" 14 times in 7 frames.
The fine-tuned models were trained on 344 frames from 4 *other* clips (t2-2, t2-16, t2-304,
t2-50), labelled by OWLv2 (robot ≥ 0.25, ball ≥ 0.30) **without hand correction**; 40
epochs at 1280 px take about 8 minutes on the RTX 4060.

### Game pieces (22 hand-labelled 2025 balls, 2 frames)

| Model | P | R | F1 |
|---|---|---|---|
| YOLO-World v2 l | 0.75 | 0.82 | 0.78 |
| OWLv2 | 0.62 | 0.95 | 0.75 |
| YOLO11s fine-tuned | 0.63 | 0.86 | 0.73 |
| Grounding DINO tiny | 0.63 | 0.86 | 0.73 |

Precision is a lower bound: I very likely missed some balls while labelling. The 2025
ball is not the 2026 ball, so this table is about the method, not about 2026.

### Tracking and identity (t2-1 and t2-308, both trackers)

| | t2-1 ByteTrack | t2-1 BoT-SORT | t2-308 ByteTrack | t2-308 BoT-SORT |
|---|---|---|---|---|
| tracklets (≥ 2 s) for 6 robots | 58 | 65 | – | – |
| tracklets inside field mask | 101 | 105 | 172 | 181 |
| robots with a single track from start to end | 0 | 0 | 0 | 0 |
| chain coverage of match time (6 chains, median) | 0.38 | 0.35 | 0.51 | 0.50 |
| start anchor: 3 red + 3 blue seeds | 3/3 | 3/3 | **2/4** | 3/3 |

Robots disappear behind the three towers and the referee every few seconds; trackers
cannot bridge that. The stitcher joins tracklets into 6 chains by position and gap time.
**ID-switch rate, judged by eye on all 70 joins of t2-1 BoT-SORT** (crop before vs crop
after each join, `bench/identity_run.py KEY botsort joins`):

| same robot | confident switch | probable switch | can't tell at 90 px |
|---|---|---|---|
| 29 | 19 | 6 | 16 |

That is 35–46 % switches per judgeable join, with ~12 joins per chain: a chain survives
start to end intact with probability around 1 %.

Next approach class tried: appearance. On those judged joins, CLIP ViT-B/32 image
similarity separates same-robot joins from switches with AUC 0.86 (HSV histogram 0.78,
box-size ratio 0.73; `bench/reid_probe.py`). Stitcher v2 (`bench/stitch_v2.py`) adds a CLIP
zero-shot "robot vs person vs tower vs balls" filter (drops 10 of 105 tracklets) and an
appearance floor (cosine ≥ 0.80) and cost. Re-judged on all 66 of its joins:

| stitcher | same | confident switch | probable switch | can't tell | switch rate |
|---|---|---|---|---|---|
| v1 (position + gap) | 29 | 19 | 6 | 16 | 35–46 % |
| v2 (+ CLIP filter, appearance gate) | 35 | 17 | 2 | 12 | 31–35 % |

Better, but still roughly one wrong join in three. v2 crops were padded more (40 %), which
makes judging a little easier, so part of the gain may be the judge, not the stitcher.

### Identity by classification against start galleries

Before "go" every robot stands still at its start spot. `bench/gallery_extract.py`
clusters 0.5 s-spaced detections in that window into 6 start robots; that found all 6
spots in t2-1, t2-308, t2-361, t2-30, t2-2, t4-3 and t2-50, but only 2–3 of 6 in t2-16
and t2-304 (people in front of the rail). Each track fragment is then assigned to the
nearest gallery independently, so errors do not compound.

- **Alliance LEDs are not visible** at stream resolution (upscaled crops checked), so
  they cannot be the hard constraint; alliance comes only from the start side.
- **Ground truth is the bottleneck.** Of 71 robot fragments in t2-1, I could name the
  robot with confidence for 20. Two robots (an orange-LED arm and a black shooter wheel)
  are distinctive; the other four are near-identical silver cages once they move. t2-361
  looked the same (one distinctive robot of six), so I stopped labelling there.

| Embedding (frozen) | accuracy on the 20 | alliance correct | robot-vs-non-robot AUC |
|---|---|---|---|
| CLIP ViT-B/32 | 0.60 | 0.90 | 0.90 |
| CLIP ViT-L/14 | 0.65 | 0.70 | 0.96 |
| DINOv2-base | **0.75** | 0.85 | 0.99 |
| DINOv2-large | 0.75 | 0.95 | 0.96 |
| DINOv2-base + trained re-ID head | 0.80 | – | – |

Chance is 0.17. Adding "two fragments on screen at once are different robots" (greedy by
margin) made it *worse* (0.40–0.50), because unlabelled duplicates and false positives
take the slots. The trained head (`bench/reid_train.py`: contrastive, positives = two crops
of one tracklet, negatives = concurrent tracklets, 8 matches, t2-1 held out) gains one
fragment out of 20, which is noise.

Share of t2-1 match time (6 robots × 150 s) covered by fragments that are identified
correctly *and verifiably*: **0.17**. Per robot: the black-shooter robot 0.47, the
arm robot 0.29, the others 0.06–0.17. The 0.24 I could label is the ceiling of what can
be checked; the rest is unknown, not known to be wrong.

### Cross-match consensus (team identity at the start)

A team's robot looks the same in every match. Each match tells us the 3 teams per side,
so per alliance only the ordering (1 of 6) is unknown. `bench/consensus.py` solves all
orderings jointly over 97 matches / 202 alliances (pre-start windows of 136 matches
downloaded; 28 did not decode, 11 had fewer than 3 robots per side) by maximising
within-team appearance similarity, with per-match centring to remove camera/lighting.
Control: the same solver with team labels scrambled.

| Embedding | objective real | objective control | ratio |
|---|---|---|---|
| DINOv2-base (10 crops) | 202.5 | 99.3 | 2.04 |
| DINOv2-base (4 crops) | 188.6 | 94.3 | 2.00 |
| CLIP ViT-L/14 | 204.9 | 159.4 | 1.29 |
| trained re-ID head | 267.4 | 177.5 | 1.51 |

**Blind check** (`bench/blind_sheet.py`, 24 team rows, 12 real and 12 control in random
order, judged before reading the key, `labels/blind_7_judgement.json`): real rows had
0.64 of crops matching the row's majority robot, control rows 0.48, and I told real from
control in 15/24 rows (chance 12). A signal, far from usable identity. Many switches go onto a referee or a
tower AprilTag. The 16 undecidable joins are the deeper problem: at stream resolution
the REV-kit robots look alike even to a human.

### End anchor and the RobotOne = station x1 assumption

- 2025 per-robot end values: 0.5 ×849, 0.125 ×553, 0.25 ×357, 0.375 ×348, 0 ×179. On
  t2-1 the end frame shows three robots hanging from the canopy, matching the three 0.5
  values, so 0.5 reads as a full hang. The other levels are not mapped to video yet.
- Climb signature (sustained rise ≥45 px with ≤45 px sideways drift): on t2-1 BoT-SORT
  it fires on 1 chain (rise 110 px, 12.8 s); officially 3 robots hung. Since the chains
  are not one robot each, this number is not a usable climb time.
- **RobotOne/Two/Three = station x1/x2/x3 is verified, without vision.** The rankings
  carry a per-team season total, `protectionPoints`. Summing each team's `*Robot<k>Parking`
  over its 12 ranking matches reproduces it exactly for **181/181 teams** under the mapping
  One/Two/Three = x1/x2/x3. The best other permutation matches 11/181 (`fgc-vision
  check-mapping`, which finds the field pairs generically so it can be rerun on 2026 data).

### End-of-match anchor

At the buzzer in t2-308 all six robots are bunched under the central canopy, overlapping,
and referees step in front. End-of-match crops are poor identity anchors in 2025; the
2026 BRACE (one sloped pipe, 3 zones) will likely bunch robots the same way.

### Country stickers

Dropped on Bogdan's call after he watched the footage. A robot is 70–120 px wide at
1080p, so a 13×8 cm sticker is roughly 20–30 px wide with ~5 px letters. Not attempted
as OCR.

### Throughput

| Path | Speed | Per 2.5 min match |
|---|---|---|
| YOLO11s + BoT-SORT, 10 Hz, GPU, incl. 1080p decode | 22–23 ms per processed frame | ~36 s |
| same, CPU only (Ryzen 5 5600, 12 threads) | 238 ms per processed frame | ~6.2 min (≈75 s at 2 Hz) |
| OWLv2 on GPU | 0.51 s/frame | – (too slow to track with) |
| Overlay sync, CPU | ~60 s per clip at 1 Hz sampling (tesseract) | ~60 s |
| Download, 180 s of 1080p60 | ~140 s wall | – |

## Install

```sh
ownbox install fgc-vision
```

or by hand (torch and weights are large, so on archserver the venv lives on /mnt/offload):

```sh
git clone https://github.com/BogdanStamenovic/fgc-vision ~/data/fgc-vision
cd ~/data/fgc-vision
uv venv -p 3.12 /mnt/offload/fgc-vision/venv && ln -s /mnt/offload/fgc-vision/venv .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -e ".[models,dev]"
```

Needs `tesseract`, `ffmpeg`, `node`, and a working fgc-matchwatch install (its venv's
`yt-dlp` and its `align` command). Data, videos and weights go under
`$FGC_VISION_DATA` (default `/mnt/offload/fgc-vision`). The fine-tuned weights are not in
git; regenerate them with `bench/autolabel.py` and `train.py` (8 min on the 4060).

## Usage

| Command | What it does |
|---|---|
| `fgc-vision place KEY` | stream id and start second of a match (from fgc-matchwatch) |
| `fgc-vision fetch KEY [--video ID --start S] [--dry-run]` | download the match window at 1080p |
| `fgc-vision sync CLIP` | match start frame from the overlay timer |
| `fgc-vision track KEY [--tracker botsort\|bytetrack] [--device cpu]` | detect + track, JSON |
| `fgc-vision check-mapping [--tournament t2]` | which station each per-robot official field belongs to |
| `fgc-vision chains KEY` | stitch chains, climb detection, official end values (diagnostic only) |

Global: `--year`, `-v`, `-q`, `--version`. stdout is JSON only. Exit codes: 0 ok,
1 failed, 2 usage, 130 interrupted.

Licences: Ultralytics (YOLO11, YOLO-World, trackers) is **AGPL-3.0**. Fine for internal
use; publishing a service built on it would oblige releasing its source. OWLv2 and
Grounding DINO (via transformers) are Apache-2.0.

## 2026: what would be needed on day 1 (Thu 8 Oct)

2026 changes the game (100 mm orange balls shot into the SUPPRESSION UNIT, 6.4 m sloped
BRACE pipe climb), so the detector, the field masks and the climb signature all need
re-calibrating:

| Need | Why | Time |
|---|---|---|
| per-field stream ids (matchwatch `streams discover`) + 4–6 played matches per field | training/eval footage | download ~2.5 min per match |
| 1 frame per field camera | field mask polygon | 5 min per camera |
| ~30 hand-labelled frames (robots, orange balls, robots on the BRACE) | eval set | ~45 min |
| OWLv2 pseudo-labels on ~350 frames + YOLO11s fine-tune | detector | ~15 min GPU |
| climb signature along the pipe direction instead of straight up | climb timing | ~1 h |

About 3 h of work, plus GPU time, which means unloading cvoiced.

## Limitations

- **Identity does not work**: 31–46 % wrong joins when chaining, 17 % of match time
  verifiably identified when classifying, a weak cross-match signal (blind 0.64 vs 0.48).
  Every per-team metric depends on it; none are produced.
- Fragment ground truth exists for one match only (20 fragments), biased toward the
  distinctive robots; accuracy on the other robots is unmeasured.
- The cross-match test reuses pre-start crops from 8 matches whose in-match tracklets
  trained the re-ID head.
- Evaluation sets are small (28 robots, 22 balls) and hand-labelled by eye on a 50 px grid;
  thresholds were tuned on the same frames.
- Pseudo-labels were not hand-corrected; the student inherits OWLv2's mistakes
  (it boxes referees and tower AprilTags).
- The field mask is one polygon fitted to fields 1 and 3; the field-5 camera is framed
  differently and has no mask yet.
- The overlay ROI positions are 2025's scorebug; a 2026 redesign moves them.
- Only official per-robot end values exist as truth for robots. Per-alliance ball counts
  are not usable as 2025 truth: the live score is shared by both alliances and the
  official alliance scores include multipliers.
- Direct-URL seeking into YouTube streams returned 403 after about an hour of use;
  `yt-dlp --download-sections` kept working. Not worked around further.

## What does not exist yet

- Per-team output for fgc-scout. A proposed shape, for when identity is good enough:
  `{"code":"KAZ","matchKey":"t2-12","metrics":{"climbSeconds":11.2,"pathMeters":41.0,
  "idleSeconds":18.5,"shots":7},"confidence":{"identity":0.9,"climb":0.8},
  "source":{"tool":"fgc-vision","version":"0.1.0","video":"<id>","t0":1234.5}}`
- Homography to metres (path length, speed), per-zone time.
- Anything that beats the resolution limit. Options, none tried: a team-side scout who
  tags robots by hand at the start of each match (cheap and probably the real answer), our
  own camera at the event (needs FGC's permission), or per-alliance instead of per-team
  stats, which need no identity at all.
