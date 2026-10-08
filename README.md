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

## v2: metres, motion-model linking, humans in the loop (2026-10-04 evening)

Bogdan's v2 design: calibrate each fixed camera to metres, link tracks with a motion
model, and let scouts settle identity by tapping crops in fgc-scout. Measured on t2-1, the
one match with complete time-sampled hand ground truth. Treat every number as provisional
for 2026: the 2026 field puts three 201 cm goals side by side in the middle and two
sloped BRACE pipes, so occlusion will differ.

### Ground truth, and the ceiling it reveals

`bench/gt_points.py`: every 5 s (28 times, 0–149 s), every tracked box in the field is
shown at 200 px next to the six start robots, and I label it A–F, not-a-robot, duplicate,
or "can't tell" (`labels/gtp_t2-1.json`). Per-fragment labels were abandoned because
tracker fragments switch robots mid-way (fragment 2349 starts on robot C, ends on B).

| | share of 6 robots × 28 times |
|---|---|
| any tracked box on a robot | **0.45** |
| box whose robot I could name | **0.27** |

Of 75 robot boxes I could name 46 (61 %). Three robots are distinctive (A: rollers on
top; C: black wheel and long arm; D: white plate, parks at the rail); B, E and F are
near-identical silver cages and I named them 10 times in total. After 115 s (the climb)
I could name nothing. At 142 s no robot had a box at all.

**That ceiling binds humans too.** A scout tapping the same 360 px crop sees what I saw.

### Metres

`calibrate.py`: median frame → carpet mask → rail edges → homography plus one radial
distortion term, fitted to the rail lines with robust least squares. Check, not used in
the fit: the regional-zone tape (50 cm from the side rail).

| camera | rails (median abs, m) | tape at (expected 0.50 m) | accepted |
|---|---|---|---|
| d1f1 | 0.01–0.03 | 0.49 | yes |
| d1f3 | 0.03–0.08 | 0.48 | yes |
| d3f3 | 0.02–0.09 | 0.47 | yes |
| d3f1, d3f2 | 0.03–0.23 | 0.43, 0.59 | no |
| d1f2, d3f5 | only 1–3 rails found | tape not found | no |

On d1f1 the six robots standing before "go" map to x = 0.50–0.56 m (red) and 6.65–6.74 m
(blue), consistent with robots against the rails, and all inside the 4.1 m zone.
Stationary jitter: 2–15 mm std per axis (box bottom centre as floor contact). Under
occlusion the bottom edge jumps, and that is not measured. 3 of 7 cameras calibrate
automatically from rough corner hints; the rest need a manual point set (about 5 min each).

### Linking in metres with a Kalman filter vs the v1 pixel stitcher

Scored on the 46 identified samples: a sample counts if its box sits in a chain seeded
by the right start robot.

| linker | samples in a chain | correct |
|---|---|---|
| v1 pixels, gap-gated nearest | 36 | **20 (0.56)** |
| metres + constant-velocity Kalman, defaults | 36 | 9 (0.25) |
| same, best of 180 parameter sets tuned on this test set | 40 | 18 (0.45) |

Metres and velocity **do not beat** the pixel stitcher, even when tuned on the test set.
The likely cause: occluded boxes have a false bottom edge, so the foot point and its
velocity jump by metres.

### Human taps (simulated with my labels as the answers)

Fragments are first classified against the start galleries (DINOv2-base, 0.72 of the 46
samples right with no taps). Each tap asks about one fragment; the simulated scout answers
with my label at one of its sampled times, and "can't tell" leaves it open.

| taps per match | 0 | 10 | 20 | 25 | 30 | 36 | 40 | 46 |
|---|---|---|---|---|---|---|---|---|
| accuracy, least-confident first | 0.72 | 0.72 | 0.72 | 0.80 | 0.89 | **0.95** | 0.96 | 0.98 |
| accuracy, random order | 0.72 | 0.76 | 0.76 | 0.85 | 0.89 | 0.89 | 0.98 | 0.98 |

- **About 36 taps per match** reach 95 %, and picking the least-confident fragments first
  is no better than random (the DINOv2 margin is not a usable uncertainty).
- The 95 % is over **the samples a human can identify at all**, i.e. 27 % of robot × time.
  The other 73 % stays unknown however many taps are spent. It is also partly circular:
  the simulated answers are the same labels the score uses.
- The CLIP join score (AUC 0.86) was not used for tap selection after the margin result;
  it would select joins, but the tap budget is set by the fragment count anyway (46
  fragments with any identifiable sample, 117 in total).

Client: `tags.py` posts the crop and the request and reads answers. Exercised against a
throwaway local fgc-scout (image, request, idempotent re-post, answer, read-back); nothing
was posted to the live server.

### Pit photos

`GET /api/state` on the live fgc-scout lists 178 teams and **0 photos** (checked
2026-10-04 20:00 UTC), so the pit-photo prior could not be measured. It would need
photos of the 2026 robots first.

### Driving metrics in metres (`motion.py`)

Implemented: path, mean and p90 speed, p90 acceleration, RMS jerk, idle time. They are
computed per continuous run (no bridging over gaps over 0.35 s), on a 0.5 s moving
average. On t2-1, using only the fragments I verified by hand:

| robot | seconds observed (of 150) | path m | mean speed m/s | idle s |
|---|---|---|---|---|
| A | 82 | 19.5 | 0.24 | 47 |
| C | 28 | 9.5 | 0.34 | 8 |
| F | 22 | 10.7 | 0.49 | 6 |
| D | 21 | 0.8 | 0.04 | 20 (parked at the rail) |
| E | 11 | 1.1 | 0.10 | 7 |
| B | 0 | – | – | – |

These are what the footage supports for the *best* case: a hand-identified track, and
still at most 82 of 150 s per robot. That is not a driver-precision measurement.

### Shots

Not built. In 2025 the scoring counts are per ecosystem, shared by both alliances, so
they would only validate a robot-agnostic shot counter. Attributing a shot to a robot
needs the identity that the sections above show is missing. For 2026 a shot counter can
be validated against per-alliance SUPPRESSION UNIT counts if the API reports them, as an
anonymous or per-alliance metric.

### v2 verdict

Usable identity is not reachable on 2025 stream footage with tapping. About 36 taps per
match would make the identifiable 27 % of robot-time 95 % right, and the other 73 % is
either invisible to the camera (55 %) or not nameable by a human from the crop (18 %).
Metres work on 3 of 7 cameras; motion-model linking in metres is worse than pixels.

## 2026 Day 1 re-measurement (2026-10-08)

Three Day-1 ranking matches (t2-66 and t2-63 on field 3, t2-46 on field 2), 1080p60,
fetched by DASH fragment (`dash.py`: `--download-sections` fails on post-live DASH with
ffmpeg exit 183; fetching the ~40 fragments of a match window takes 36 s). Field 5 was
live again and exposed no fragment list, so it was skipped.

**Camera.** A fixed high wide shot from the front, as Bogdan expected. The goals stand
together at the back, so the 2025 towers in mid-field are gone. Robots in the open field
are rarely hidden. What hides them now is the referees and camera crew at the front rail,
and the pile-up at the tops of the two BRACE pipes at the end.

**Sync.** The 2026 scorebug needs a tighter timer ROI (`overlay.use_year(2026)`). With
it, 148/149 one-per-second readings agree. Overlay t0 vs matchwatch: +1.2 to +1.9 s.

**Detection** (31 hand-labelled robot centres, 6 frames; `labels/robots_2026.json`):

| model | P | R | F1 |
|---|---|---|---|
| YOLO11s fine-tuned on 2026 OWLv2 pseudo-labels (6 other Day-1 matches, 40 epochs) | 0.56 | 0.74 | **0.64** |
| OWLv2 zero-shot | 0.64 | 0.52 | 0.57 |
| 2025-trained YOLO11s | 0.86 | 0.39 | 0.53 |

**Visibility and identity**, scored on the crop sheets (`bench/crop_sheet26.py`, every
in-field box at 8–10 times per match, judged by eye; `labels/gt26_sheet.json`):

| | t2-66 | t2-63 | t2-46 | 2025 t2-1 |
|---|---|---|---|---|
| robot × time with a correct tracker box | 0.72 | 0.69 | 0.65 | 0.45 |
| robot × time visible to me (t2-46 only, rough count at 860 px) | – | – | ≈0.96 | – |
| robot × time I could name **as a team** | 0.22 | 0.10 | 0.21 | – |
| robot-seconds in tracks of 20 s or more (of 900) | 402 | 211 | 414 | – |
| crops on the right robot in those long tracks | 65/80 | – | 90/110 (+ 1 track on a statue) | – |

How a team got named: VAN and COK (pit photo, plus COK's zone-2 and SRB's zone-3 end
positions on the red pipe); AUT and MEX (pit photo); BIH (pit photo and its large flag
panel); NAM (flag on the robot). GRE has a pit photo but I could not find it in the stream.
Teams without a pit photo or a distinctive flag stay anonymous.

**Pit photos.** For a human they help. In these 3 matches I matched 5 of the 6 pit-photo
teams to stream robots; the name-able share goes from ≈0.10 to ≈0.21 in t2-46. For a
model they do not help: DINOv2 / CLIP-L retrieval from pit photo to stream crops gives
precision@k 0.0, 0.0 and 0.2 against chance 0.09, 0.06 and 0.16 (`bench/pit_reid.py`).
The domain gap (close-up phone photo vs. a 100 px crop) is too large.

**Climbs** (`climb26.py`: box centre within 70 px of the hand-clicked pipe segment,
progress along it). Officially 16 robots touched or climbed a BRACE in these matches
(brace state > 0). The detector found 11 climb segments: 7 on red pipes and 4 on blue.

| | red pipes | blue pipes |
|---|---|---|
| official climbers | 8 | 8 |
| detected climb segments | 9 (incl. at least 2 spurious in t2-63) | 2 |

Timing against my hand timing at 3 s steps, on t2-66's red pipe (the one pipe where
both climbers were detected):

| climber | hand | detector | start / end error |
|---|---|---|---|
| zone 3 (SRB by end position) | 96 → 106 s | 99.1 → 109.9 s | +3 / +4 s |
| zone 2 (COK by end position) | 139 → 147 s | 133.9 → 149.7 s | −5 / +3 s |

On t2-66's blue pipe, 3 hand-timed climbs (≈9 s each) gave 0 detections: the pipe base is
behind the referees and the blue climbers were not tracked. At the top, zone-3 robots
overlap one another, even for a human; I under-counted the official climbers by about 1
robot in 4 of 6 alliances.

**2026 verdict: no-go for per-team driving stats on Fri/Sat matches.** Detection and
visibility are much better than 2025. Identity is still the limit: only robots with a
pit photo, a flag or an end anchor can be named, and that is about 10–22 % of robot-time.
Climb timing works only when the pipe base is in clear view (2 of 5 climbs on one match,
±3–5 s).

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
