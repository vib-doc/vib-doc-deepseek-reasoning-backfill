# Swallowed replies in tool-using reasoning models

**Reasoning emitted, content never produced — and what `reasoning_content` backfill has
to do with it.**

A field note from running a long-lived assistant on DeepSeek Harness + `deepseek-flash`
with tools enabled. Everything below is measured on real traffic. **The scripts that
produce each number are in this repo; the session archives they read are not** — the
transcripts are private, so the tools are written to run against your own archives under
`~/.dsh/sessions/`. Numbers quoted here are from one deployment and will differ.

---

## 1. The symptom

Occasionally a turn produces **no visible reply at all**. The raw response stream shows
why:

```
chunk(block-start, blockType=reasoning)
reasoning-chunks        ← the model thinks, normally
chunk(block-end)
chunk(usage)
chunk(finish)           ← reason.kind = "error", failure.code = EMPTY_RESPONSE
```

There is no `text` block anywhere in the stream. The model reasoned — often *fully*,
with the answer already drafted inside the reasoning — and then finished without ever
opening a content block.

```
failure.message = "model returned a completed response with no content"
usage.outputTokens == usage.reasoningTokens      # all of it went to reasoning
```

Three things this is **not**:

* **not truncation** — that would be `finish.reason.kind = "length"`;
* **not a content problem** — retrying the same step usually succeeds;
* **not a length problem** — measured chain lengths ranged from **56 to 6,657** output
  tokens (a 100× spread) and all failed the same way. Successful retries were often
  *shorter* than the attempts that were swallowed.

### How often, and in what shape

`tools/silence_shape_report.py` reads the session archives (`--empty-only`):

```
date     events  turns  EMPTY  other   shape
09-20         9      3      9      0   sticky (>=2 in a row) (longest run 4)
09-22         3      3      3      0   isolated               (longest run 1)
09-23        19     10     19      0   sticky (>=2 in a row) (longest run 5)
```

Two notes so the table is not read as curated: it lists **only days with an
`EMPTY_RESPONSE`** — run the tool without `--empty-only` and other retry reasons
(transport, server) add four more days that never show this symptom. And these events
are **not all from one session**: 29 belong to the long session discussed here, 1 and 1
to two others. The shape is what generalizes, not the count.

**29 swallowed responses** in one long session over ~3 days. Note the two shapes:
sometimes a single isolated swallow, sometimes *the same step swallowed 4–6 times in a
row*. When a step is swallowed repeatedly and the retry budget (5) runs out, the user
experiences a turn that simply never answers. In the recorded case the same user
message was then re-sent, and the assistant — which cannot see retry events — read the
two adjacent user messages as *"the user sent two messages"*. The failure silently
rewrote the shape of its own context.

The reason this was worth chasing to the bottom: **in the empty turns, someone was
sitting at the screen waiting.** Everything below is machinery; that is the part that
made it urgent.

### The tail of a swallowed chain

Swallowed chains typically degenerate at the end into a self-prompt loop:

```
write.  -> output.   write.  -> output!   (write.)   (output.) x40 ...
```

**This loop is not a predictor.** Successful chains show the same tail. It is a
symptom, not a signal — do not build a detector on it.

---

## 2. Why tool-using reasoning models are exposed

This is the part that generalizes beyond one harness.

The DeepSeek docs are explicit ([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)):

> If the request **carries the `tools` parameter**: the `reasoning_content` of all
> previous turns **should be passed back** to the API and **will be concatenated into
> the context**.

So in any tool-using agent loop, **the model re-reads its own past reasoning on every
request**. Two consequences:

1. **Cost.** Measured here: ~**280,000 tokens per request**, the overwhelming majority
   of it cache reads of accumulated reasoning.
2. **A self-conditioning channel.** Not only the *conclusions* of past turns persist
   (those are in the visible replies), but the *reasoning that produced them*. For a
   model known to ruminate, that is a lot of its own reflection to re-read.

The relevant published finding is *DeepSeek-R1 Thoughtology*
([arXiv:2504.07128](https://arxiv.org/abs/2504.07128)), which names and quantifies this:

> we find a tendency for DeepSeek-R1 to **persistently ruminate on previously explored
> problem formulations**, obstructing further exploration

Their operational definition of rumination is worth quoting because it is directly
measurable: the **5-gram repetition rate** of the reasoning chain (their Figure 3.3b
plots it per task, range 0–35%; the figures below strip whitespace before counting, see
§2's matched-pair table). They also note that *"a high time spent in
reconstruction does not necessarily mean high rumination"* — duration is not the signal,
repetition is.

### What the data here does and does not support

`tools/rumination_compare.py --session <id>` (this counts **every retried attempt**,
not only `EMPTY_RESPONSE`; 29 of the 31 in this session were `EMPTY_RESPONSE`):

| comparison | swallowed | successful |
|---|---|---|
| median chain length, all data | 2,431 chars | 2,557 chars |
| 5-gram repetition, **all data** | **20.7%** | 13.5% |
| **matched within a step** (13 pairs) - repetition | **20.7%** | **20.8%** |
| **matched within a step** (13 pairs) - length | 2,526 chars | 2,167 chars |
| matched within a step - all seven metrics | see below | see below |

* **Matched within a turn** (same input, different sample): *nothing separates them.*
  `rumination_compare.py` scores both members of every pair on **seven metrics** — chain
  length, line count, longest repeated-line run, tail repetition share, 20-char shingle
  repetition, 5-gram repetition (the paper's rumination measure), and self-referential
  word density — then reports how many differ by more than 20% of the larger median.
  Measured: **0 of 7**. Two conventions matter and are stated rather than assumed:
  **one pair per (session, turn, step), using the first swallowed attempt** — one step was
  swallowed 6 times, three others 4 times, two more twice (35 attempts over 19 steps), and
  taking the last attempt instead changes the values while keeping the same row count, which
  is exactly the kind of thing that makes a reader suspect the table; and **5-gram /
  shingle rates are computed with whitespace stripped**, a choice worth ~8 points here
  (20.7 / 20.8 stripped vs 28.3 / 28.8 unstripped — both support the same conclusion, but
  an independent implementation needs to know which convention produced these figures).

  ```
  metric                 swallowed  successful  separates?
  chars                    2526.00     2167.00          no
  lines                      85.00       97.00          no
  max_line_run               32.00       35.00          no
  tail_repeat_%             100.00      100.00          no
  shingle20_%                 7.02        6.25          no
  ngram5_%                   20.68       20.78          no
  selfref_per_1000            0.58        0.60          no
  ```

  Two of the seven are degenerate on this data and are reported for completeness:
  `tail_repeat_%` is ~100 in **both** groups because every chain ends in the same short
  self-prompt loop, and `max_line_run` (~32 vs ~35) *is* that loop — present in successful
  chains too, which is exactly why it cannot serve as a detector. The self-referential
  word list is language-specific (Chinese here); pass `--self-ref-words` for other
  languages, or that row reports ~0 and looks like a clean negative result when it is only
  a mismatched word list.

  **This is the strongest single result here**: at the level of the visible chain, there
  is no warning sign. Re-sampling is what fixes it.
* **Across the whole dataset**: swallowed contexts carry ~1.6–3.3× the density of
  self-referential wording, and ~1.6× the reasoning repetition. That says *which kinds
  of conversations* swallow more often — long, reflective, self-referential ones — not
  which attempt will fail.

So: **topic sets the probability; sampling decides the instance.** Two mechanisms that
people reach for first were tested and rejected: it is not "it thought too much to
finish" (swallowed chains were not longer — often shorter), and it is not "the context
got too big" (swallowed steps had *smaller* contexts, and showed no clustering after
context compaction).

---

## 3. Narrowing the backfill

If you cannot predict the failure, the next best move is to change the conditions it
happens under — and the backfill channel is the one condition a client actually
controls.

### The probe first: is the docs' "must" enforced?

The docs threaten a 400 if reasoning is not fully passed back. That claim decides
whether a client is *allowed* to fold old reasoning out. So we tested it
(`probes/probe_backfill_enforcement.py`, two requests):

```
[C tool-turn, reasoning stripped]  HTTP 200   ← docs say this must be 400
[A plain turn, reasoning stripped] HTTP 200
```

**The server did not enforce it.** Stated with its limits: one model, one endpoint,
sampled once. It does not mean the field will stay optional — keep the `--revert` path.

### Three designs, two of which failed structurally

Worth recording, because both failures are easy to walk into:

**v1 — "backfill if this turn has already called a tool."** The predicate flips *inside*
one turn:

```
step 1 (no tool yet)  142,563 tokens
step 2 (tool called)  311,237 tokens     +169k, mid-answer
```

Two problems. It **splits the KV-cache prefix**, so the second request largely
recomputes. And it changes what the model can see *of itself* halfway through producing
one answer. A further bug: any mid-segment user message was treated as a turn boundary,
flipping the mode back and forth (`324,393 → 150,120 → 326,701`).

**v2 — "decide once per segment, from the previous segment."** No more flipping, but the
predicate now contradicts what the segment actually does:

```
segments: 678   (segment = everything after the last final reply)
wrong:    184  = 27.2%
  previous used tools / this one doesn't  → backfill ON   92
  previous didn't     / this one does     → backfill OFF  92
```

And this is **structural, not an implementation slip**: to hold the mode constant for a
segment you must choose it at the segment's start — at which point the segment has not
happened yet. Only the previous segment is available. (Measured against the
alternatives: always-on is right 59.9% of the time, always-off 40.1%, previous-segment
72.8%. Definitely better — and definitely still wrong a quarter of the time.)

**v3 — drop the global switch; decide per message.** This is what the patch implements:

> **Only assistant messages that carry a tool call, within the current segment, keep
> their reasoning. Everything older is stripped.**

* no prediction needed — the disagreement between v1 and v2 disappears;
* the expensive flip disappears: within a turn the payload only grows (measured over 16
  consecutive steps, every delta positive), where v1 and v2 alternated between two sizes
  twice per turn — each alternation invalidating the cached prefix
  (see *Measured effect* below for the numbers, and for what does **not** go to zero);
* the fuel is cut where it is thickest: the long, reflective reasoning lives on the
  *final reply* side of a segment, and none of it enters history.

**How long reasoning actually survives — say it precisely.** Because the segment
boundary *is* the final reply, the next turn starts with every message of the previous
turn behind `segStart`. So:

> **Reasoning lives exactly one turn.** It is kept for the tool steps of the turn that
> produced it (so the tool loop keeps its "why am I calling this" working state), and
> it is stripped from every earlier turn — including the tool steps of those turns.

An earlier draft of this note claimed the model "can still see why it fetched something
last time". **That was wrong**, and it matters for what you expect to observe: there is
**no cross-turn tool-reasoning continuity** in this design. If a task spans several
turns and the user says "continue", the model re-derives its plan from the visible
replies and tool *results*, not from its previous reasoning. That is the cost side of
the trade, and it is what the §4 measurement is meant to catch.

Measured directly (a two-turn sequence, first turn has a tool step):

```
turn 1 in progress   [tool step, final reply]              -> tool step's reasoning KEPT (segStart = 0)
turn 2 begins        [t1 tool step, t1 reply, t2 tool step]-> false, false, TRUE
```

### Measured effect

**Do not quote a single "before" number.** The pre-patch request size was bimodal, not
flat: it alternated between roughly 143k and roughly 357k *between adjacent turns* (and
sometimes *within* one turn). A mean over that period describes neither mode. Measure a
matched pair instead — consecutive turns, same session, restart in between:

```
turn 698 (last full turn before v3, tool-dense)  step 1   357,268
                                                 step 2   359,555
turn 699 (first full turn under v3, 16 steps)    step 1   166,777
                                                 step 2   167,253
                                                 ...
                                                 step 16  189,916
```

So: **~357k → ~167k at the same point in a working turn** (about half), and the growth
within the turn becomes monotonic instead of a sawtooth:

```
before, one turn:   142,563 -> 311,237        (+168,700 mid-turn, twice back and forth)
after, one turn:    166,777 -> 189,916        (+23,139 over 16 steps, never negative)
```

`tools/prefix_switch_report.py` measures the flip directly: count requests whose payload
is **smaller** than the previous one by more than a threshold (default 10,000 tokens),
excluding **the first request after a compaction event** (a compaction legitimately cuts
the payload — one recorded case went 792k → 16k; that is the client discarding history,
not a prefix switch). The exclusion is that exact position rather than a time window: a
window wide enough to be safe (30 min) also swallows real flips happening after a
compaction, and a narrow one (2 min) misses the same legitimate shrink when the user walks
away after a compaction. Measured on this deployment, all 22 exclusions were within 87s of
a compaction **and all 22 were the first request after it**, so the exact test reproduces
every case a window got right.

The last line of the table is **provisional while a turn is running** — archives are
append-only and a turn in progress keeps appending, so the newest boundary's figure can
change (measured: one boundary read smaller mid-write than its settled value). Re-run
after the turn ends before quoting it.

```
criterion: totalTokens(N+1) - totalTokens(N) <= -10,000
session REDACTED-SESSION, 2026-09-23, 112 requests scanned -> 7 shrinking steps

11:32:31  turn 684   310,072 -> 142,563   (-167,509)   \  pre-patch: the payload
11:33:10  turn 685   311,237 -> 143,098   (-168,139)    |  repeatedly halves and
11:47:15  turn 691   318,616 -> 147,670   (-170,946)    |  doubles, twice per turn
11:48:36  turn 691   324,393 -> 150,120   (-174,273)    |
12:07:08  turn 694   341,727 -> 159,025   (-182,702)   /
12:41:48  turn 699   359,555 -> 166,777   (-192,778)   <- v3 takes effect; first request
                                                          strips the whole history (once)
13:05:15  turn 700   189,916 -> 179,907    (-10,009)   <- turn boundary: the previous
                                                          turn's tool reasoning leaves
```

**Be precise about the zero.** "Prefix switches drop to 0" would be wrong, and an earlier
draft of this note said it. What actually happens:

* **mid-turn flips are gone** — within a turn the payload only grows (16 steps measured,
  every delta positive). This was the expensive failure mode: two ±170k flips per turn,
  each one invalidating the cached prefix;
* **a small step down remains at every turn boundary**, equal to the previous turn's
  tool-step reasoning (measured: −10,009, −1,967, then +292 and +214 where the new user
  message outweighed the stripped reasoning). That step is **inherent** to any rule that
  strips past reasoning: the moment the segment boundary passes, those messages change,
  so the cached prefix ends there. It is one step per turn instead of two large flips,
  and it is bounded by how much reasoning the turn produced.

Roughly **half the tokens**, with no observed loss of task continuity.

---

## 4. The side effect, and its ruler

Stripping past non-tool reasoning means the model can no longer re-read it. The
plausible cost is **repeated labour** — re-reading a file, re-running a command,
re-deriving something it already concluded. That is measurable, which is the only
reason it is acceptable to take the risk.

`tools/repeat_labor_report.py` builds a key for every tool call (tool + arguments) and
reports two numbers. **Only one of them can be used as a trend**, and getting this wrong
is easy:

| number | what it is | usable as a trend? |
|---|---|---|
| **groups** | keys appearing in more than one segment | **No.** A group is a property of the whole history; the total grows with the archive, so it "rises" forever |
| **returning calls / 1000 calls** | a call whose key already appeared in an earlier segment **of the same session** | **Yes.** Denominator is that day's calls, so archive growth does not move it |

```
tool calls scanned: 16,887
  within a segment            : 184   (re-checking, not counted)
  cross-segment, same window  : 244   <- stock, not a trend
  cross-segment, global       : 1525  (legacy grouping; counts two windows each
                                       reading the same file once as a "repeat")
  polling commands excluded   : 4

returning calls per 1000      : 91.3

per day (returning / calls = per 1000):
  09-17    3/154  =  19.5      09-21  108/1110 =  97.3
  09-18    1/72   =  13.9      09-22   48/843  =  56.9
  09-19    0/62   =   0.0      09-23   13/281  =  46.3
  09-20   28/746  =  37.5
```

**Cross-day rates are not directly comparable** — that table spans 0 to 232 per 1000
over two weeks. Comparing them requires holding the *kind of work* constant: a day spent
reading files scores high by construction, a day spent running commands scores low. So:

* compare **matched windows** (same session, similar work), not arbitrary days;
* prefer the pre/post pair around a change over a time series.

And a correction to an earlier draft of this note: it explained a pre-patch decline in the
rate as "the sampling structure changed" — several session branches sharing one history,
each counting the same repeat. That explanation **was tested and does not hold**: looking
at a single branch's own rate, the series is not monotonic (it falls 136 → 0 from 09-10 to
09-19, then rises to 107–221 on 09-20 … 09-22), and branch dilution cannot produce the
rise. **It is recorded here as unexplained**, not as a finding.

Three traps were hit while building this, all of which produced confidently wrong reads:

* **Grouping keys globally** counted two windows each reading the same file once as one
  "repeat". That is two agents doing their own work, not one agent repeating itself —
  it inflated the number ~6× (1,524 vs 244).
* **A per-day table that credited "spans days" to every day except the group's last.**
  The last day of any group could therefore never receive credit, so **new repeats were
  booked against old rows** — a table that reads "it is getting worse" no matter what
  happens. (One real group spans 09-11 … 09-23 and was credited to 09-11.) Replaced by
  the per-day rate.
* **A polling filter that matched nothing.** The pattern ended in a literal `\b` instead
  of a word boundary (so even `ls -la` failed to match), was anchored at `^` (missing the
  common `cd x && ls` form), and its curl branch required `/dev/null` in a fixed position.
  The line "polling excluded: 0" was a no-op reported as a finding.

Recommended use: fix a baseline, then re-run over the window since the last check and
compare **rates**, not totals.

```
baseline (before the change, one session):  50 cross-segment groups
every few days:  --since <last check>   ->  read returning calls per 1000
```

Read it as a trend, not a verdict: reading a file twice is not automatically waste (it
may have changed, or it may be a deliberate re-check). This counts **traces**; a human
decides what they mean. **If the rate climbs after deploying `patch_backfill.py`, revert.**

---

## 5. Using the pieces

```bash
# 1. test whether the API still tolerates reduced backfill on your model/endpoint
DEEPSEEK_API_KEY=sk-... python3 probes/probe_backfill_enforcement.py

# 2. look at your own swallowed replies
python3 tools/silence_shape_report.py --shape            # per-attempt shapes
python3 tools/silence_shape_report.py --watch            # log turns that produced nothing

# 3. apply the narrowed backfill to DeepSeek Harness (restart required)
python3 patches/patch_backfill.py --check
python3 patches/patch_backfill.py
python3 patches/patch_backfill.py --revert               # undo

# 4. watch the side effect
python3 tools/repeat_labor_report.py

# 5. count payload shrinkages (prefix invalidations)
python3 tools/prefix_switch_report.py

# 6. compare the chains of swallowed vs. successful attempts
python3 tools/rumination_compare.py --session <id>
```

The patch targets `@deepseek-ai/dsh-llm-deepseek` and refuses to run if the anchors it
needs have moved. It also **refuses to guess**: every substitution must match exactly
once, and the file is syntax-checked afterwards with an automatic rollback. Vendor
patches get overwritten by `npx` upgrades — that is the normal failure mode, and a
`--check` in a scheduled job is the cheap guard.

Overrides, highest priority first:

| control | effect |
|---|---|
| `~/.dsh/backfill_mode` = `on` / `off` / `auto` | forces the mode for A/B comparison |
| `【工作】` / `【聊天】` at the start of a user message | per-segment override; stripped before sending |
| the rule | per-message, as described in §3 |

---

## 6. What we still don't know

* **Why the content block never opens.** The proximate description is "sampling fell
  back into the reasoning block at the boundary where content should start, then looped
  until finish". That is a description, not an explanation. Retrying works, which says
  the boundary is a sampling-sensitive position — but nothing here identifies the cause.
* **Whether narrowing the backfill actually reduces the failure rate.** The token
  effect is immediate and large; the failure-rate effect needs weeks of traffic. The
  baseline is ~3% of delivered responses with a bursty distribution, so a few quiet days
  prove nothing. (Early signal, explicitly not evidence: after deployment there was one
  swallow two minutes in and none in the following 78 minutes of heavy tool use.)
* **The `reasoning_content` requirement on other models and providers.** One 200 is one
  200. If you rely on reduced backfill, probe your own endpoint — and keep the revert.

## 7. Context

* Harness: DeepSeek Harness (`@deepseek-ai/dsh-llm-deepseek` 0.1.5-rc.2)
* Model: `deepseek-flash`, thinking enabled, `reasoning_effort: max`
* Scale: one long-lived session, ~680 turns, ~3,500 tool calls
* Working copy this note was produced from: `the working copy`
  — the scripts are meant to be run from the repo root and read the archives under
  `~/.dsh/sessions/`
* Prior art: *DeepSeek-R1 Thoughtology: Let's think about LLM reasoning*
  ([arXiv:2504.07128](https://arxiv.org/abs/2504.07128)) — rumination, the "sweet spot"
  of thought length, and the 5-gram repetition measure used above

## License

MIT
