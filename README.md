# Swallowed replies in tool-using reasoning models

**Reasoning emitted, content never produced — and what `reasoning_content` backfill has
to do with it.**

A field note from running a long-lived assistant on DeepSeek Harness + `deepseek-flash`
with tools enabled. Everything below is measured on real traffic; the scripts to
reproduce each number are in this repo.

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

`tools/silence_shape_report.py` reads the session archives:

```
date     events  turns  EMPTY  other   shape
09-20         9      3      9      0   sticky (>=2 in a row) (longest run 4)
09-22         3      3      3      0   isolated               (longest run 1)
09-23        19     10     19      0   sticky (>=2 in a row) (longest run 5)
```

**31 swallowed responses** in one long session over ~3 days. Note the two shapes:
sometimes a single isolated swallow, sometimes *the same step swallowed 4–6 times in a
row*. When a step is swallowed repeatedly and the retry budget (5) runs out, the user
experiences a turn that simply never answers. In the recorded case the same user
message was then re-sent, and the assistant — which cannot see retry events — read the
two adjacent user messages as *"the user sent two messages"*. The failure silently
rewrote the shape of its own context.

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
plots it per task, range 0–35%). They also note that *"a high time spent in
reconstruction does not necessarily mean high rumination"* — duration is not the signal,
repetition is.

### What the data here does and does not support

Measured on 31 swallowed vs. 713 successful chains (`--shape`, plus offline analysis):

| comparison | swallowed | successful |
|---|---|---|
| median chain length | 2,431 chars | 2,559 chars |
| 5-gram repetition, **all data** | **19.7%** | 12.2% |
| 5-gram repetition, **matched within turn** (13 pairs) | **20.7%** | **20.8%** |

* **Matched within a turn** (same input, different sample): *nothing separates them.*
  Five different text metrics all failed to distinguish a swallowed attempt from the
  retry of the same step that succeeded. **This is the strongest single result here**:
  at the level of the visible chain, there is no warning sign. Re-sampling is what
  fixes it.
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
* **prefix switches drop to 0** (v2 had 187; an "always start false" variant would have
  had 818 — each one a chance to invalidate the cache);
* the fuel is cut where it is thickest: the long, reflective reasoning lives on the
  *final reply* side of a segment, and none of it enters history.

### Measured effect

```
before   280,003 tokens/request (median-ish, 40 turns)
after    143,000 tokens/request
```

And the shape of a long working segment becomes flat instead of a sawtooth:

```
before (v2, restart pending)   turn 698            357,268
after  (v3)                    turn 699  step 1    166,777
                                        step 2    167,253
                                        ...
                                        step 16   189,916     +23k over 16 steps
```

Roughly **half the tokens**, with no observed loss of task continuity. Flat growth
within a segment; no mid-turn steps.

---

## 4. The side effect, and its ruler

Stripping past non-tool reasoning means the model can no longer re-read it. The
plausible cost is **repeated labour** — re-reading a file, re-running a command,
re-deriving something it already concluded. That is measurable, which is the only
reason it is acceptable to take the risk.

`tools/repeat_labor_report.py` counts tool calls whose key (tool + arguments) recurs in
**different segments**, excluding polling commands (`ls`, `ps`, `git status`, health
checks) which are *supposed* to repeat:

```
tool calls scanned: 3,508
  across segments : 50
  within a segment: 85   (normal re-checking)
  polling excluded: 0
  => headline: 50 cross-segment repeat groups
  by type: {'read': 39, 'run': 6, 'job_output': 3, 'edit': 1, 'list_agents': 1}

per day:   09-20  6    09-21  5    09-22  2    09-23  0
```

Read it as a trend, not a verdict: reading a file twice is not automatically waste (it
may have changed, or it may be a deliberate re-check). This counts **traces**; a human
decides what they mean. The pre-deployment baseline is ~0–6 groups/day, dominated by
re-reads. **If that number climbs after deploying `patch_backfill.py`, revert.**

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
* Prior art: *DeepSeek-R1 Thoughtology: Let's think about LLM reasoning*
  ([arXiv:2504.07128](https://arxiv.org/abs/2504.07128)) — rumination, the "sweet spot"
  of thought length, and the 5-gram repetition measure used above

## License

MIT
