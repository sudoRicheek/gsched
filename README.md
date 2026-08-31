# gsched

A tiny GPU job scheduler. One small daemon per machine owns a list of GPUs,
polls every few seconds, and hands each **queued** job the GPUs it asked for as
they come **free**. Submit jobs from anywhere on the box; they queue and run as
GPUs open up. That's the whole idea.

- **One file** (`gsched.py`), one dependency (`rich`).
- **1 GPU or 8:** `--ngpu N` gets a job N GPUs at once, all in its
  `CUDA_VISIBLE_DEVICES`; single-GPU jobs fill in around it.
- **Shared-box friendly:** only uses the GPUs you name, treats a GPU as free
  only when its memory is actually low, and never touches other processes.
- **No server:** all state is a single SQLite file (`~/.gsched/gsched.db`) that
  the daemon and the CLI share.
- **It tells you when jobs finish** — a line straight back to the terminal that
  submitted them, plus any hook you want (Slack, ntfy, email). No polling.

## Install

```bash
pip install git+https://github.com/sudoRicheek/gsched.git
# or, from a clone:
git clone git@github.com:sudoRicheek/gsched.git && cd gsched && pip install -e .
```

This gives you a `gsched` command. (No install needed either — you can always
just run `python gsched.py ...`.)

## Quickstart

**1. Start the daemon** (once per machine, in the background). Name the GPUs it
may use; export any env vars your jobs need first — the daemon passes its
environment to every job.

```bash
nohup gsched daemon --gpus 0,1,2,3 > ~/.gsched/daemon.log 2>&1 &
```

**2. Submit jobs.** The daemon sets `CUDA_VISIBLE_DEVICES` for you, so write a
job **as if it owns GPU 0** (or GPUs `0..N-1` when you ask for `--ngpu N`).

```bash
gsched submit "python train.py --lr 3e-4" --name run-a      # one GPU (default)
gsched submit "torchrun --nproc-per-node 4 train.py" --ngpu 4 --name ddp
gsched submitf jobs.txt                                     # a whole file of jobs
```

**3. Get told when they're done** (nothing to set up — see
[Notifications](#notifications)). The line appears in the terminal you
submitted from, whatever you're doing in it:

```
$ gsched submit "python train.py --lr 3e-4" --name run-a
queued job 7 (will notify /dev/pts/3)
...
[gsched] job 7 'run-a' done rc=0 on GPU0 after 1:12:33  (/home/you/.gsched/logs/job7.log)
[gsched] job 8 'ddp' done rc=0 on GPUs 0,1,2,3 after 4:20:07  (/home/you/.gsched/logs/job8.log)
```

**4. Or watch.**

```bash
gsched status
```

```
● alive  daemon pid 12345  gpus 0,1,2,3  poll 7s  mem_free<2000MiB
 GPUs                            running                            queued
 gpu  mem    job                 id name  gpus  pid   elapsed       # id name       ngpu
   0  38210  #8 ddp (2 GPUs)      8 ddp   0,1   9002   0:31:02      1 11 sweep-lr-0    1
   1  38210  #8 ddp (2 GPUs)      7 run-a 2     9001   0:12:33      2 12 big-eval      4
   2  17800  #7 run-a
   3      1  idle
queued 2  running 2 (3/4 GPUs)  done 0  failed 0  evicted 0
```

## Commands

| command | what it does |
|---|---|
| `daemon --gpus 0,1,2,3 [--poll 7] [--mem 2000] [--notify "<cmd>"] [--backfill]` | run the scheduler loop (background it) |
| `submit "<cmd>" [--ngpu N] [--name N] [--notify "<cmd>"] [--no-tty]` | queue one command |
| `submitf <file> [--ngpu N] [--name N] [--notify "<cmd>"] [--no-tty]` | queue every line of a file (`name<TAB>cmd` or `name<TAB>ngpu<TAB>cmd`; `#`/blank lines skipped) |
| `status [--recent K]` | dashboard: daemon health, per-GPU, running, queued, last K finished |
| `gpus 0,1,2,3,4,5` | change the allowed GPU list **live** (no restart) |
| `notify ["<cmd>"] [--clear] [--test]` | show/set/clear the machine-wide finish hook |
| `evict <id>` | cancel a queued job, or `SIGTERM` a running one (all its GPUs free for the next) |
| `logs <id> [-n K]` | tail that job's stdout/stderr |
| `rm --done` / `rm <id>` | forget finished rows |

## Multi-GPU jobs

`--ngpu N` asks for **N GPUs at once**. The daemon only starts the job when N of
its GPUs are free together, then puts all of them in that job's
`CUDA_VISIBLE_DEVICES` — so the job sees them as `cuda:0 … cuda:N-1` and never
learns (or needs to care) which physical GPUs it got:

```bash
gsched submit "torchrun --nproc-per-node 4 train.py" --ngpu 4 --name ddp
gsched submit "python train.py --world-size 2" --ngpu 2
```

```
launched job 12 (ddp) on GPUs 1,2,4,5 pid=41221     # CUDA_VISIBLE_DEVICES=1,2,4,5
```

In a batch file, put the count between the name and the command:

```
# name<TAB>ngpu<TAB>command      (ngpu is optional; --ngpu sets the file default)
ddp-a	4	torchrun --nproc-per-node 4 train.py --lr 1e-4
ddp-b	4	torchrun --nproc-per-node 4 train.py --lr 3e-4
probe		python eval.py                                  # 1 GPU
```

### How the queue handles a mix of sizes

The queue is strictly FIFO, and **a job that doesn't fit yet holds the GPUs**
rather than being overtaken. So a 4-GPU job waits while 4 GPUs gradually free up
instead of being starved forever by a stream of 1-GPU jobs. The cost is that
those GPUs idle while it waits.

If you'd rather keep every GPU busy and accept that a big job may wait longer,
start the daemon with `--backfill` — smaller jobs then run past a blocked one:

```bash
nohup gsched daemon --gpus 0,1,2,3 --backfill > ~/.gsched/daemon.log 2>&1 &
gsched daemon --gpus 0,1,2,3 --no-backfill    # ... and back to strict FIFO
```

The setting sticks across daemon restarts (like the notify hook), and `status`
shows it in the header line.

Two things worth knowing:

- Asking for more GPUs than the daemon manages (`--ngpu 9` on a 4-GPU daemon)
  warns at submit time. The job stays queued and is **skipped over** rather than
  blocking everyone — `status` flags it in red. Widen the daemon with
  `gsched gpus ...` and it becomes runnable.
- GPUs handed to one job aren't necessarily adjacent — you get whichever N are
  free. If a job needs specific topology (NVLink pairs, say), run a daemon that
  only manages those GPUs.

## Notifications

Nobody should have to babysit `status`. When a job reaches a final state
(`done` / `failed` / `evicted`) the daemon announces it two ways.

**1. Back to your terminal — on by default, nothing to configure.** `submit`
records the tty you ran it from, and the daemon writes the result line straight
to it (with a bell). It works over SSH, across reconnects to a `tmux`/`screen`
pane, and while you're in the middle of doing something else in that shell —
because it's the terminal being written to, not your program. If the terminal
is gone by then, the write is silently skipped. `--no-tty` opts out per submit.

**2. A hook — any shell command.** Set it once for the whole machine:

```bash
gsched notify 'curl -s -d "$GSCHED_MSG" ntfy.sh/my-gpu-box'          # phone push
gsched notify --test                                                 # fire one now
gsched notify --clear
```

or per job (`--notify`), or per shell (`export GSCHED_NOTIFY=...` in your
`.bashrc` — every job you submit from that shell inherits it). Precedence:
job → machine-wide. Some recipes:

```bash
# Slack / Discord incoming webhook
gsched notify 'curl -s -X POST -H "Content-type: application/json" \
  -d "{\"text\": \"$GSCHED_MSG\"}" https://hooks.slack.com/services/XXX'

# email yourself
gsched notify 'echo "$GSCHED_MSG" | mail -s "gsched: job $GSCHED_ID $GSCHED_STATUS" you@x.edu'

# only bother me when something breaks, with the tail of the log
gsched submit "python train.py" --notify \
  '[ "$GSCHED_STATUS" = done ] || tail -20 "$GSCHED_LOG" | mail -s "job $GSCHED_ID failed" you@x.edu'

# chain: kick off the next stage automatically
gsched submit "python pretrain.py" --notify \
  '[ "$GSCHED_RC" = 0 ] && gsched submit "python finetune.py" --name ft'
```

The hook runs detached, as the daemon user, with the job in its environment:

| var | |
|---|---|
| `GSCHED_ID` `GSCHED_NAME` `GSCHED_CMD` | which job |
| `GSCHED_STATUS` | `done` / `failed` / `evicted` |
| `GSCHED_RC` | exit code (empty if it never ran) |
| `GSCHED_GPUS` | every GPU it got, e.g. `0,3` (`GSCHED_GPU` is just the first, `GSCHED_NGPU` the count) |
| `GSCHED_HOST` `GSCHED_ELAPSED` | where and how long |
| `GSCHED_LOG` | path to its log file |
| `GSCHED_MSG` | the whole one-line summary, pre-formatted |

Hook output (and errors) go to `~/.gsched/notify.log`. A hook that hangs, fails,
or points at a dead terminal never affects scheduling — notifications are fired
and forgotten.

### Change which GPUs are used, live

The allowed list lives in the DB and the daemon re-reads it every poll, so you
never restart it:

```bash
gsched gpus 0,1,2,3,4,5      # add 4 and 5
gsched gpus 0,1             # go back to just 0 and 1 (running jobs keep going)
```

## Specifying commands (the one thing to know)

Storage is safe — commands are stored verbatim in SQLite. Jobs run via `sh -c`
(so inline env vars, redirects, `&&`, globs all work), which means there are two
shell layers:

- **At submit time** your shell parses `submit "..."`, so quote the whole
  command. Plain commands like `python train.py +a=b +c=d` need nothing special.
- **At run time** `sh -c` re-parses it; `$VAR` expands in the **daemon's**
  environment, not your submit shell's.
- For commands containing quotes, prefer **`submitf`** (each line is taken as-is,
  no submit-time quoting). Its only caveat: the leading `name` (and optional
  `ngpu`) fields are split off on **TAB**s.

**Don't set `CUDA_VISIBLE_DEVICES` yourself** — the daemon owns it. Write a job
for `cuda:0`, or `cuda:0..N-1` if you asked for `--ngpu N`.

## How it works

Each job is a row with a `status`: **queued → running → done / failed / evicted**.
Every `--poll` seconds the daemon:

1. **reaps** finished processes (records exit code → `done`/`failed`);
2. **evicts** any job you asked to kill;
3. **dispatches**: it collects the allowed GPUs that are free (memory `< --mem`
   MiB *and* not held by one of our jobs), then walks the queue oldest-first,
   giving each job its `ngpu` GPUs until the free set runs out or the next job
   doesn't fit (see [Multi-GPU jobs](#multi-gpu-jobs)).

Any row reaching a final state in steps 1–2 is announced (tty + hook) as part of
the same poll.

Logs go to `~/.gsched/logs/job<id>.log`. The daemon can be restarted anytime — it
re-reads the DB, and jobs whose process died while it was down are marked done on
the next poll. An existing `~/.gsched/gsched.db` from an older version is upgraded
in place the first time any command opens it; old single-GPU rows keep their
history.

## Tests

```bash
pytest
```

`tests/test_gsched.py` drives the scheduling core (`_step`) with fake GPU memory
and trivial jobs (`true`/`false`/`sleep`) — the full reap/evict/dispatch logic,
multi-GPU allocation (packing, FIFO blocking, backfill, `CUDA_VISIBLE_DEVICES`)
and the notification paths (tty write-back, job vs. machine-wide hooks, hook env,
failures not breaking the loop) are verified deterministically, no real GPUs
needed.

## License

MIT
