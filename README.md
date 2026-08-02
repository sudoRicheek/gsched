# gsched

A tiny GPU job scheduler. One small daemon per machine owns a list of GPUs,
polls every few seconds, and runs the next **queued** job on any of its GPUs
that is **free**. Submit jobs from anywhere on the box; they queue and run as
GPUs open up. That's the whole idea.

- **One file** (`gsched.py`), one dependency (`rich`).
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

**2. Submit jobs.** Write each job **as if it has one GPU** — the daemon sets
`CUDA_VISIBLE_DEVICES` for you.

```bash
gsched submit "python train.py --lr 3e-4" --name run-a     # one job
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
```

**4. Or watch.**

```bash
gsched status
```

```
● alive  daemon pid 12345  gpus 0,1,2,3  poll 7s  mem_free<2000MiB
 GPUs                    running                          queued
 gpu  mem    job         id name   gpu pid   elapsed      # id name
   0  17800  #7 run-a     7 run-a   0  9001   0:12:33      1 11 sweep-lr-0
   1  17800  #8 run-b     8 run-b   1  9002   0:12:31      2 12 sweep-lr-1
   2      1  idle
   3      1  idle
```

## Commands

| command | what it does |
|---|---|
| `daemon --gpus 0,1,2,3 [--poll 7] [--mem 2000] [--notify "<cmd>"]` | run the scheduler loop (background it) |
| `submit "<cmd>" [--name N] [--notify "<cmd>"] [--no-tty]` | queue one command |
| `submitf <file> [--name N] [--notify "<cmd>"] [--no-tty]` | queue every line of a file (`name<TAB>cmd` supported; `#`/blank lines skipped) |
| `status [--recent K]` | dashboard: daemon health, per-GPU, running, queued, last K finished |
| `gpus 0,1,2,3,4,5` | change the allowed GPU list **live** (no restart) |
| `notify ["<cmd>"] [--clear] [--test]` | show/set/clear the machine-wide finish hook |
| `evict <id>` | cancel a queued job, or `SIGTERM` a running one (its GPU frees for the next) |
| `logs <id> [-n K]` | tail that job's stdout/stderr |
| `rm --done` / `rm <id>` | forget finished rows |

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
| `GSCHED_GPU` `GSCHED_HOST` `GSCHED_ELAPSED` | where and how long |
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
  no submit-time quoting). Its only caveat: the optional name is split on the
  first **TAB**.

**Don't set `CUDA_VISIBLE_DEVICES` yourself** — the daemon owns it. Write jobs as
single-GPU (`cuda:0`).

## How it works

Each job is a row with a `status`: **queued → running → done / failed / evicted**.
Every `--poll` seconds the daemon:

1. **reaps** finished processes (records exit code → `done`/`failed`);
2. **evicts** any job you asked to kill;
3. **dispatches** the oldest `queued` job onto each allowed GPU that is free
   (memory `< --mem` MiB *and* not already running one of our jobs), FIFO.

Any row reaching a final state in steps 1–2 is announced (tty + hook) as part of
the same poll.

Logs go to `~/.gsched/logs/job<id>.log`. The daemon can be restarted anytime — it
re-reads the DB, and jobs whose process died while it was down are marked done on
the next poll.

## Tests

```bash
pytest
```

`tests/test_gsched.py` drives the scheduling core (`_step`) with fake GPU memory
and trivial jobs (`true`/`false`/`sleep`) — the full reap/evict/dispatch logic and
the notification paths (tty write-back, job vs. machine-wide hooks, hook env,
failures not breaking the loop) are verified deterministically, no real GPUs
needed.

## License

MIT
