#!/usr/bin/env fish

# Run the same model and evaluation contract with several samplers. Cells are
# sequential on each machine and can be sharded deterministically by rank.

function _usage
    echo "Usage:"
    echo "  fish scripts/eval/vbvr_pro/sweep.fish --output-base PATH [sweep options] -- [run.fish options]"
    echo
    echo "Sweep options:"
    echo "  --output-base PATH    Parent directory for sampler cells (required)"
    echo "  --samplers LIST       Comma-separated unipc, euler, or cps:FLOAT entries"
    echo "                        (default: unipc,euler,cps:0.3,cps:0.7)"
    echo "  --world-size N        Number of machines sharing the cells"
    echo "  --rank N              Zero-based machine rank"
    echo "  --assignment-only     Print this rank's cells without running them"
    echo "  -h, --help            Show this help"
    echo
    echo "Everything after -- is forwarded to run.fish. Do not pass --output-root,"
    echo "--sampler, or --cps-noise there; the sweep owns those options."
end

function _fail
    echo "[error] $argv" >&2
    exit 2
end

argparse -n vbvr-pro-sweep \
    'h/help' \
    'output-base=' \
    'samplers=' \
    'world-size=' \
    'rank=' \
    'assignment-only' \
    -- $argv
or exit 2

if set -q _flag_help
    _usage
    exit 0
end
set -q _flag_output_base; or _fail "--output-base is required"

for arg in $argv
    switch $arg
        case --output-root '--output-root=*' --sampler '--sampler=*' --cps-noise '--cps-noise=*'
            _fail "$arg is controlled by sweep.fish and cannot be forwarded"
    end
end

set -l sampler_spec unipc,euler,cps:0.3,cps:0.7
set -q _flag_samplers; and set sampler_spec $_flag_samplers
set -l sampler_tokens
for raw in (string split , -- $sampler_spec)
    set -l token (string lower -- (string trim -- $raw))
    test -n "$token"; or _fail "--samplers contains an empty entry"
    if contains -- $token unipc euler
        # Valid as-is.
    else if string match -qr '^cps:' -- $token
        set -l level (string replace 'cps:' '' -- $token)
        string match -qr '^(0(\.[0-9]+)?|1(\.0+)?)$' -- $level
        or _fail "invalid Flow-CPS entry '$token'; expected cps:FLOAT with FLOAT in [0, 1]"
    else
        _fail "unknown sampler entry '$token'; use unipc, euler, or cps:FLOAT"
    end
    contains -- $token $sampler_tokens; and _fail "duplicate sampler entry: $token"
    set -a sampler_tokens $token
end
test (count $sampler_tokens) -gt 0; or _fail "--samplers selected no cells"

set -l eval_world 1
set -l eval_rank 0
if set -q WORLD_SIZE
    set eval_world $WORLD_SIZE
end
if set -q RANK
    set eval_rank $RANK
end
set -q _flag_world_size; and set eval_world $_flag_world_size
set -q _flag_rank; and set eval_rank $_flag_rank
string match -qr '^[1-9][0-9]*$' -- "$eval_world"; or _fail "--world-size must be a positive integer"
string match -qr '^[0-9]+$' -- "$eval_rank"; or _fail "--rank must be a nonnegative integer"
test $eval_rank -lt $eval_world; or _fail "rank $eval_rank is outside world size $eval_world"

set -l script_dir (realpath (dirname (status filename)))
source $script_dir/../../lib/env.fish; or exit 1

set -l assigned 0
for index in (seq (count $sampler_tokens))
    if test (math "($index - 1) % $eval_world") -ne $eval_rank
        continue
    end

    set assigned (math $assigned + 1)
    set -l token $sampler_tokens[$index]
    set -l sampler $token
    set -l sampler_args
    set -l label $token
    if string match -qr '^cps:' -- $token
        set sampler cps
        set -l level (string replace 'cps:' '' -- $token)
        set sampler_args --cps-noise $level
        set label cps-noise-$level
    end
    set -l output_root (string trim -r -c / -- $_flag_output_base)/$label
    echo "[assignment] rank=$eval_rank/$eval_world sampler=$token output=$output_root"
    if set -q _flag_assignment_only
        continue
    end

    fish $script_dir/run.fish \
        --output-root $output_root \
        --sampler $sampler \
        $sampler_args \
        $argv
    or _fail "evaluation failed for sampler cell: $token"
end

if test $assigned -eq 0
    echo "[done] rank $eval_rank has no assigned sampler cells"
else if set -q _flag_assignment_only
    echo "[done] assignment-only mode; assigned cells=$assigned"
else
    echo "[done] completed sampler cells=$assigned"
end
