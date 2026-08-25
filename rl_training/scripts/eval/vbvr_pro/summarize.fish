#!/usr/bin/env fish

source (dirname (status filename))/../../lib/env.fish

.venv/bin/python -m src.cli.summarize_vbvr_pro_results $argv
