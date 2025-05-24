# Judge Tuning

This repository allows to tune judge configurations and reproduce the results of [this ICML 2025 paper](https://arxiv.org/abs/2501.17178).

## How to install

First download the repository:
```bash
git clone https://github.com/geoalgo/judgetuning
cd judgetuning
```

Then install with uv (recommended):
```
uv sync .
uv add -r requirements.txt
source .venv/bin/activate
PYTHONPATH=. python results_scripts/figure1.py 
```

or with pip:
```
pip install -r requirements.txt
PYTHONPATH=. python results_scripts/figure1.py 
```

For JudgeLM evaluations, you will need to do those additional steps:

```
cd ..
git clone https://github.com/baaivision/JudgeLM
cd JudgeLM
pip3 install --upgrade pip 
pip3 install -e .
pip install flash-attn==2.0.4 --no-build-isolation
pip install pydantic==2.10.0 #correcting the version again
```

## Evaluating a judge

### Evaluating your own judge

To evaluate your own judge, you can define the `JudgeCustom` object here [class](judgetuning/judge/judge_custom.py). This will require you to define:
- `preference`: the function you use for judging which output is preferred.
- `swap`: flag for indicating the positions of the outputs were swapped for future analysis of position bias.
- `judge_completion`: the text completion from an LLM judge used for evaluating `preference` (this one is optional)

### Reevaluating a given judge configurations

If you want to reevaluate one of the 4480 judge configuration, you can run

```
PYTHONPATH=. python judgetuning/script/evaluate_human_agreement.py --expid test --judge_class judge-length --dataset pandalm --split test
PYTHONPATH=. python judgetuning/script/evaluate_human_agreement.py --expid test --judge_class judge-option --dataset lmsys \
--split val --provide_confidence 1 --provide_example 0 --json_output 1 --temperature 0.001 --score_type likert 
```

See `parse_args` in `evaluate_spearman_correlation.py` for 
other supported options, for instance dataset can be "lmsys", "pandalm", "llmbar" for `evaluate_human_agreement.py`, 
you can configure the judge class to be our tunable class, arena-hard, alpaca-eval, pandalm, judgelm, etc.

To evaluate Spearman correlation on chatbot arena, you can run:

```
# Evaluate the length judge baseline on the test split of Alpaca-Eval dataset
PYTHONPATH=. python judgetuning/script/evaluate_spearman_correlation.py  --expid test --judge_class judge-length --dataset alpaca-eval --split test 
```
As above, you can customize the judge and other options, see `parse_args` to get the list of supported options.

### Evaluating your own judge

TODO Omar.

### Evaluating baselines

TODO Omar.

## Computing Multiobjective Successive-Halving

TODO David.

## Generating figures & results

See [results_scripts](results_scripts/), for instance:
```bash
python results_scripts/figure1.py
python results_scripts/table2.py
```

where you can replace by the name of figure/table needed.


## Citation

If you use this work in your research, please cite the following paper:

```bash
@misc{salinas2025tuningllmjudgedesign,
      title={Tuning LLM Judge Design Decisions for 1/1000 of the Cost}, 
      author={David Salinas and Omar Swelam and Frank Hutter},
      year={2025},
      eprint={2501.17178},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.17178}, 
}
```

TODO update with ICML bibtex.
