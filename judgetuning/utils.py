import os

import numpy as np
import pickle
import random
import string
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
from timeblock import Timeblock

from judgetuning.judge_paths import judge_tuning_datadir

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download


def upload_hf(name: str, local_path: Path):
    # upload the model located in `local_path` to hugging-face
    api = HfApi()
    api.upload_folder(
        folder_path=local_path,
        repo_id="judge-arena/judge-arena-dataset",
        path_in_repo=name,
        commit_message=f"Upload {name}",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )


def download_hf(name: str, local_path: Path):
    local_path.mkdir(exist_ok=True, parents=True)
    # downloads the model from huggingface into `local_path` folder
    snapshot_download(
        repo_id="judge-arena/judge-arena-dataset",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )


def figure_path(folder: str | None, name: str, verbose: bool = True):
    # utils to get a figure path in {root}/figures/folder/name
    root = Path(__file__).parent.parent / "figures"
    figure_folder = root / folder if folder is not None else root
    figure_folder.mkdir(exist_ok=True, parents=True)
    res = figure_folder / name
    if verbose:
        print(f"Figure path: {res}")
    return res


def read_and_format(filename, **kwargs):
    with open(filename, "r") as f:
        return f.read().format(**kwargs)


def random_string(length: int):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


default_cache_path = judge_tuning_datadir / "cache"
default_cache_path.mkdir(parents=True, exist_ok=True)


def cache_function(
    fun: Callable[[], object],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
):
    f"""
    :param fun: a function whose result obtained `fun()` will be cached, the output of the function must be serializable.
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.pkl`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = default_cache_path
    cache_file = cache_path / (cache_name + ".pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.loads(f.read())
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            res = fun()
            with open(cache_file, "wb") as f:
                cache = pickle.dumps(res)
                print(
                    f"Writing cache with size {round(sys.getsizeof(cache) / 1e6, 3)} MB"
                )
                f.write(cache)
            return res


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
):
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = default_cache_path
    cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


def load_judge_evaluations():
    """
    Load judge evaluations and downloads from HugginFace dataset if the data is not available locally.
    TODO documentation how to obtain (long)
    :return: a dictionary mapping the number of instructions to judge evaluations.
     The number of instructions are [400, 1200, 3548]. For each, the dictionary maps to a dataframe containing judge
     configuration, a matrix containing judge costs and a matrix containing human-agreements.
    """
    path = judge_tuning_datadir / "judge-evals"
    if not path.exists():
        print(f"Local folder {path} not found, downloading.")
        download_hf(name="judge-evals", local_path=path.parent)
    else:
        print(f"Local folder {path} found, loading eval data from there.")

    list_n_instructions = [400, 1200, 3548]
    df_configs_per_eval = {}
    mat_cost_per_eval = {}
    mat_human_agreement_per_eval = {}
    for n_instructions in list_n_instructions:
        df_configs_per_eval[n_instructions] = pd.read_csv(
            path / f"df_configs_{n_instructions}_instr.csv.zip"
        )
        mat_cost_per_eval[n_instructions] = np.load(
            path / f"mat_cost_per_eval_{n_instructions}_instr.npy"
        )
        mat_human_agreement_per_eval[n_instructions] = np.load(
            path / f"mat_human_agreement_per_eval_{n_instructions}_instr.npy"
        )
        n_judges = len(df_configs_per_eval[n_instructions])
        assert (
            len(df_configs_per_eval[n_instructions])
            == len(mat_cost_per_eval[n_instructions])
            == len(mat_human_agreement_per_eval[n_instructions])
        )
        print(
            f"Loaded {n_judges} judge evaluations with {n_instructions} instructions."
        )
    return df_configs_per_eval, mat_cost_per_eval, mat_human_agreement_per_eval
