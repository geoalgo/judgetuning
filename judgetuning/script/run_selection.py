import pickle
from pathlib import Path

from judgetuning.script.multiobjective import argsort_nondominated
import numpy as np
import pandas as pd



def load_pkl(cache_file):
    with open(cache_file, "rb") as f:
        return pickle.loads(f.read())


cols = [
    "model",
    "provide_confidence",
    "provide_example",
    "provide_answer",
    "provide_explanation",
    "json_output",
    "n_sample_with_shift",
    "score_type",
    "temperature",
]


def compare(df_configsA, df_configsB):

    config_dicts_A = set(
        [tuple(x.values()) for x in df_configsA.loc[:, cols].to_dict(orient="records")]
    )
    config_dicts_B = set(
        [tuple(x.values()) for x in df_configsB.loc[:, cols].to_dict(orient="records")]
    )
    print(f"Number of configs in A: {len(config_dicts_A)}")
    print(f"Number of configs in B: {len(config_dicts_B)}")
    print(
        f"Number of configs missing: {len(config_dicts_A.difference(config_dicts_B))}"
    )

def select_top(df_configs, mat_human_agreement, mat_cost, n_configs_to_select: int):
    df_configs.reset_index(inplace=True)
    n_instructions_to_load = mat_cost.shape[-1]

    cols = [
        "model",
        "provide_confidence",
        "provide_example",
        "provide_answer",
        "provide_explanation",
        "json_output",
        "n_sample_with_shift",
        "score_type",
        "temperature",
    ]
    config_dicts = list(df_configs.loc[:, cols].to_dict(orient="records"))
    val_scores = mat_human_agreement[:, :n_instructions_to_load].mean(axis=-1)
    cost_scores = mat_cost[:, :n_instructions_to_load].mean(axis=-1)

    min_score = 0.425
    valid_indices = np.array(
        [i for i in range(len(df_configs)) if val_scores[i].mean() > min_score]
    )

    top_configs_alive_indices = argsort_nondominated(
        X=np.stack([cost_scores[valid_indices], val_scores[valid_indices]]).T,
        max_items=n_configs_to_select,
        minimize=[True, False],
        dim=1,
    )
    top_configs_alive_indices = [valid_indices[i] for i in top_configs_alive_indices]
    return pd.DataFrame([config_dicts[i] for i in top_configs_alive_indices])

def model_rename(s: str):
    if "/" in s:
        s = s.split("/")[1].lower()
        for token in ["-instruct", "-gptq-int8", "-it", "-fp8", "meta-"]:
            s = s.replace(token, "")
        return s
    else:
        return s
def model_size(s: str):
    return int(s.split("-")[-1].replace("b", ""))


def main():
    """
    We show how to run the multiobjective selection for successive halving.
    :return:
    """
    cache_root = Path("/Users/salinasd/judge-tuning-data/cache/")

    df_configs1, mat_cost1, mat_human_agreement1 = load_pkl(
        cache_root / "multiobj-data-fidelity-v2-400.pkl"
    )

    # compute top configurations
    df_configs_alive_2 = select_top(
        df_configs1,
        mat_human_agreement1,
        mat_cost1,
1200
    )

    # compare with configurations that were evaluated,
    # 4 configurations are different, this is because the ND sort was used without fixing the seed which introduces a
    # small variation (the first point was picked at random since `dim` was not set)
    # we changed the default to pick first the configuration optimizing the human-agreement (dim=1)
    print("Compare available configs and recomputed selection (step1)")
    df_configs2, mat_cost2, mat_human_agreement2 = load_pkl(
        cache_root / "multiobj-data-fidelity-v2-val-1200-fix-400-1200.pkl"
    )
    compare(
        df_configs2,
        df_configs_alive_2,
    )

    df_configs_alive_3 = select_top(
        df_configs2,
        mat_human_agreement2,
        mat_cost2,
        400
    )
    tag = "fidelity-v2-val-3600-fix"
    print(f"Compare available configs and recomputed selection {tag} (step 2)")
    df_configs3, mat_cost3, mat_human_agreement3 = load_pkl(
        cache_root / f"multiobj-data-{tag}-3548.pkl"
    )

    # compare with configurations that were evaluated,
    # 17 configurations are missing due to hardware error, we will consider rerunning if someone express the need
    compare(
        df_configs3,
        df_configs_alive_3,
    )

    # compute the top configuration for each size from the configurations evaluated with 3548 instructions
    df_configs3["model_size"] = df_configs3["model"].apply(model_rename).apply(model_size)
    df_configs3["cost"] = mat_cost3.mean(axis=1)

    n_params = {
        "Ours-small": 10,
        "Ours-medium": 32,
        "Ours-large": 72
    }
    top_config_by_size = {}
    for name, n in n_params.items():
        top_config_by_size[n] = df_configs3[df_configs3.model_size <= n].sort_values(
            by="human_agreement", ascending=False
        ).loc[:, ["human_agreement", "cost"] + cols].reset_index(drop=True).loc[0].to_dict()
        top_config_by_size[n]["name"] = name

    dd = pd.DataFrame(top_config_by_size).T
    print("\nTop judges per model sizes, small, medium and large:")
    print(dd.to_string(index=False))
    dd.to_csv(Path(__file__).parent / "top_judge.csv", index=False)


if __name__ == '__main__':
    main()