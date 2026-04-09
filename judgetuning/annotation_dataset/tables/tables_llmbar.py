import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from judgetuning.annotation_dataset.tables.table_annotation_dataset import (
    table_root_default,
)
from judgetuning.utils import random_string
import os
from huggingface_hub import HfApi

from judgetuning.annotation_dataset import AnnotationDataset

dataset_name = "llmbar"


def generate_instructions(df, table_root, name):
    # Note: updates df with the concatenated "instruction" and "input" fields
    def create_index(instruction: str) -> str:
        x = (
            instruction[:20].replace(" ", "-").lower()
            + "...-"
            + random_string(length=5)
        )
        return x.encode("utf-8", errors="replace").decode("utf-8")

    df_instruction = df.copy()
    # this dataset has an "instruction" and an "input" column. The first is to be applied to the latter
    # when present.
    df_instruction["instruction"] = df_instruction["input"]

    df_instruction["domain"] = name
    # df_instruction['instruction'] = df_instruction['instruction'].map(
    #     lambda x: x.encode('unicode-escape').decode('utf-8'))
    df_instruction.drop_duplicates(subset="instruction", inplace=True)
    df_instruction["instruction_index"] = df_instruction["instruction"].apply(
        create_index
    )
    df_instruction.to_csv(
        table_root / "instructions" / f"{name}.csv", index=False, errors="replace"
    )
    instructions, instructions_index = df_instruction.loc[
        :, ["instruction", "instruction_index"]
    ].values.T
    return dict(zip(instructions, instructions_index))


def generate_annotations(df, table_root, name, instruction_index_dict):
    df_judge_rows = []
    for i, row in df.iterrows():
        common_kwargs = {
            "model1": "model1",
            "model2": "model2",
            "output1": str(row["output_1"]),
            "output2": str(row["output_2"]),
            "instruction_index": instruction_index_dict[row["input"]],
            "prompt": "NA",
            "swap": False,
            "number_output_tokens": 0,
            # TODO we could put avg salary?
            "cost": 0,
            "time": 0,
        }
        # can be 0, 1, 2 which corr. to tie, win model 1 and win model 2
        raw_pref = row["label"]
        match raw_pref:
            case 0:
                preference = 0.5
            case 1:
                preference = 0.0
            case 2:
                preference = 1.0
            case _:
                raise ValueError(f"Invalid preference {raw_pref}")

        df_judge_rows.append(
            {
                "preference": preference,
                "judge_index": "annotator",
                **common_kwargs,
            }
        )

    df_judge = pd.DataFrame(df_judge_rows)
    df_judge.to_parquet(
        table_root / "judge_annotations" / f"{name}.parquet", index=False
    )


def generate_dataset():
    random.seed(0)
    np.random.seed(0)

    table_root = table_root_default
    llmbar_root = table_root / dataset_name
    llmbar_root.mkdir(exist_ok=True)
    datasets = [
        "Natural",
        "Adversarial_GPTInst",
        "Adversarial_GPTOut",
        "Adversarial_Manual",
        "Adversarial_Neighbor",
    ]
    # download pandalm file if does not exists
    dataframes = []
    for dataset in datasets:
        local_file = llmbar_root / f"{dataset}.json"
        if not local_file.exists():
            print(f"Downloading {local_file} as does not exists")
            ds_origins = dataset.split("_")
            if len(ds_origins) == 1:
                url = f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/refs/heads/main/Dataset/LLMBar/{ds_origins[0]}/dataset.json"
            else:
                url = f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/refs/heads/main/Dataset/LLMBar/{ds_origins[0]}/{ds_origins[1]}/dataset.json"
            response = requests.get(url)
            with open(local_file, "wb") as file:
                file.write(response.content)
        else:
            print(
                f"Reusing existing dataset at {local_file}, remove the file to download again."
            )

        # goes through the json and convert to annotation format
        # this requires generating 1) instruction 2) model 3) annotation files
        df = pd.read_json(local_file)
        df["origin"] = dataset
        dataframes.append(df)

    df = pd.concat(dataframes)

    # 1) save instructions
    instruction_index_dict = generate_instructions(df, table_root, dataset_name)
    # 2) save human annotations
    generate_annotations(df, table_root, dataset_name, instruction_index_dict)
    print(f"done generating {dataset_name}")


def read_df(filename: str, **kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **kwargs)


def load_table_instructions(
    name: str, table_root: Path = table_root_default
) -> pd.DataFrame:
    return read_df(
        table_root / "instructions" / name,
        index_col="instruction_index",
    )


class JudgeCols:
    judge_index: str
    instruction_index: str
    model1: str
    model2: str
    prompt: str
    completion: str
    number_output_tokens: int
    preference: float
    swap: bool
    cost: float
    time: float


def load_judge_annotations(
    name: str, table_root: Path = table_root_default
) -> pd.DataFrame:
    annotation_path = table_root / "judge_annotations" / name
    return read_df(annotation_path)


class LLMBarDataset(AnnotationDataset):
    def __init__(
        self,
        name: str = "llmbar",
        table_root: Path = table_root_default,
        instruction_filenames=["llmbar.csv"],
        judge_filenames=["llmbar.parquet"],
    ):
        super().__init__(name)
        self.table_root = table_root

        self.instruction_filenames = instruction_filenames
        self.judge_filenames = judge_filenames

        self._df_instructions = pd.concat(
            [
                load_table_instructions(name=name, table_root=table_root)
                for name in instruction_filenames
            ],
            ignore_index=False,
        )
        # print(f"Loaded {len(self.models)} models")
        if judge_filenames is None or len(judge_filenames) == 0:
            self._df_judge = None
        else:
            self._df_judge = pd.concat(
                [
                    load_judge_annotations(name, table_root=table_root)
                    for name in judge_filenames
                ],
                ignore_index=True,
            )
        print(f"Loaded {name} dataset with {len(self._df_instructions)} instructions")

    @property
    def instructions(self) -> list[str]:
        return self._df_instructions.loc[:, "instruction"].tolist()

    @property
    def instruction_index(self) -> list[str]:
        return self._df_instructions.index.tolist()

    def get_instruction_index(self, instruction: str) -> str:
        return self._df_instructions.loc[
            self._df_instructions.loc[:, "instruction"] == instruction, :
        ].index.values[0]

    def get_instruction(self, instruction_index: str) -> str:
        return self._df_instructions.loc[instruction_index, "instruction"]

    @property
    def judges(self) -> list[str]:
        if self._df_judge is not None:
            return self._df_judge.judge_index.unique().tolist()
        else:
            return []

    def _sub_df_judge(
        self,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.DataFrame:
        # initialize mask to True
        mask = ~(self._df_judge.loc[:, self._df_judge.columns[0]] == np.nan)
        if judge_name is not None:
            assert judge_name in self.judges
            mask = self._df_judge.judge_index == judge_name
        if instructions is not None:
            mask &= self._df_judge.instruction_index.isin(instructions)
        return self._df_judge.loc[mask]

    def cost_per_annotation(
        self,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(
            self._sub_df_judge(
                judge_name=judge_name, instructions=instructions
            ).cost.mean()
        )

    def time_per_annotation(
        self,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(
            self._sub_df_judge(
                judge_name=judge_name, instructions=instructions
            ).time.mean()
        )

    def num_annotations(
        self,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> int:
        return len(self._sub_df_judge(judge_name=judge_name, instructions=instructions))

    def judge_annotation(
        self,
        judge_name: str,
        instructions: list[str] | None = None,
    ):
        pass

    def push_to_hf(self):
        files = [
            Path("instructions") / instruction_filename
            for instruction_filename in self.instruction_filenames
        ] + [
            Path("judge_annotations") / judge_filename
            for judge_filename in self.judge_filenames
        ]
        api = HfApi()
        for file in files:
            api.upload_file(
                path_or_fileobj=self.table_root / file,
                path_in_repo=str(file),
                repo_id="judge-arena/judge-arena-dataset",
                commit_message=f"Upload {str(file)}",
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
            )

    def _local_files_present(self, instruction_filenames, judge_filenames) -> bool:
        files = [
            self.table_root / "instructions" / instruction_filename
            for instruction_filename in instruction_filenames
        ] + [
            self.table_root / "judge_annotations" / judge_filename
            for judge_filename in judge_filenames
        ]
        for file in files:
            if not Path(file).exists():
                return False
        return True

    @staticmethod
    def generate():
        generate_dataset()


if __name__ == "__main__":
    generate_dataset()
    ds = LLMBarDataset(name=dataset_name)
    ds.push_to_hf()
