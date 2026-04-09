import os
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from judgetuning.annotation_dataset import AnnotationDataset
from judgetuning.utils import download_hf
from judgetuning.chatbot_arena_utils import load_chatbot_arena_elo_ratings
from judgetuning.annotation_dataset.tables import default_table_path
from judgetuning.chatbot_arena_utils import compute_mle_elo

table_root_default = default_table_path()


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


def load_model_outputs(
    name: str, table_root: Path = table_root_default
) -> pd.DataFrame:
    return read_df(
        table_root / "model_outputs" / name,
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
    name: str, table_root: Path = Path(__file__).parent
) -> pd.DataFrame:
    annotation_path = table_root / "judge_annotations" / name
    return read_df(annotation_path)


class TableAnnotationDataset(AnnotationDataset):
    def __init__(
        self,
        name: str,
        table_root: Path = table_root_default,
        instruction_filenames: list[str] | None = None,
        model_filenames: list[str] | None = None,
        judge_filenames: list[str] | None = None,
    ):
        super().__init__(name)
        self.table_root = table_root

        self.instruction_filenames = instruction_filenames
        self.model_filenames = model_filenames
        self.judge_filenames = judge_filenames

        # if files not present locally, download them
        if not self._local_files_present(
            instruction_filenames, model_filenames, judge_filenames
        ):
            print(f"Local files for {name} not found, downloading them.")
            download_hf(name=name, local_path=table_root)
        self._df_instructions = pd.concat(
            [
                load_table_instructions(name=name, table_root=table_root)
                for name in instruction_filenames
            ],
            ignore_index=False,
        )
        # print(f"Loaded {len(self.instructions)} instruction")
        self._df_model = pd.concat(
            [
                load_model_outputs(name=name, table_root=table_root)
                for name in model_filenames
            ],
            ignore_index=False,
        )
        # pivot it to make access to whole set of models easier
        self._df_model = self._df_model.reset_index().pivot_table(
            index="model", columns="instruction_index", values="output", aggfunc="last"
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

    def __str__(self) -> str:
        n_models = len(self.models)
        n_instructions = len(self.instructions)
        res = f"{self.name} dataset with {n_models} models and {n_instructions} instructions."
        if self._df_judge is not None:
            elo = self.elo_ratings().sort_values(ascending=False)
            top = elo.index[0]
            worse = elo.index[-1]
            elo_string = f" Top model {top}, worse model {worse}."
            res += elo_string
        return res

    def subselect_models(self, models: list[str]):
        """
        Only keep the models given as arguments, model should already be present or an error is thrown.
        :param models:
        :return:
        """
        for model in models:
            assert model in self.models, f"Model {model} was not in {self.models}."
        self._df_model = self._df_model.loc[models]
        if self._df_judge is not None:
            self._df_judge = self._df_judge.loc[self._df_judge.model2.isin(models), :]
        return self

    def rename_models(self, model_renaming: dict[str, str], drop_missing: bool = True):
        """
        Rename models given the dictionary if `drop_missing` is True models that dont appear in the dictionary
        are discarded.
        :param models:
        :return:
        """
        if drop_missing:
            models = [x for x in self.models if x in model_renaming]
            self.subselect_models(models)
        self._df_model.rename(index=model_renaming, inplace=True)
        if self._df_judge is not None:
            self._df_judge.loc[:, "model2"] = self._df_judge.loc[:, "model2"].apply(
                lambda s: model_renaming[s]
            )

    @property
    def models(self) -> list[str]:
        return list(sorted(self._df_model.index.tolist()))

    def model_output(self, model: str) -> pd.Series:
        return self._df_model.loc[model]

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

    def df_winrate_against_baseline(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        :param models:
        :param judge_name:
        :param instructions:
        :return: a table whose index/columns/values are the instructions/models and preference values in [0, 1]
        against the baseline (higher is better for the model)
        """
        return self._sub_df_judge(
            models=models, judge_name=judge_name, instructions=instructions
        ).pivot_table(
            index="instruction_index",
            columns="model2",
            values="preference",
        )

    def _sub_df_judge(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.DataFrame:
        # initialize mask to True
        mask = ~(self._df_judge.loc[:, self._df_judge.columns[0]] == np.nan)
        if judge_name is not None:
            assert judge_name in self.judges
            mask = self._df_judge.judge_index == judge_name
        if models is not None:
            mask &= self._df_judge.model2.isin(models)
        if instructions is not None:
            mask &= self._df_judge.instruction_index.isin(instructions)
        return self._df_judge.loc[mask]

    def chatbot_arena_elo_ratings(
        self,
        models: list[str] | None = None,
    ) -> pd.Series:
        if models is None:
            models = self.models
        return load_chatbot_arena_elo_ratings().loc[models]

    def elo_ratings(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> pd.Series:
        def winner_from_pref(preference: float) -> str:
            if preference < 0.5:
                return "model_a"
            elif preference > 0.5:
                return "model_b"
            else:
                return "tie"

        sub_df = self._sub_df_judge(
            models=models, instructions=instructions, judge_name=judge_name
        )
        df_battles = sub_df.loc[
            :, ["instruction_index", "model1", "model2", "preference"]
        ]
        df_battles = df_battles.rename(
            columns={"model1": "model_a", "model2": "model_b"}
        )
        df_battles["winner"] = df_battles.apply(
            lambda x: winner_from_pref(x["preference"]), axis=1
        )
        return compute_mle_elo(df_battles)

    def cost_per_annotation(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(
            self._sub_df_judge(
                models=models, judge_name=judge_name, instructions=instructions
            ).cost.mean()
        )

    def time_per_annotation(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> float:
        return float(
            self._sub_df_judge(
                models=models, judge_name=judge_name, instructions=instructions
            ).time.mean()
        )

    def num_annotations(
        self,
        models: list[str] | None = None,
        judge_name: str | None = None,
        instructions: list[str] | None = None,
    ) -> int:
        return len(
            self._sub_df_judge(
                models=models, judge_name=judge_name, instructions=instructions
            )
        )

    def judge_annotation(
        self,
        judge_name: str,
        models: list[str] | None = None,
        instructions: list[str] | None = None,
    ):
        pass

    def push_to_hf(self):
        files = (
            [
                Path("instructions") / instruction_filename
                for instruction_filename in self.instruction_filenames
            ]
            + [
                Path("model_outputs") / model_filename
                for model_filename in self.model_filenames
            ]
            + [
                Path("judge_annotations") / judge_filename
                for judge_filename in self.judge_filenames
            ]
        )
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

    def _local_files_present(
        self, instruction_filenames, model_filenames, judge_filenames
    ) -> bool:
        files = (
            [
                self.table_root / "instructions" / instruction_filename
                for instruction_filename in instruction_filenames
            ]
            + [
                self.table_root / "model_outputs" / model_filename
                for model_filename in model_filenames
            ]
            + [
                self.table_root / "judge_annotations" / judge_filename
                for judge_filename in judge_filenames
            ]
        )
        for file in files:
            if not Path(file).exists():
                return False
        return True
