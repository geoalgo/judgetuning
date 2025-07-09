from pathlib import Path
import pandas as pd
from judgetuning.judge import AnnotationRequest
from judgetuning.script.utils import make_judge, EvaluateFidelityArgs

# This script demonstrates how to use one of the tuned judge model to annotate a given instruction.
# It starts by loading the configuration for a specific judge, in this case, "Ours-small".
# The configuration is used to set up the evaluation arguments, which are then passed to the judge.
# Finally, the script defines an instruction and two possible outputs, and the judge annotates them.

# The name of the judge model to use for the annotation.
# It can be one of "Ours-small", "Ours-medium", or "Ours-large".
name = "Ours-small"

assert name in ["Ours-small", "Ours-medium", "Ours-large"]

# Load the configuration for the selected judge model from the 'top_judge.csv' file.
# The configuration includes hyperparameters such as the model name, temperature, and other settings.
df_config = pd.read_csv(Path(__file__).parent / "judgetuning/script/top_judge.csv")
hp_dict = df_config[df_config.name == name].loc[0].to_dict()

print(hp_dict)

# Create an instance of the evaluation arguments, passing the loaded hyperparameters.
# These arguments control the behavior of the judge, such as whether to provide confidence scores or explanations.
args = EvaluateFidelityArgs(
    model=hp_dict["model"],
    provide_confidence=hp_dict["provide_confidence"],
    provide_explanation=hp_dict["provide_explanation"],
    provide_example=hp_dict["provide_example"],
    provide_answer=hp_dict["provide_answer"],
    json_output=hp_dict["json_output"],
    score_type=hp_dict["score_type"],
    temperature=hp_dict["temperature"],
    n_sample_with_shift=hp_dict["n_sample_with_shift"],
)

# Create the judge instance using the specified evaluation arguments.
judge = make_judge(args)

# Define the instruction and two possible outputs to be annotated by the judge.
instruction = "Give me two countries that starts with S."
output_good = "Spain, Sweden."
output_dummy = "No way I would answer to this question!"

# Annotate the request using the judge.
# The judge will evaluate the two outputs based on the given instruction and provide an annotation.
annotations = judge.annotate(
    requests=[
        AnnotationRequest(
            instruction=instruction,
            instruction_index="feaoh",
            output1=output_dummy,
            output2=output_good,
            model1="dummy",
            model2="foo",
        )
    ]
)[0]
for annotation in annotations:
    print(annotation)


