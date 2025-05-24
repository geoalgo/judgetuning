import json
from typing import Callable

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest


class JudgeCustom(Judge):
    """This class gives an example of a custom judge implementation to help you in case you want to benchmark your own judge."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return "JudgeCustom"

    def _annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        """
        This is the main function used for evaluating the judgements. Generating a collection of `JudgeAnnotation` 
        which contains the preference of the judge based on the pairwise outputs together with additional metadata.

        Args:
        requests -> list[AnnotationRequest]: The input is in the format of a list of AnnotationRequest objects
                                              that contain the pairwise LLM outputs to be evaluated, each element in the list is a different battle.

        Output:
        annotations -> list[list[JudgeAnnotation]]: A list of lists of JudgeAnnotation, which is a wrapper for your
                                                    custom judge evaluations. The inner list contains multiple annotations
                                                    (from different seeds, swapped positions of requests, etc.), while the 
                                                    outer list size should correspond to the number of input requests.
        
        
        """

        # An example implementation here for a judge that prefers the longest answer.
        # replace the code below for your custom judge implementation. 
        
        return [
            [
                JudgeAnnotation(
                    preference=1 - float(len(request.output1) > len(request.output2)), # indicates the function used for evaluating preference
                    swap=False,                                                        # used for swapped positions of requests (for position bias)
                    judge_completion="NA",                                             # textual output if you are using an LLM judge
                    **{k: v for k, v in request.__dict__.items() if k != "instruction"} # adding meta information from request as model names and different LLMs outputs
                )
            ]
            for request in requests
        ]

    def to_json(self):
        json_repr = {
            "judge_cls": "JudgeCustom",
            "judge_kwargs": {},
        }
        return json.dumps(json_repr)

    @classmethod
    def from_json(cls, json_str: str):
        return cls()

if __name__ == "__main__":
    
    judge = JudgeCustom()
    
    request = AnnotationRequest(
        instruction_index=0,
        instruction="Write a short story about a cat",
        output1="The cat was a good cat",
        output2="The cat was a bad cat",
    )
    print(judge.annotate([request]))
