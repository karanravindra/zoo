"""Score generated stories with the local System One Laya evaluator."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QUESTIONS = {
    "creativity": {
        "type": "score",
        "instructions": "How creative and original is the story?",
        "criteria": [
            "Entirely generic or repetitive",
            "Mostly predictable",
            "Some original ideas",
            "Fairly inventive",
            "Highly original and imaginative",
        ],
    },
    "grammar": {
        "type": "score",
        "instructions": "How grammatically correct is the story text?",
        "criteria": [
            "Pervasive errors, hard to read",
            "Frequent errors",
            "Occasional errors",
            "Rare minor errors",
            "Flawless grammar",
        ],
    },
    "consistency": {
        "type": "score",
        "instructions": "How consistent is the story with itself (characters, objects, facts, no contradictions)?",
        "criteria": [
            "Contradicts itself constantly",
            "Many inconsistencies",
            "Some inconsistencies",
            "Mostly consistent",
            "Fully consistent",
        ],
    },
    "instruct": {
        "type": "score",
        "instructions": "How well does the story follow the given instruction?",
        "criteria": [
            "Ignores the instruction",
            "Barely related",
            "Partially follows",
            "Mostly follows",
            "Follows fully",
        ],
    },
    "plot": {
        "type": "score",
        "instructions": "How coherent is the plot, with a clear beginning, development and resolution?",
        "criteria": [
            "No plot",
            "Fragmented events",
            "Weak plot with gaps",
            "Clear plot, minor gaps",
            "Coherent, well-structured plot",
        ],
    },
}


def score_generation(endpoint, instruction, story, timeout):
    payload = {
        "model": "laya",
        "state": {"instruction": instruction, "story": story},
        "questions": QUESTIONS,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Evaluator returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach evaluator at {endpoint}: {error.reason}") from error

    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("Evaluator response does not contain an answers object")
    try:
        return {name: answer["score"] for name, answer in answers.items()}
    except (KeyError, TypeError) as error:
        raise RuntimeError("Evaluator response contains an answer without a score") from error


def read_generations(path):
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                generation = json.loads(line)
                instruction = generation["instruction"]
                story = generation["story"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"{path}:{line_number} must be JSON with instruction and story fields"
                ) from error
            if not isinstance(instruction, str) or not isinstance(story, str):
                raise ValueError(f"{path}:{line_number} instruction and story must be strings")
            yield generation, instruction, story


def main():
    parser = argparse.ArgumentParser(description="Score stories with the Laya evaluator")
    parser.add_argument("--instruction", help="Instruction used to generate a single story")
    parser.add_argument("--story", help="Single generated story to score")
    parser.add_argument("--input", type=Path, help="JSONL file with instruction and story fields")
    parser.add_argument("--endpoint", default="http://localhost:11434/v1/systemone")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.input:
        if args.instruction is not None or args.story is not None:
            parser.error("--input cannot be combined with --instruction or --story")
        try:
            generations = read_generations(args.input)
            for generation, instruction, story in generations:
                print(json.dumps({**generation, "scores": score_generation(args.endpoint, instruction, story, args.timeout)}))
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    elif args.instruction is not None and args.story is not None:
        try:
            print(json.dumps(score_generation(args.endpoint, args.instruction, args.story, args.timeout)))
        except RuntimeError as error:
            parser.error(str(error))
    else:
        parser.error("provide both --instruction and --story, or provide --input")


if __name__ == "__main__":
    main()
