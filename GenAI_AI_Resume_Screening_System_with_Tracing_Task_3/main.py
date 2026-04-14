import json
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langsmith import Client  # pyright: ignore[reportMissingImports]

from chains.resume_pipeline import ResumeScreeningPipeline


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def ensure_required_environment() -> None:
    required_variables = [
        "GROQ_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
    ]
    missing_variables = [
        variable_name
        for variable_name in required_variables
        if not os.getenv(variable_name)
    ]
    if missing_variables:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing_variables)
        )


def load_resume_files(resume_directory: Path) -> Dict[str, str]:
    resume_file_paths = {
        "strong_candidate": resume_directory / "strong_candidate.txt",
        "average_candidate": resume_directory / "average_candidate.txt",
        "weak_candidate": resume_directory / "weak_candidate.txt",
    }
    return {
        candidate_label: read_text_file(file_path)
        for candidate_label, file_path in resume_file_paths.items()
    }


def run_assignment_pipeline() -> None:
    load_dotenv()
    ensure_required_environment()

    base_directory = Path(__file__).resolve().parent
    job_description_text = read_text_file(base_directory / "data" / "job_description.txt")
    resumes = load_resume_files(base_directory / "data" / "resumes")

    pipeline = ResumeScreeningPipeline(base_directory=base_directory)
    all_results: Dict[str, Dict] = {}

    for candidate_label, resume_text in resumes.items():
        result = pipeline.run(resume_text=resume_text, job_description_text=job_description_text)
        all_results[candidate_label] = result
        print(f"\n===== {candidate_label.upper()} =====")
        print(json.dumps(result, indent=2))

    output_file_path = base_directory / "output_results.json"
    output_file_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved pipeline outputs to: {output_file_path}")

    print("\nLangSmith tracing is enabled. Open LangSmith UI to inspect all steps and tags.")
    print("For debug requirement: run with a deliberately noisy resume to inspect incorrect extraction.")

    if os.getenv("LANGSMITH_API_KEY"):
        client = Client()
        print(f"Connected to LangSmith API endpoint: {client.api_url}")


if __name__ == "__main__":
    run_assignment_pipeline()
