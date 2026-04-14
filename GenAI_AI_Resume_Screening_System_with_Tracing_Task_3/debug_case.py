import json
from pathlib import Path

from dotenv import load_dotenv

from chains.resume_pipeline import ResumeScreeningPipeline


def main() -> None:
    load_dotenv()
    base_directory = Path(__file__).resolve().parent

    pipeline = ResumeScreeningPipeline(base_directory=base_directory)
    job_description_text = (base_directory / "data" / "job_description.txt").read_text(encoding="utf-8")

    noisy_resume = """
    Name: Test Candidate
    Experience: Worked on many projects. Familiar with several technologies.
    Skills: Problem solving, team player.
    """

    result = pipeline.run(resume_text=noisy_resume, job_description_text=job_description_text)
    print(json.dumps(result, indent=2))
    print(
        "\nInspect this run in LangSmith to show at least one incorrect extraction/match "
        "and describe your prompt improvement."
    )


if __name__ == "__main__":
    main()
