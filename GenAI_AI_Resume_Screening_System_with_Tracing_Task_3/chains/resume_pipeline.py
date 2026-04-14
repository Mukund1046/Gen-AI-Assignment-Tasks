import json
from pathlib import Path
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


def _load_prompt_text(prompt_file_path: Path) -> str:
    return prompt_file_path.read_text(encoding="utf-8")


class ResumeScreeningPipeline:
    def __init__(
        self,
        base_directory: Path,
        model_name: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.base_directory = base_directory
        self.prompt_directory = self.base_directory / "prompts"
        self.model = ChatGroq(model=model_name, temperature=0)
        self.json_parser = JsonOutputParser()

        self.skill_extraction_chain = self._build_chain(
            prompt_file_name="skill_extraction.txt",
            input_variables=["resume_text"],
        )
        self.matching_chain = self._build_chain(
            prompt_file_name="matching.txt",
            input_variables=["extracted_resume_json", "job_description_text"],
        )
        self.scoring_chain = self._build_chain(
            prompt_file_name="scoring.txt",
            input_variables=["matching_json"],
        )
        self.explanation_chain = self._build_chain(
            prompt_file_name="explanation.txt",
            input_variables=["extracted_resume_json", "matching_json", "scoring_json"],
        )

    def _build_chain(self, prompt_file_name: str, input_variables: list[str]):
        prompt_text = _load_prompt_text(self.prompt_directory / prompt_file_name)
        prompt_template = PromptTemplate(
            template=prompt_text,
            input_variables=input_variables,
        )
        return prompt_template | self.model | self.json_parser

    def run(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        extracted_resume = self.skill_extraction_chain.invoke(
            {"resume_text": resume_text},
            config={"tags": ["step:extract", "resume-screening"]},
        )

        matching = self.matching_chain.invoke(
            {
                "extracted_resume_json": json.dumps(extracted_resume, ensure_ascii=True),
                "job_description_text": job_description_text,
            },
            config={"tags": ["step:match", "resume-screening"]},
        )

        scoring = self.scoring_chain.invoke(
            {"matching_json": json.dumps(matching, ensure_ascii=True)},
            config={"tags": ["step:score", "resume-screening"]},
        )

        explanation = self.explanation_chain.invoke(
            {
                "extracted_resume_json": json.dumps(extracted_resume, ensure_ascii=True),
                "matching_json": json.dumps(matching, ensure_ascii=True),
                "scoring_json": json.dumps(scoring, ensure_ascii=True),
            },
            config={"tags": ["step:explain", "resume-screening"]},
        )

        return {
            "extracted_resume": extracted_resume,
            "matching": matching,
            "scoring": scoring,
            "explanation": explanation,
        }
