# AI Resume Screening System with LangChain and LangSmith

This project implements a modular AI-powered resume screening pipeline:

Resume -> Extract -> Match -> Score -> Explain

## Tech Stack

- Python
- LangChain (PromptTemplate + LCEL + invoke)
- LangSmith tracing
- Groq model via `langchain-groq` (`llama-3.3-70b-versatile` by default)

## Project Structure

```text
prompts/
  skill_extraction.txt
  matching.txt
  scoring.txt
  explanation.txt
chains/
  resume_pipeline.py
data/
  job_description.txt
  resumes/
    strong_candidate.txt
    average_candidate.txt
    weak_candidate.txt
main.py
debug_case.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure environment variables:

```bash
copy .env.example .env
```

Then update `.env` with your keys:

- `GROQ_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGCHAIN_PROJECT`
- `LANGCHAIN_TRACING_V2=true`

## Run the Assignment Pipeline

```bash
python main.py
```

This runs 3 required candidate types:

- Strong candidate
- Average candidate
- Weak candidate

Results are printed and saved to `output_results.json`.

## LangSmith Requirements Coverage

The pipeline uses tags at each step:

- `step:extract`
- `step:match`
- `step:score`
- `step:explain`

For grading:

1. Run `python main.py` (creates at least 3 runs).
2. Open LangSmith and show full traces for each candidate.
3. Run `python debug_case.py` and show one incorrect output, then explain how to tighten prompt constraints.

## Notes on Prompt Engineering

- Prompts enforce JSON-only output.
- Prompts explicitly disallow assuming missing skills.
- Scoring prompt enforces score bounds and sum consistency.
