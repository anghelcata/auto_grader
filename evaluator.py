import os
from chains import load_llm

def evaluate_with_ollama(question, answer, rubric, llm=None):
    """
    Build a prompt based on the rubric and send it to the LLM.
    If no LLM is provided, load it from .env using load_llm().
    Returns the raw string response from the model.
    """

    if llm is None:
        llm = load_llm()

    # Format rubric criteria into a bulleted list
    criteria_list = "\n".join([
        f"- {c['name']}: {c['description']}" for c in rubric["criteria"]
    ])

    # Determine if rubric is argumentative
    is_argumentative = any(c["name"] == "dialecticality" for c in rubric["criteria"])

    prompt_intro = "You are an expert evaluator of Computer Science student answers."
    if is_argumentative:
        prompt_intro += " Focus on argument quality, structure, creativity, and dialectical engagement."
    else:
        prompt_intro += " Focus on technical correctness, clarity, completeness, and appropriate terminology."

    # Build JSON field structure
    json_fields = ",\n  ".join([f'\"{c["name"]}\": [1-10]' for c in rubric["criteria"]])
    json_fields += ',\n  \"feedback\": \"Constructive and specific feedback for the student\"'

    # Compose the full prompt with stricter instructions
    prompt = f"""
    {prompt_intro}

    Evaluate the student's answer using the rubric below.
    Each criterion must be scored from 1 (very poor) to 10 (excellent).

    {criteria_list}

    Question: {question}
    Student's Answer: {answer}

    IMPORTANT: Provide ONLY the evaluation in JSON format as shown below. Do NOT include explanations, additional text, or formatting characters.

    {{
      {json_fields}
    }}
    """.strip()

    print(f"\n🔎 Using LLM: {os.getenv('LLM_MODEL')}")

    response = llm.invoke(prompt)

    return response
