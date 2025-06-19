import json
import re
import ast
import unicodedata

def extract_json_from_response(response: str):
    """
    Extracts JSON block from response using multiple fallback strategies.
    Raises informative errors when extraction fails.
    """
    # First attempt: JSON block between triple backticks
    json_blocks = re.findall(r"```(?:json)?\s*({[\s\S]*?})\s*```", response, re.MULTILINE)

    # Second attempt if no triple backticks found: first JSON-like block
    if not json_blocks:
        json_blocks = re.findall(r"({[\s\S]*?})", response, re.MULTILINE)

    if not json_blocks:
        raise ValueError("No JSON blocks found in response.")

    # Attempt parsing each JSON block found
    for json_text in json_blocks:
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try sanitizing common problematic characters
            safe_text = json_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            try:
                return json.loads(safe_text)
            except json.JSONDecodeError:
                # Last resort: ast.literal_eval
                try:
                    fixed_text = re.sub(r'(?<!\\)"(.*?)"(?!\\)',
                                        lambda m: m.group(0).replace('\n', '\\n'), safe_text)
                    return ast.literal_eval(fixed_text)
                except Exception:
                    continue  # Continue with next found block if current fails

    # If all blocks failed to parse
    raise ValueError(f"Unable to parse any JSON block from response. Response was:\n{response}")

# Normalize text (punctuație + unicode)
def normalize(text):
    replacements = {'–': '-', '—': '-', '‑': '-', '“': '"', '”': '"', '‘': "'", '’': "'"}
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return unicodedata.normalize('NFKC', text).strip()

# Load rubrics
def load_rubrics():
    with open("rubrics/technical.json", "r", encoding="utf-8") as f:
        rubric_tech = json.load(f)
    with open("rubrics/argumentative.json", "r", encoding="utf-8") as f:
        rubric_arg = json.load(f)
    return {"technical": rubric_tech, "argumentative": rubric_arg}

rubrics = load_rubrics()


# Save evaluation to Neo4j
def save_evaluation(tx, props):
    criteria = rubrics[props["rubric_type"]]["criteria"]
    final_grade = round(sum(props.get(c["name"], 0) for c in criteria) / len(criteria), 2)
    props["final_grade"] = final_grade

    query = """
    MERGE (s:Student {id: $student_id})
    MERGE (q:Question {id: $question_id})
    SET q.text = $question_text, q.rubric = $rubric_type
    MERGE (a:Answer {id: $answer_id})
    SET a.answer = $answer, a.student_id = $student_id, a.question_id = $question_id
    MERGE (m:Model {name: $model_name})
    CREATE (e:Evaluation)
    SET e += $props
    MERGE (s)-[:ANSWERED]->(a)
    MERGE (a)-[:RESPONDS_TO]->(q)
    MERGE (a)-[:EVALUATED_BY]->(e)
    MERGE (e)-[:EVALUATES_QUESTION]->(q)
    MERGE (e)-[:FOR_STUDENT]->(s)
    MERGE (e)-[:USING_MODEL]->(m)
    """

    tx.run(query,
           student_id=props["student_id"],
           question_id=props["question_id"],
           question_text=props["question_text"],
           rubric_type=props["rubric_type"],
           answer_id=props["answer_id"],
           answer=props["answer"],
           model_name=props["model_name"],
           props=props)
