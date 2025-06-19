import os
import json
import time
import uuid
import unicodedata
from dotenv import load_dotenv
from neo4j import GraphDatabase
from evaluator import evaluate_with_ollama
from chains import load_llm
from utils import extract_json_from_response

# Load environment variables
load_dotenv()

models = [m.strip() for m in os.getenv("EVALUATOR_MODELS", "").split(",") if m.strip()]
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

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

# Load input file
with open("data/questions_all_augmented.json", encoding="utf-8") as f:
    questions = json.load(f)

# Ensure errors folder
os.makedirs("errors", exist_ok=True)

# Neo4j connection
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

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

# Evaluation loop
for model in models:
    os.environ["LLM_MODEL"] = model
    llm = load_llm()

    print(f"\n🚀 Evaluating with model: {model}")

    with driver.session() as session:
        for idx, item in enumerate(questions, 1):
            student_id = normalize(item["student_id"])
            question_text = normalize(item["question"])
            rubric_type = normalize(item["rubric"])
            answer = normalize(item["answer"])
            question_id = item["question_id"]
            answer_id = item["answer_id"]

            print(f"🔄 Model: {model} | Student: {student_id} | Q: {idx}/{len(questions)}")

            try:
                start_time = time.time()
                response = evaluate_with_ollama(question_text, answer, rubrics[rubric_type], llm)
                duration = round(time.time() - start_time, 2)

                evaluation_id = str(uuid.uuid4())

                if "no student answer" in response.lower():
                    print("ℹ️ No student answer provided. Grade = 1.")
                    evaluation_result = {c["name"]: 1 for c in rubrics[rubric_type]["criteria"]}
                    evaluation_result["feedback"] = "No answer provided. Grade set to minimum."
                else:
                    evaluation_result = extract_json_from_response(response)
                    if isinstance(evaluation_result.get("feedback"), dict):
                        evaluation_result["feedback"] = "\n\n".join(
                            f"- {k.capitalize()}: {v}" for k, v in evaluation_result["feedback"].items())

                props = {
                    "id": evaluation_id,
                    "student_id": student_id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "answer_id": answer_id,
                    "answer": answer,
                    "rubric_type": rubric_type,
                    "model_name": model,
                    "model_response": response,
                    "duration": duration,
                    **evaluation_result
                }

                session.execute_write(save_evaluation, props)

                final_grade = props["final_grade"]
                print(f"✅ Saved | Student: {student_id} | Time: {duration}s | Grade: {final_grade}")

            except Exception as e:
                err_file = f"errors/error_{student_id}_{model.replace(':', '_')}_Q{idx}.txt"
                with open(err_file, "w", encoding="utf-8") as ef:
                    ef.write(f"Student: {student_id}\nModel: {model}\nQuestion: {question_text}\n"
                             f"Answer: {answer}\n\nError: {str(e)}\n\n"
                             f"Model Response:\n{response if 'response' in locals() else 'N/A'}\n")
                print(f"❌ Error logged: {err_file}")

driver.close()
print("\n🎯 Evaluare completă.")
