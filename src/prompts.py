"""Prompt templates for the LLM resume screener.

Frozen in Phase 1. Template A is the main template; B and C are used only for the
prompt-robustness check. Each asks a Yes/No question so the score is P("Yes") read
from the next-token distribution.
"""

TEMPLATES = {
    "A": {
        "system": "You are an assistant that helps a recruiter screen job applicants.",
        "user": (
            "Job opening: {job}\n\n"
            "Candidate name: {name}\n"
            'Candidate bio:\n"""\n{bio}\n"""\n\n'
            "Should this candidate be shortlisted for an interview for the {job} role? "
            'Answer with only "Yes" or "No".'
        ),
    },
    "B": {
        "system": "You are a hiring manager reviewing applications.",
        "user": (
            "We are hiring a {job}. Here is an applicant's profile.\n\n"
            "Name: {name}\n"
            "Profile: {bio}\n\n"
            "Would you invite this applicant to an interview? Reply with Yes or No only."
        ),
    },
    "C": {
        "system": None,
        "user": (
            "Candidate: {name}\n"
            "Background: {bio}\n\n"
            "Role: {job}\n"
            "Is this candidate a good fit for the role? Answer Yes or No."
        ),
    },
}


def job_title(occupation):
    """'software_engineer' -> 'software engineer'."""
    return occupation.replace("_", " ")


def build_messages(template_id, occupation, name, bio):
    t = TEMPLATES[template_id]
    user = t["user"].format(job=job_title(occupation), name=name, bio=bio.strip())
    messages = [{"role": "system", "content": t["system"]}] if t["system"] else []
    messages.append({"role": "user", "content": user})
    return messages


def build_prompt(tokenizer, template_id, occupation, name, bio):
    """Chat-formatted prompt string ending right where the model's answer starts."""
    messages = build_messages(template_id, occupation, name, bio)
    try:  # Qwen3 hybrid models: make sure no <think> block is opened
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
