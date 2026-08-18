import json
from pathlib import Path


REQUIRED_SECTIONS = [
    "candidate",
    "capabilities",
]


def validate_master_resume(file_path: str) -> tuple[bool, list[str]]:
    """Validate the basic structure of the master resume JSON."""

    errors = []
    path = Path(file_path)

    if not path.exists():
        return False, [f"File not found: {file_path}"]

    try:
        with path.open("r", encoding="utf-8") as file:
            resume = json.load(file)
    except json.JSONDecodeError as error:
        return False, [f"Invalid JSON: {error}"]

    for section in REQUIRED_SECTIONS:
        if section not in resume:
            errors.append(f"Missing required section: {section}")

    # Validate candidate section
    if "candidate" in resume:
        candidate = resume["candidate"]

        if "personal_info" not in candidate:
            errors.append("Missing required field: candidate.personal_info")

        if "professional_identity" not in candidate:
            errors.append(
                "Missing required field: candidate.professional_identity"
            )

    # Validate capabilities section
    if "capabilities" in resume:
        capabilities = resume["capabilities"]

        if "skills" not in capabilities:
            errors.append("Missing required field: capabilities.skills")

    return len(errors) == 0, errors


if __name__ == "__main__":
    file_path = "data/master_resume.json"

    is_valid, errors = validate_master_resume(file_path)

    if is_valid:
        print("✅ Master resume structure is valid.")
    else:
        print("❌ Master resume validation failed:")
        for error in errors:
            print(f"  - {error}")