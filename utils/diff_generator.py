def generate_plan_diff(old_plan: dict | None, new_plan: dict | None) -> list[str]:
    """Generate human-readable diff between two plans."""
    changes = []

    if old_plan is None and new_plan is not None:
        changes.append(f"+ Created plan: {new_plan.get('title', 'Untitled')}")
        for step in new_plan.get("steps", []):
            changes.append(f"  + Added step {step['step_number']}: {step['title']}")
        return changes

    if old_plan is None or new_plan is None:
        return changes

    # title change
    if old_plan.get("title") != new_plan.get("title"):
        changes.append(f"~ Renamed: {old_plan.get('title')} -> {new_plan.get('title')}")

    old_steps = {s["step_number"]: s for s in old_plan.get("steps", [])}
    new_steps = {s["step_number"]: s for s in new_plan.get("steps", [])}

    old_nums = set(old_steps.keys())
    new_nums = set(new_steps.keys())

    # removed steps
    for num in old_nums - new_nums:
        changes.append(f"- Removed step {num}: {old_steps[num]['title']}")

    # added steps
    for num in new_nums - old_nums:
        changes.append(f"+ Added step {num}: {new_steps[num]['title']}")

    # modified steps
    for num in old_nums & new_nums:
        old_step = old_steps[num]
        new_step = new_steps[num]
        if old_step["title"] != new_step["title"]:
            changes.append(
                f"~ Modified step {num}: {old_step['title']} -> {new_step['title']}"
            )
        elif old_step.get("description") != new_step.get("description"):
            changes.append(f"~ Updated step {num} description: {new_step['title']}")
        elif old_step.get("status") != new_step.get("status"):
            changes.append(
                f"~ Step {num} status: {old_step.get('status')} -> {new_step.get('status')}"
            )

    return changes
