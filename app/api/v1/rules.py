from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.rules.models import ActionRule
from app.rules.registry import action_rule_registry
from app.tools.registry import tool_registry
from app.tools.action_tool import ClientActionTool

router = APIRouter()


@router.get("/rules", response_model=List[ActionRule], summary="List all external Action Rules")
async def list_action_rules(domain: Optional[str] = Query(default=None, description="Filter by domain (LMS, LIBRARY, PORTAL)")):
    return action_rule_registry.list_rules(domain=domain)


@router.get("/rules/{action_id}", response_model=ActionRule, summary="Get Action Rule by ID")
async def get_action_rule(action_id: str):
    rule = action_rule_registry.get_rule(action_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Action rule '{action_id}' not found")
    return rule


@router.put("/rules/{action_id}", response_model=ActionRule, summary="Hotfix Action Rule without app deployment")
async def update_action_rule(action_id: str, rule: ActionRule):
    if rule.action_id != action_id:
        raise HTTPException(status_code=400, detail="Path action_id does not match body action_id")

    action_rule_registry.upsert_rule(rule)
    # Re-register corresponding tool
    tool_registry.register(ClientActionTool(rule))
    return rule
