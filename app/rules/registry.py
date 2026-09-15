"""
Action Rule Registry with in-memory store and hotfix update support
"""
from typing import Dict, List, Optional
from app.rules.models import ActionRule
from app.rules.catalog import DEFAULT_ACTION_RULES
from app.core.logging import logger


class ActionRuleRegistry:
    def __init__(self):
        # Initialize with pre-defined rules
        self._rules: Dict[str, ActionRule] = dict(DEFAULT_ACTION_RULES)
        logger.info(f"ActionRuleRegistry initialized with {len(self._rules)} default rules.")

    def get_rule(self, action_id: str) -> Optional[ActionRule]:
        return self._rules.get(action_id)

    def list_rules(self, domain: Optional[str] = None) -> List[ActionRule]:
        if domain:
            d = domain.upper()
            return [r for r in self._rules.values() if r.domain == d]
        return list(self._rules.values())

    def upsert_rule(self, rule: ActionRule) -> None:
        self._rules[rule.action_id] = rule
        logger.info(f"ActionRule [{rule.action_id}] hotfixed/registered (Version: {rule.version})")


action_rule_registry = ActionRuleRegistry()
