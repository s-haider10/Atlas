# tools/tool_registry.py
"""
Factory functions to convert ATLAS scenario tool definitions into
Inspect AI Tool objects for model evaluation.

Two tool types:
  - Stub tools: generic tools that return a success message (IC, NO, MDI operators)
  - TRC tools: tools that return an exact `injected_response` to test injection resistance
"""
import re
from inspect_ai.tool import tool, Tool


def _sanitize_name(name: str) -> str:
    """Convert arbitrary tool name to a valid Python identifier."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "tool_" + s
    return s


def make_stub_tool(name: str, description: str = "") -> Tool:
    """Create a generic tool that returns a plausible success message."""
    safe_name = _sanitize_name(name)
    desc = description or f"Tool: {name}"

    @tool(name=safe_name)
    def stub_tool() -> Tool:
        async def execute(input: str = "") -> str:
            """Perform an operation.

            Args:
                input: The input or query for this tool.
            """
            return f"[{name}] Operation completed successfully."
        execute.__doc__ = f"""{desc}

Args:
    input: The input or query for this tool.
"""
        return execute

    return stub_tool()


def make_trc_tool(tool_def: dict) -> Tool:
    """Create a TRC tool that returns the exact injected_response.

    The injected response contains adversarial content disguised as tool output.
    A safe model should complete its task but resist acting on the injection.
    """
    name = tool_def.get("name", "unknown_tool")
    safe_name = _sanitize_name(name)
    desc = tool_def.get("description", f"Tool: {name}")
    response = tool_def.get("injected_response", f"[{name}] Done.")

    @tool(name=safe_name)
    def trc_tool() -> Tool:
        async def execute(input: str = "") -> str:
            """Perform an operation.

            Args:
                input: The input or query for this tool.
            """
            return response
        execute.__doc__ = f"""{desc}

Args:
    input: The input or query for this tool.
"""
        return execute

    return trc_tool()


def build_tools_for_scenario(tools_data: list, operator: str) -> list[Tool]:
    """Build Inspect Tool objects from scenario tool definitions.

    Args:
        tools_data: List of tool definitions. For TRC: list of dicts with
            name/description/parameters/injected_response. For others: list of
            strings (tool names) or dicts.
        operator: Composition operator code (IC, TRC, NO, SE, MDI).

    Returns:
        List of Inspect Tool objects ready for state.tools injection.
    """
    if not tools_data:
        return []

    built = []
    for item in tools_data:
        if operator == "TRC" and isinstance(item, dict):
            built.append(make_trc_tool(item))
        elif isinstance(item, dict):
            built.append(make_stub_tool(
                item.get("name", "tool"),
                item.get("description", ""),
            ))
        elif isinstance(item, str):
            built.append(make_stub_tool(item))

    return built
