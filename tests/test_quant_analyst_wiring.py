def test_quant_in_analyst_node_specs():
    from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

    assert "quant" in ANALYST_NODE_SPECS
    spec = ANALYST_NODE_SPECS["quant"]
    assert spec.report_key == "quant_report"
    assert spec.agent_node == "Quant Forecast Analyst"


def test_quant_report_field_in_agent_state():
    from tradingagents.agents.utils.agent_states import AgentState

    assert "quant_report" in AgentState.__annotations__


def test_quant_report_initialized_by_propagator():
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("NVDA", "2026-06-30")
    assert state["quant_report"] == ""


def test_should_continue_quant_exists():
    from tradingagents.graph.conditional_logic import ConditionalLogic

    assert hasattr(ConditionalLogic(), "should_continue_quant")


def test_analyst_type_quant_exists():
    from cli.models import AnalystType

    assert AnalystType.QUANT == "quant"


def test_default_selected_analysts_includes_quant():
    import inspect

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sig = inspect.signature(TradingAgentsGraph.__init__)
    assert "quant" in sig.parameters["selected_analysts"].default
