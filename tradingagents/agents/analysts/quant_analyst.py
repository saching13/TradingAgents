from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_kronos_forecast,
    get_language_instruction,
)


def create_quant_analyst(llm):

    def quant_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_kronos_forecast]

        system_message = (
            """You are a quantitative trading assistant. Your one job is to call get_kronos_forecast for this ticker and the current date, then report what it says.

Kronos is an autoregressive transformer foundation model trained on historical K-line (OHLCV) data across 45+ global exchanges. It produces a probabilistic forecast of the future price path, not a guarantee or a fundamental analysis. Treat a single sample_count=1 run as one draw from a distribution, not a certainty.

In your report:
- State the predicted direction, the % change in close over the forecast horizon, and the predicted path's high/low.
- Note the forecast horizon explicitly (how many trading days ahead).
- If this forecast's direction disagrees with what a reasonable technical read of recent price action would suggest, say so explicitly rather than silently blending the two views — flag the disagreement for the research team to weigh.
- Do not invent confidence levels or probabilities the tool did not report."""
            + """ Make sure to append a Markdown table at the end of the report showing the predicted daily OHLCV path."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "quant_report": report,
        }

    return quant_analyst_node
