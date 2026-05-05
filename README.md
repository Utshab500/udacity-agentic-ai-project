# Udacity Agentic AI Project

## Module - Building Agents
Here I build an AI Agent which implements below items 
1. State Machine
2. Session Handeling
3. Long-Term Memory for multiple query handling in same `session` and remembering <b>previous context</b>.
4. Tool calling for
    - Semantic search from RAG pipeline.
    - Result Evaluation.
    - Registering user data in `Long-Term memory` to remember user specific context, in a session.
    - Getting user preferences from the `previous context` in a `session`.

## Module - Multi-Agent System
Here I build an multi agent system which helps the Munder Difflin paper company (virtual) to solve and enhance their paper ordering system with restocking inventory and managing profitability. This systme consists below items

1. Orchestrator Agent.
2. Various agents to solve specific task with their own tools.
3. DB to store intermediate agent communications.
