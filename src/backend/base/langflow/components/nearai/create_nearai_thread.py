import json
import logging

import httpx
from pydantic import SecretStr

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import IntInput, SecretStrInput, StrInput
from langflow.io import Output

logger = logging.getLogger(__name__)


class CreateNearAIThreadTool(LCToolComponent):
    display_name = "Create NEAR AI Thread"
    name = "CreateNearAIThreadTool"
    description = "Creates a NEAR AI thread using an agent run and outputs the thread_id. Usable as a LangChain tool."
    icon = "NearAI"

    inputs = [
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            required=True,
            info="JSON string containing NEAR credentials with 'auth'.",
        ),
        StrInput(
            name="agent_id",
            display_name="Agent ID",
            required=True,
            value=None,  # ✅ absolutely required for port to appear
            tool_mode=True,
            info="Full NEAR AI agent ID (e.g., ai-aaron.near/my-agent/1.0.0)",
        ),
        StrInput(
            name="new_message",
            display_name="Init Message",
            value="Initialize memory thread.",
            info="Initial message to seed the thread.",
        ),
        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            value=0,
            info="Number of iterations to allow the agent to run.",
        ),
    ]

    outputs = [Output(name="thread_id", display_name="Thread ID", method="build")]

    async def build(self) -> str:
        try:
            credentials_str = SecretStr(self.near_credentials).get_secret_value()
            credentials = json.loads(credentials_str)
            token = json.dumps(credentials["auth"])
        except Exception as e:
            raise ValueError(f"[CreateThreadTool] Failed to extract token: {e}")

        if not self.agent_id:
            raise ValueError("Missing required input: agent_id")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        payload = {
            "agent_id": self.agent_id.strip(),
            "new_message": self.new_message,
            "max_iterations": self.max_iterations,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"[CreateThreadTool] Payload: {payload}")
            response = await client.post("https://api.near.ai/v1/threads/runs", headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"[NEAR AI] Failed to create thread: {response.status_code} {response.text}")

        data = response.json()
        thread_id = data.get("thread_id") or data.get("id")
        if not thread_id:
            raise ValueError("No thread_id returned from NEAR AI.")

        logger.info(f"[CreateThreadTool] Created thread_id: {thread_id}")
        return thread_id
