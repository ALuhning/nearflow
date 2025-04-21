import json
import asyncio
import httpx
from pydantic import SecretStr
from langflow.base.memory.model import LCChatMemoryComponent
from langflow.inputs import SecretStrInput, StrInput
from langflow.io import Output
from langflow.field_typing.constants import Memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langflow.logging import logger

class NearAIChatMemory(LCChatMemoryComponent):
    display_name = "NearAI Chat Memory"
    name = "NearAIChatMemory"
    description = "Creates and stores NEAR AI chat memory using threads and agents."
    icon = "NearAI"

    inputs = [
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            required=True,
            info="JSON with 'auth' containing token and account_id."
        ),
        StrInput(
            name="agent_name",
            display_name="Agent Name",
            required=True,
            info="Name of the NEAR AI agent to use or register."
        ),
        StrInput(
            name="model",
            display_name="Model",
            value="openai/gpt-3.5-turbo",
            required=False,
            advanced=True,
            info="Model used if agent needs to be registered."
        ),
        StrInput(
            name="description",
            display_name="Agent Description",
            value="Langflow-generated memory agent.",
            required=False,
            advanced=True
        ),
        StrInput(
            name="base_url",
            display_name="API Base URL",
            value="https://api.near.ai/v1",
            required=True,
            advanced=True
        )
    ]

    outputs = [
        Output(name="memory", display_name="Memory", method="build_message_history")
    ]

    def build_message_history(self) -> Memory:
        if not hasattr(self, "_persistent_thread_id"):
            self._persistent_thread_id = None
        return NearAIChatMessageHistory(
            near_credentials=self.near_credentials,
            agent_name=self.agent_name,
            model=self.model,
            description=self.description,
            base_url=self.base_url,
            thread_id=self._persistent_thread_id  # ✅ supply it if it exists
        )

class NearAIChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, near_credentials, agent_name, model, description, base_url, thread_id=None):
        self.agent_name = agent_name
        self.model = model or "openai/gpt-3.5-turbo"
        self.description = description or "Langflow auto-generated agent"
        self.base_url = base_url.rstrip("/")
        self.credentials = json.loads(SecretStr(near_credentials).get_secret_value())
    
        self.token = self._get_token()
        self.owner_id = self._get_owner_id()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
        self.thread_id = thread_id

    def _get_token(self):
        auth = json.dumps(self.credentials["auth"])
        return auth

    def _get_owner_id(self):
        return self.credentials["auth"]["account_id"]

    def clear(self) -> None:
        # No-op: clearing not currently supported in NEAR AI
        pass

    async def aget_messages(self):
        if not self.thread_id:
            await self.ensure_agent_and_thread()

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.get(
                f"{self.base_url}/threads/{self.thread_id}/messages",
                headers=self.headers
            )
            resp.raise_for_status()
            data = resp.json()

        messages = []
        for msg in data.get("data", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    async def aadd_message(self, message):
        if not self.thread_id:
            await self.ensure_agent_and_thread()

        payload = {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": message.content
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(
                f"{self.base_url}/threads/{self.thread_id}/messages",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()

    async def ensure_agent_and_thread(self):
        if self.thread_id:
            return
        def safe_get_agents(resp_json):
            if isinstance(resp_json, dict):
                return resp_json.get("data", [])
            elif isinstance(resp_json, list):
                return resp_json
            return []

        async def fetch_agents():
            payload = {
                "owner_id": self.owner_id,
                "with_capabilities": False,
                "latest_versions_only": True,
                "limit": 100,
                "offset": 0
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    f"{self.base_url}/find_agents",
                    headers=self.headers,
                    json=payload
                )
                resp.raise_for_status()
                return safe_get_agents(resp.json())

        async def upload_metadata():
            payload = {
                "metadata": {
                    "category": "agent",
                    "description": self.description,
                    "tags": ["langflow"],
                    "details": {},
                    "show_entry": False
                },
                "entry_location": {
                    "namespace": self.owner_id,
                    "name": self.agent_name,
                    "version": "1.0.0"
                }
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    f"{self.base_url}/registry/upload_metadata",
                    headers=self.headers,
                    json=payload
                )
                resp.raise_for_status()

        async def create_thread(agent_id):
            payload = {
                "agent_id": agent_id,
                "new_message": "",
                "max_iterations": 0
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(
                    f"{self.base_url}/threads/runs",
                    headers=self.headers,
                    json=payload
                )
                try:
                    data = resp.json()
                    if isinstance(data, str):
                        return data
                    if isinstance(data, dict):
                        return data.get("thread_id") or data.get("id")
                    raise TypeError(f"[NearAI] Unexpected response type: {type(data).__name__} - {data}")
                except Exception as e:
                    raw = resp.text.strip()
                    if raw.startswith("thread_"):
                        return raw
                    raise Exception(f"[NearAI] Failed to parse thread response: {raw}") from e

        print(f"[NearAI] Looking for agent '{self.agent_name}' for user '{self.owner_id}'...")
        agents = await fetch_agents()
        agent = next(
            (a for a in agents if isinstance(a, dict)
             and a.get("name") == self.agent_name
             and a.get("namespace") == self.owner_id),
            None
        )

        if not agent:
            print(f"[NearAI] Agent not found. Registering '{self.agent_name}'...")
            await upload_metadata()
            await asyncio.sleep(1)
            agents = await fetch_agents()
            agent = next(
                (a for a in agents if isinstance(a, dict)
                 and a.get("name") == self.agent_name
                 and a.get("namespace") == self.owner_id),
                None
            )
            if not agent:
                raise Exception("[NearAI] Agent registration failed. Could not retrieve after creation.")

        agent_id = f"{agent['namespace']}/{agent['name']}/{agent['version']}"
        print(f"[NearAI] Using agent: {agent_id}")

        self.thread_id = await create_thread(agent_id)
        
        if not self.thread_id:
            self.thread_id = await create_thread(agent_id)
            logger.info(f"[NearAI] Created new thread: {self.thread_id}")
        else:
            logger.info(f"[NearAI] Reusing existing thread: {self.thread_id}")

        # persist for reuse
        if hasattr(self, "set_persistent_thread_id"):
            self.set_persistent_thread_id(self.thread_id)
            
    def set_persistent_thread_id(self, tid):
        try:
            from langflow.base.memory.model import LCChatMemoryComponent
            LCChatMemoryComponent._persistent_thread_id = tid
        except Exception as e:
            print(f"[NearAI] Could not persist thread ID: {e}")
