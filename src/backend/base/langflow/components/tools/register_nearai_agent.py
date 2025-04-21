from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import SecretStrInput, StrInput
from langflow.io import Output
import httpx
import json
from pydantic import SecretStr


class RegisterNearAIAgentTool(LCToolComponent):
    display_name = "Register NEAR AI Agent"
    name = "RegisterNearAIAgentTool"
    description = "Registers a new NEAR AI agent if it doesn't already exist."
    icon = "NearAI"

    inputs = [
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            info="JSON with 'auth' key from NEAR credentials.",
            required=True,
        ),
        StrInput(
            name="agent_id",
            display_name="Agent ID",
            required=True,
            info="Full ID like yourname.near/my-agent/1.0.0",
            tool_mode=True
        ),
        StrInput(
            name="agent_name",
            display_name="Agent Name",
            required=True,
        ),
        StrInput(
            name="description",
            display_name="Description",
            value="A NEAR AI agent created via Langflow",
        ),
        StrInput(
            name="model",
            display_name="Model",
            value="openai/gpt-3.5-turbo",
            info="The model ID to assign to the agent.",
        )
    ]

    outputs = [
        Output(name="agent_id_out", display_name="Agent ID", method="build")
    ]

    def get_api_token(self):
        credentials_str = SecretStr(self.near_credentials).get_secret_value()
        credentials_json = json.loads(credentials_str)
        return json.dumps(credentials_json["auth"])
    
    def get_account_id(self):
        credentials_str = SecretStr(self.near_credentials).get_secret_value()
        credentials_json = json.loads(credentials_str)
        credentials = json.dumps(credentials_json["auth"]["account_id"])
        account_id = credentials.replace('"', '')
        if not account_id:
            raise ValueError("account_id is missing from NEAR credentials")
        return account_id

    async def build(self) -> str:
        base_url = "https://api.near.ai/v1"
        token = self.get_api_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        owner_id = self.get_account_id()

        # Step 1: Check if agent already exists
        find_payload = {
            "owner_id": owner_id,
            "with_capabilities": False,
            "latest_versions_only": True,
            "limit": 100,
            "offset": 0
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            find_resp = await client.post(
                f"{base_url}/find_agents",
                headers=headers,
                json=find_payload
            )

        if find_resp.status_code != 200:
            raise Exception(f"[NEAR AI] Failed to query agents: {find_resp.status_code} {find_resp.text}")

        existing_agents = find_resp.json()
        print(f"[Agent search] Found {len(existing_agents)} agents for {owner_id}")

        matching_agent = next(
            (agent for agent in existing_agents if agent.get("name") == self.agent_name and agent.get("namespace") == owner_id),
            None
        )
        print(f"matching_agent: {matching_agent}")
        if matching_agent:
            return f"{matching_agent.get('namespace')}/{matching_agent.get('name')}/{matching_agent.get('version')}"
        else:
            raise ValueError("Agent appears to be registered but couldn't be found.")
            
        # Step 2: Register agent metadata
        payload = {
            "metadata": {
                "category": "agent",
                "description": self.description,
                "tags": ["langflow"],
                "details": {},
                "show_entry": False
            },
            "entry_location": {
                "namespace": owner_id,
                "name": self.agent_name,  # ✅ Not a set
                "version": "1.0.0"
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/registry/upload_metadata",
                headers=headers,
                json=payload
            )
            
            # Step 2: Re-query find_agents to get agent's full ID
            find_payload = {
                "owner_id": owner_id,
                "with_capabilities": False,
                "latest_versions_only": True,
                "limit": 100,
                "offset": 0
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                find_resp = await client.post(
                    f"{base_url}/find_agents",
                    headers=headers,
                    json=find_payload
                )
            
            if find_resp.status_code != 200:
                raise Exception(f"[NEAR AI] Failed to re-fetch agents: {find_resp.status_code} {find_resp.text}")
            
            agents = find_resp.json()
            
            # Step 3: Find matching agent (by name or agent_id)
            matching_agent = next(
                (agent for agent in agents if agent.get("name") == self.agent_name and agent.get("namespace") == owner_id),
                None
            )
            
            if not matching_agent:
                raise ValueError("Agent was registered but could not be found in follow-up /find_agents call.")
            
            # Step 4: Return the system-assigned NEAR agent `id` (not the path)
            return matching_agent.get("id")
            
        if response.status_code != 200:
            raise Exception(f"[NEAR AI] Failed to register agent: {response.status_code} {response.text}")
        
        return self.agent_id  # ✅ Correct return type

