from langflow.base.agents.events import ExceptionWithMessageError
from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.io import (
    BoolInput,
    DropdownInput,
    MultilineInput,
    SecretStrInput,
    Output,
)
from langflow.schema.message import Message
from langflow.logging import logger
from pydantic import SecretStr
import json
import httpx
from typing import Union, List, Dict, Any, Optional


class NearAIAgentComponent(ToolCallingAgentComponent):
    display_name = "NearAI Agent"
    description = "Multi-tool agent using NEAR AI SDK with Langflow tool input support."
    icon = "NearAI"
    name = "NearAIAgent"

    _model_display_map = {}
    default_credentials = ""
    nearai_api_base = "https://api.near.ai/v1"
    _openai_models = []

    # Pull in all memory inputs without modifying them
    memory_inputs = MemoryComponent().inputs

    # Find and patch the main memory input only (usually named "memory")
    for component_input in memory_inputs:
        if component_input.name == "memory":
            component_input.advanced = False

    @classmethod
    def fetch_openai_models(cls, api_key=None, base_url=None):
        from openai import OpenAI
        try:
            if not api_key:
                return cls._openai_models
            client = OpenAI(api_key=api_key, base_url=base_url or cls.nearai_api_base)
            response = client.models.list()
            cls._openai_models = [m.id for m in response.data]
            cls._model_display_map = {
                cls.format_model_display_name(m): m for m in cls._openai_models
            }
            return cls._openai_models
        except Exception as e:
            logger.error(f"[NearAI] Model fetch error: {e}")
            return cls._openai_models

    @classmethod
    def format_model_display_name(cls, name: str):
        if "::" in name:
            provider, path = name.split("::", 1)
            return f"{provider} - {path.split('/')[-1]}"
        return name

    def update_build_config(cls, build_config, field_value, field_name=None):
        try:
            credentials = SecretStr(cls.near_credentials).get_secret_value()
            api_key = json.loads(credentials)["auth"]
        except Exception as e:
            logger.warning(f"[NearAI] Credential parse error: {e}")
            api_key = None

        if field_name in {"nearai_api_base", "model_name"}:
            models = cls.fetch_openai_models(api_key)
            cls._model_display_map = {
                cls.format_model_display_name(m): m for m in models
            }
            cls._openai_models = models
            build_config["model_name"]["options"] = list(cls._model_display_map.keys())

        return build_config

    # Define the inputs list including memory with the correct type
    inputs = [
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=[],
            refresh_button=True,
        ),
        MultilineInput(
            name="input_value",
            display_name="User Input",
            required=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
        ),
        # Include the memory input with the correct type
        DropdownInput(
            name="memory",
            display_name="Memory",
            info="Connect a memory component to enable conversation history",
            input_types=["Memory"],
        ),
        BoolInput(
            name="add_current_date_tool",
            display_name="Add Current Date Tool",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="force_new_thread",
            display_name="Start Fresh Thread",
            value=False,
            advanced=False,
        ),
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            info="JSON containing API key",
            value=default_credentials,
            required=True,
        ),
        *LCToolsAgentComponent._base_inputs,
        # Add memory inputs for backwards compatibility but mark them as advanced
        *memory_inputs,
    ]

    outputs = [
        Output(name="value", display_name="Chat (Message)", method="message_response")
    ]

    def get_credentials_api_key(self):
        if not hasattr(self, "near_credentials") or not self.near_credentials:
            return None
        try:
            credentials_str = SecretStr(self.near_credentials).get_secret_value()
            credentials_json = json.loads(credentials_str)
            return json.dumps(credentials_json["auth"])
        except Exception as e:
            logger.error(f"[get_credentials_api_key] Failed: {e}")
            return None

    async def run_near_ai_completion(self, api_key, base_url, model, messages, tools):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.7
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"[NEARAI Payload]: {payload}")
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"[NEARAI ERROR] Response: {response.text}")
                response.raise_for_status()
            logger.info(f"[NEARAI response]: {response.json()}")
            return response.json()

    def _format_chat_history_for_api(self, chat_history: List[Any]) -> List[Dict[str, str]]:
        formatted = []
    
        if not isinstance(chat_history, list):
            logger.warning(f"[Chat Format] Chat history is not a list: {type(chat_history)}")
            return []
    
        for msg in chat_history:
            role = None
            raw_content = None
            try:
                # Determine role and extract content
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    role = "user" if msg.type == "human" else "assistant"
                    raw_content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    raw_content = msg.get("content", "")
                else:
                    logger.warning(f"[Chat Format] Skipping unrecognized message: {msg}")
                    continue
    
                # Unwrap content using shared logic
                content = self._extract_value(raw_content)
    
                logger.debug(f"[Chat Format] Message formatted: role={role}, content={content}")
                formatted.append({"role": role, "content": content})
            except Exception as e:
                logger.warning(f"[Chat Format] Failed to process message: {msg} → {e}")
                continue
    
        return formatted


    async def message_response(self) -> Message:
        try:
            logger.info("[NearAI] Starting message_response")
    
            # Step 1: Get API credentials
            api_key = self.get_credentials_api_key()
            if not api_key:
                raise ValueError("Missing or invalid NEAR AI credentials")
    
            # Step 2: Load memory history (already sanitized)
            memory_data = await self.get_memory_data()
            self.chat_history = memory_data or []
            logger.info(f"[NearAI] Retrieved memory data: {len(self.chat_history)} messages")
    
            # Step 3: Fetch available models if needed
            if not self.__class__._openai_models:
                self.__class__.fetch_openai_models(api_key=api_key, base_url=self.nearai_api_base)
    
            # Step 4: Resolve model name
            model_name = self.model_name
            if not self.__class__._model_display_map:
                self.__class__._model_display_map = {
                    self.__class__.format_model_display_name(m): m for m in self.__class__._openai_models
                }
            resolved_model = self.__class__._model_display_map.get(model_name, model_name)
            base_url = self.nearai_api_base
    
            # Step 5: Parse and register tool schemas
            openai_tool_schemas = []
            for i, tool in enumerate(getattr(self, "tools", []) or []):
                if tool and hasattr(tool, "name") and hasattr(tool, "func"):
                    args_schema = getattr(tool, "args_schema", None)
                    annotations = getattr(args_schema, "__annotations__", {}) or {}
                    openai_tool_schemas.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or f"{tool.name} tool",
                            "parameters": {
                                "type": "object",
                                "properties": {k: {"type": "string"} for k in annotations},
                                "required": list(annotations)
                            }
                        }
                    })
                    logger.info(f"[Tool schema] Tool #{i} registered: {tool.name}")
    
            # 6. Format messages
            formatted_history = self._format_chat_history_for_api(self.chat_history)
            messages = [{"role": "system", "content": self.system_prompt}]
            if formatted_history:
                messages.extend(formatted_history)
    
            clean_user = self._extract_value(self.input_value)
            messages.append({"role": "user", "content": clean_user})
    
            logger.debug(f"[NearAI] Final payload messages: {json.dumps(messages, indent=2)}")
    
            # Step 7: Add system and user messages
            messages = [{"role": "system", "content": self.system_prompt}]
            if formatted_history:
                messages.extend(formatted_history)
            messages.append({"role": "user", "content": self._extract_value(self.input_value)})
    
            logger.debug(f"[NearAI] Final payload messages: {json.dumps(messages, indent=2)}")
    
            # 7. Run initial API call
            response = await self.run_near_ai_completion(api_key, base_url, resolved_model, messages, openai_tool_schemas)
            if not response or "choices" not in response or not response["choices"]:
                return Message(text="[NEARAI ERROR] No response generated.", sender="Assistant")
    
            message = response["choices"][0].get("message", {})
            tool_calls = message.get("tool_calls")
    
            # 8. Tool-call follow-up logic
            if tool_calls:
                logger.info(f"[NearAI] Tool calls received: {tool_calls}")
                messages.append({"role": "assistant", "tool_calls": tool_calls})
    
                for call in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", "call_id"),
                        "name": call["function"]["name"],
                        "content": "[Tool response not executed locally]"
                    })
    
                followup = await self.run_near_ai_completion(api_key, base_url, resolved_model, messages, openai_tool_schemas)
                followup_msg = followup["choices"][0].get("message", {})
                assistant_msg = followup_msg.get("content", "[⚠️ Empty followup response]")
            else:
                assistant_msg = message.get("content", "[No assistant reply]")
    
            # 9. Clean and append to memory
            clean_assistant = self._extract_value(assistant_msg)
            logger.debug(f"[Memory Append Raw] Assistant: {assistant_msg}")
            logger.debug(f"[Memory Append Cleaned] Assistant: {clean_assistant}")
    
            if clean_assistant.strip():
                await self.append_to_memory(clean_user, clean_assistant)
                logger.info(f"[Memory Append] Stored → User: '{clean_user}' | Assistant: '{clean_assistant}'")
            else:
                logger.warning(f"[Memory Append] Skipped: assistant response was blank → {assistant_msg}")
    
            # 10. Return assistant response to UI
            return Message(text=clean_assistant, sender="Assistant")
    
        except ExceptionWithMessageError as e:
            logger.error(f"[ExceptionWithMessageError] {e}")
            return Message(text=str(e), sender="Assistant")
        except ValueError as e:
            logger.error(f"[ValueError] {e}")
            return Message(text=f"[Configuration Error] {str(e)}", sender="Assistant")
        except Exception as e:
            logger.exception("[NearAIAgentComponent] chat_response failed:")
            return Message(text=f"[❌ Exception in message_response] {str(e)}", sender="Assistant")


    def _extract_value(self, val):
        try:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                    return self._extract_value(val)
                except json.JSONDecodeError:
                    return val.strip()
            if isinstance(val, dict):
                if "text" in val:
                    return self._extract_value(val["text"])
                if "value" in val:
                    return self._extract_value(val["value"])
                if "content" in val:
                    return self._extract_value(val["content"])
            if isinstance(val, list) and len(val) > 0:
                return self._extract_value(val[0])
            return str(val)
        except Exception as e:
            logger.warning(f"[Extract Value] Failed to unwrap: {val} → {e}")
            return str(val)

    async def get_memory_data(self):
        """Load and sanitize memory, ensuring message types and content are valid."""
        try:
            # Step 1: Check if we're forcing a new thread
            memory = getattr(self, "memory", None)
            if getattr(self, "force_new_thread", False) and hasattr(memory, "thread_id"):
                logger.info("[Memory Init] Forcing thread reset")
                memory.thread_id = None
                await memory.ensure_agent_and_thread()
    
            # Step 2: Use cached memory if available
            if hasattr(self, "_shared_memory") and self._shared_memory:
                logger.debug("[NearAI] Using cached memory object.")
                memory = self._shared_memory
            else:
                # Step 3: If no memory wired directly, build from fallback inputs
                if memory is None:
                    logger.warning("[NearAI] Memory not wired — fallback to component inputs.")
                    memory_kwargs = {}
                    for component_input in self.memory_inputs:
                        attr_name = component_input.name
                        if hasattr(self, attr_name):
                            value = getattr(self, attr_name)
                            if value:
                                memory_kwargs[attr_name] = value
    
                    if memory_kwargs:
                        memory_component = MemoryComponent(**self.get_base_args()).set(**memory_kwargs)
                        memory = memory_component.build_message_history()
                        logger.debug("[NearAI] Built memory from parameters.")
    
            # Step 4: Ensure memory thread is initialized
            if memory and not getattr(memory, "thread_id", None):
                await memory.ensure_agent_and_thread()
    
            # Step 5: Cache and assign memory reference
            self._shared_memory = memory
            self.memory = memory
            self.thread_id = memory.thread_id
    
            # Step 6: Retrieve raw messages
            raw_messages = await memory.aget_messages()
    
            # Step 7: Sanitize content and reconstruct message types
            from langchain_core.messages import HumanMessage, AIMessage
    
            sanitized = []
            for m in raw_messages:
                try:
                    clean_content = self._extract_value(m.content)
                    if hasattr(m, "type"):
                        if m.type == "human":
                            sanitized.append(HumanMessage(content=clean_content))
                        elif m.type == "ai":
                            sanitized.append(AIMessage(content=clean_content))
                        else:
                            sanitized.append(m)
                    else:
                        sanitized.append(m)
                except Exception as e:
                    logger.warning(f"[Memory sanitize] Failed to clean message: {m} → {e}")
                    sanitized.append(m)
    
            logger.info(f"[NearAI] Retrieved {len(sanitized)} sanitized messages from memory")
            logger.debug(f"[NearAI] Sanitized memory contents: {sanitized}")
            return sanitized
    
        except Exception as e:
            logger.error(f"[get_memory_data error] {e}")
            return []


    async def append_to_memory(self, human: str, ai: str):
        try:
            memory = getattr(self, "_shared_memory", None)
            if memory is None or not hasattr(memory, "aadd_message"):
                logger.warning("[Memory append] Shared memory not found or invalid.")
                return
    
            from langchain_core.messages import HumanMessage, AIMessage
    
            # Flatten/unwrap content
            clean_human = self._extract_value(human)
            clean_ai = self._extract_value(ai)
    
            logger.debug(f"[Memory append] Cleaned Human: {clean_human}")
            logger.debug(f"[Memory append] Cleaned AI: {clean_ai}")
    
            await memory.aadd_message(HumanMessage(content=clean_human))
            await memory.aadd_message(AIMessage(content=clean_ai))
            logger.info("[NearAI] Messages appended cleanly to memory.")
        except Exception as e:
            logger.error(f"[Memory append error] {e}")


