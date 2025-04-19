from langflow.custom import Component
from langchain_openai import ChatOpenAI
from langflow.base.agents.events import ExceptionWithMessageError
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
from typing import Union
from near_ai_agent.environment import NearAIEnvironment
from near_ai_agent.agent import NearAIAgent


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class NearAIAgentComponent(ToolCallingAgentComponent):
    display_name = "NearAI Agent"
    description = "Multi-tool agent using NEAR AI SDK with Langflow tool input support."
    icon = "NearAI"
    name = "NearAIAgent"

    _model_display_map = {}
    default_credentials = ""
    nearai_api_base = "https://api.near.ai/v1"
    _openai_models = []

    memory_inputs = [set_advanced_true(inp) for inp in MemoryComponent().inputs]

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
        *ToolCallingAgentComponent._base_inputs,
        *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Add Current Date Tool",
            value=True,
            advanced=True,
        ),
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            info="JSON containing API key",
            value=default_credentials,
            required=True,
        ),
    ]

    outputs = [
        Output(name="value", display_name="Chat (Message)", method="chat_response")
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
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"[NEARAI ERROR] Response: {response.text}")
                response.raise_for_status()
            return response.json()

    async def chat_response(self) -> Message:
        try:
            try:
                api_key = self.get_credentials_api_key()
            except Exception as e:
                return Message(text=f"[Credential error] {e}", sender="Assistant")
    
            if not self.__class__._openai_models:
                self.__class__.fetch_openai_models(api_key=api_key, base_url=self.nearai_api_base)
    
            model_name = self.model_name
            if not self.__class__._model_display_map:
                self.__class__._model_display_map = {
                    self.__class__.format_model_display_name(m): m for m in self.__class__._openai_models
                }
    
            resolved = self.__class__._model_display_map.get(model_name, model_name)
            base_url = self.nearai_api_base
    
            tool_registry = {}
            openai_tool_schemas = []
    
            for tool in getattr(self, "tools", []):
                try:
                    tool_func = getattr(tool, "func", tool)
                    tool_func.__name__ = tool.name
                    tool_func.__doc__ = tool.description or f"{tool.name} tool"
                    tool_registry[tool.name] = tool_func
    
                    args_schema = getattr(tool, "args_schema", None)
                    if args_schema and hasattr(args_schema, "__annotations__"):
                        openai_tool_schemas.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or f"{tool.name} tool",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        k: {"type": "string", "description": f"Argument: {k}"}
                                        for k in args_schema.__annotations__
                                    },
                                    "required": list(args_schema.__annotations__)
                                }
                            }
                        })
                except Exception as e:
                    logger.warning(f"[Tool setup] {e}")
    
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.input_value}
            ]
    
            try:
                response = await self.run_near_ai_completion(api_key, base_url, resolved, messages, openai_tool_schemas)
            except Exception as e:
                return Message(text=f"[NearAI call failed] {e}", sender="Assistant")
    
            if not response or "choices" not in response or not response["choices"]:
                return Message(text="[NEARAI ERROR] No response generated.", sender="Assistant")
    
            message = response["choices"][0].get("message")
            if not message:
                return Message(text="[NEARAI ERROR] Response contained no message.", sender="Assistant")
    
            tool_calls = message.get("tool_calls")

            if tool_calls and isinstance(tool_calls, list):
                for call in tool_calls:
                    function = call.get("function", {})
                    name = function.get("name")
                    args_json = function.get("arguments", "{}")
            
                    if not name:
                        logger.warning(f"[NEARAI] Skipping tool call with missing name: {call}")
                        continue
            
                    try:
                        args = json.loads(args_json)
                    except Exception as e:
                        logger.warning(f"[NEARAI] Failed to parse tool args for {name}: {e}")
                        args = {}
            
                    tool_func = tool_registry.get(name)
                    if tool_func:
                        try:
                            result = tool_func(**args)
                        except Exception as e:
                            result = f"[Tool execution error for {name}] {e}"
                            logger.warning(result)
                    else:
                        result = f"[Unknown tool: {name}]"
                        logger.warning(result)
            
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", "unknown_id"),
                        "name": name,
                        "content": str(result)
                    })
            else:
                logger.warning("[NEARAI] No tool_calls found or malformed tool_calls list.")


                followup_response = await self.run_near_ai_completion(api_key, base_url, resolved, messages, openai_tool_schemas)
                if not followup_response or "choices" not in followup_response or not followup_response["choices"]:
                    return Message(text="[NEARAI ERROR] No assistant reply received.", sender="Assistant")
    
                logger.info("[NEARAI] Final Response:")
                logger.info(json.dumps(followup_response, indent=2))
                followup_message = followup_response["choices"][0].get("message", {})
                final_content = followup_message.get("content") or "[⚠️ Assistant reply was empty]"
                return Message(text=final_content, sender="Assistant")
                
            final_content = message.get("content") or "[No response from assistant]"
            logger.info(f"[NEARAI Final Content] {final_content}")
            return Message(text=final_content, sender="Assistant")
            
        except Exception as e:
            logger.exception("[NearAIAgentComponent] chat_response failed:")
            return Message(text=f"[❌ Exception in chat_response] {str(e)}", sender="AI")

    async def get_memory_data(self):
        memory_kwargs = {
            inp.name: getattr(self, inp.name)
            for inp in self.memory_inputs
            if getattr(self, inp.name, None)
        }
        return await MemoryComponent(**self.get_base_args()).set(**memory_kwargs).retrieve_messages()
