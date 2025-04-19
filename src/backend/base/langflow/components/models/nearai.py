import json
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import SecretStr
from typing_extensions import override

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    BoolInput, DictInput, DropdownInput, IntInput, SecretStrInput, SliderInput, StrInput
)


class NearAIModelComponent(LCModelComponent):
    display_name = "NearAI"
    description = "Generates text using NearAI LLMs."
    icon = "NearAI"
    name = "NearAIModel"

    nearai_api_base = "https://api.near.ai/v1"

    _openai_models = []
    _model_display_map = {}
    default_credentials = ""

    
    @override
    def update_build_config(cls, build_config: dict, field_value: str, field_name: str | None = None):
        try:
            credentials_str = SecretStr(cls.near_credentials).get_secret_value()
            credentials_json = json.loads(credentials_str)
            api_key = credentials_json["auth"]
        except Exception as e:
            print(f"[update_build_config] Failed to get api_key: {e}")
            api_key = None
    
        if field_name in {"nearai_api_base", "model_name"}:
            models = cls.fetch_openai_models(api_key)
            cls._model_display_map = {
                cls.format_model_display_name(m): m for m in models
            }
            cls._openai_models = models  # ✅ Ensure this is updated!
            build_config["model_name"]["options"] = list(cls._model_display_map.keys())
    
        return build_config


    @classmethod
    def format_model_display_name(cls, model_name: str) -> str:
        if "::" in model_name:
            provider, full_path = model_name.split("::", 1)
            short_name = full_path.split("/")[-1]
            return f"{provider} - {short_name}"
        return model_name

    @classmethod
    def fetch_openai_models(cls, api_key=None, base_url=None):
        try:
            if not api_key:
                print("[fetch_openai_models] No API key provided.")
                return cls._openai_models
    
            if not base_url:
                base_url = cls.nearai_api_base
    
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.models.list()
    
            model_ids = [model.id for model in response.data]
    
            if model_ids:
                cls._openai_models = model_ids
                cls._model_display_map = {
                    cls.format_model_display_name(m): m for m in model_ids
                }
    
            return cls._openai_models
        except Exception as e:
            print(f"[fetch_openai_models] Error: {e}")
            return cls._openai_models


    
    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="Maximum tokens to generate.",
            range_spec=RangeSpec(min=0, max=128000)
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional kwargs for the model.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="Force JSON output.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name 📏",
            advanced=False,
            options=[],
            real_time_refresh=True,
            refresh_button=True
        ),
        StrInput(
            name="nearai_api_base",
            display_name="NearAI API Base",
            advanced=True,
            info="Base URL of the NearAI API.",
            value=nearai_api_base,
        ),
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR Credentials",
            info="Credential JSON (must include `auth.api_key`).",
            advanced=False,
            required=True,
            value=default_credentials,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01)
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            advanced=True,
            info="Controls reproducibility.",
            value=1,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            advanced=True,
            info="Retry attempts for generation.",
            value=5,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            advanced=True,
            info="Request timeout in seconds.",
            value=700,
        ),
    ]


    def get_credentials_api_key(self):
        if not hasattr(self, "near_credentials") or not self.near_credentials:
            return None
        try:
            credentials_str = SecretStr(self.near_credentials).get_secret_value()
            credentials_json = json.loads(credentials_str)
    
            # ✅ Correct extraction o
            return json.dumps(credentials_json["auth"])
        except Exception as e:
            print(f"[get_credentials_api_key] Failed: {e}")
            return None

    def build_model(self) -> LanguageModel:
        api_key = self.get_credentials_api_key()
    
        # Ensure we have models
        if not self.__class__._openai_models:
            print("[build_model] _openai_models is empty — calling fetch_openai_models() manually...")
            self.__class__.fetch_openai_models(api_key=api_key, base_url=self.nearai_api_base)
    
        model_name = self.model_name
        print(f"[build_model] self.model_name (dropdown selected): {model_name}")
    
        if not self.__class__._model_display_map:
            print("[build_model] _model_display_map is empty. Rebuilding from _openai_models...")
            print(f"[build_model] _openai_models: {self.__class__._openai_models}")
            self.__class__._model_display_map = {
                self.__class__.format_model_display_name(m): m
                for m in self.__class__._openai_models
            }
    
        print(f"[build_model] available model keys: {list(self.__class__._model_display_map.keys())}")
    
        resolved = self.__class__._model_display_map.get(model_name, model_name)
        print(f"[build_model] resolved model_name: {resolved}")
        if resolved == model_name:
            print(f"[build_model] WARNING: '{model_name}' was not mapped — using it directly.")
    
        return ChatOpenAI(
            model=resolved,
            api_key=api_key,
            base_url=self.nearai_api_base,
            temperature=self.temperature or 0.1,
            max_tokens=self.max_tokens or None,
            model_kwargs=self.model_kwargs or {},
            max_retries=self.max_retries,
            request_timeout=self.timeout,
        ).bind(response_format={"type": "json_object"}) if self.json_mode else ChatOpenAI(
            model=resolved,
            api_key=api_key,
            base_url=self.nearai_api_base,
            temperature=self.temperature or 0.1,
            max_tokens=self.max_tokens or None,
            model_kwargs=self.model_kwargs or {},
            max_retries=self.max_retries,
            request_timeout=self.timeout,
        )



    def _get_exception_message(self, e: Exception):
        try:
            from openai import BadRequestError
            if isinstance(e, BadRequestError):
                return e.body.get("message", "Unknown OpenAI error")
        except ImportError:
            return None
        return str(e)