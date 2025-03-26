import json

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import BoolInput, DictInput, DropdownInput, IntInput, SecretStrInput, SliderInput, StrInput


class NearAIModelComponent(LCModelComponent):
    display_name = "NearAI"
    description = "Generates text using NearAI LLMs."
    icon = "NearAI"
    name = "NearAIModel"

    # Default values
    nearai_api_base = "https://api.near.ai/v1"
    vector_store_id = "vs_558181d6ee76400a8227f2bd"
    _openai_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]  # Default fallback models

    # Store default credentials for initial model fetching
    default_credentials = ""

    @classmethod
    def format_model_display_name(cls, model_name):
        """Format long model names for better display in dropdown.

        Example: fireworks::accounts/fireworks/models/mistral-small-24b-instruct-2501
        becomes: fireworks - mistral-small-24b-instruct-2501
        """
        if "::" in model_name:
            provider, full_path = model_name.split("::", 1)
            # Extract just the model name from the path
            model_short_name = full_path.split("/")[-1]
            return f"{provider} - {model_short_name}"
        return model_name

    @classmethod
    def get_default_credentials_api_key(cls):
        """Generate a API key for model fetching during initialization."""
        if not (cls.default_credentials):
            return None

        authorization_object = SecretStr(cls.near_credentials).get_secret_value()
        return json.dumps(authorization_object["auth"])

    @classmethod
    def fetch_openai_models(cls, api_key=None, base_url=None):
        """Fetch available models from OpenAI API and update class variable."""
        try:
            if not api_key:
                return cls._openai_models

            if not base_url:
                base_url = cls.nearai_api_base

            client = OpenAI(base_url=base_url, api_key=api_key)
            response = client.models.list()

            # Correctly parse the response
            model_names = [model.id for model in response.data]

            if model_names:
                cls._openai_models = model_names
                # Force update dropdown immediately
                cls.update_dropdown()
            else:
                return cls._openai_models

        except Exception:  # noqa: BLE001
            return cls._openai_models  # Return default models on error

    @classmethod
    def update_dropdown(cls):
        """Updates the dropdown options dynamically and returns updated inputs."""
        new_inputs = []

        # Store the mapping between display names and actual model IDs at class level
        cls._model_display_map = {}
        display_options = []

        # Create display versions of model names
        for model in cls._openai_models:
            if "::" in model:
                provider, full_path = model.split("::", 1)
                model_short_name = full_path.split("/")[-1]
                display_name = f"{provider} - {model_short_name}"
            else:
                display_name = model

            # Store both the display name and original model ID
            cls._model_display_map[display_name] = model
            display_options.append(display_name)

        for input_field in cls.inputs:
            if isinstance(input_field, DropdownInput) and input_field.name == "model_name":
                # Replace the model_name DropdownInput with updated options
                new_input = DropdownInput(
                    name="model_name",
                    display_name="Model Name 📏",
                    advanced=False,
                    options=display_options,  # Show formatted names in dropdown
                    value=display_options[0] if display_options else "gpt-4o",
                )
                new_inputs.append(new_input)
            else:
                new_inputs.append(input_field)

        cls.inputs = new_inputs
        return cls.inputs

    # Define inputs with default models
    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name 📏",
            advanced=False,
            options=_openai_models,
            value=_openai_models[0],
        ),
        StrInput(
            name="nearai_api_base",
            display_name="NearAI API Base",
            advanced=True,
            info="The base URL of the NearAI API. Defaults to https://api.near.ai/v1.",
            value=nearai_api_base,
        ),
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR credentials",
            info="Credential file info.",
            advanced=False,
            value=default_credentials,
            required=True,
        ),
        SliderInput(
            name="temperature", display_name="Temperature", value=0.1, range_spec=RangeSpec(min=0, max=1, step=0.01)
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="The maximum number of retries to make when generating.",
            advanced=True,
            value=5,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="The timeout for requests to OpenAI completion API.",
            advanced=True,
            value=700,
        ),
    ]

    def __init__(self, **data):
        """Initialize the component and trigger model fetch if credentials are provided."""
        super().__init__(**data)

        # Set default credentials at class level if provided in instance
        if hasattr(self, "near_credentials") and self.near_credentials:
            self.__class__.default_credentials = self.near_credentials

        # Try to fetch models if we have credentials
        self.refresh_models()

    def refresh_models(self):
        """Try to fetch models with current credentials."""
        credentials_key = self.get_credentials_api_key()

        if credentials_key:
            self.__class__.fetch_openai_models(api_key=credentials_key, base_url=self.nearai_api_base)

    def get_credentials_api_key(self):
        """Constructs the credentials API key using instance-level attributes."""
        # Check if we have the necessary credentials
        if not hasattr(self, "near_credentials") or not self.near_credentials:
            return None

        try:
            # Parse the credentials string as JSON
            credentials_str = SecretStr(self.near_credentials).get_secret_value()
            credentials_json = json.loads(credentials_str)

            # Extract the auth object
            if "auth" in credentials_json:
                return json.dumps(credentials_json["auth"])
            # Handle case where auth object doesn't exist in credentials
            return None  # noqa: TRY300
        except json.JSONDecodeError:
            # Handle case where credentials are not valid JSON
            return None
        except Exception:  # noqa: BLE001
            # Handle other potential errors
            return None

    def build_model(self) -> LanguageModel:
        """Construct the model."""
        # Try to refresh models with current credentials
        self.refresh_models()

        # api_key = self.get_api_key()
        api_key = self.get_credentials_api_key()

        # Get the actual model ID if we're using a display name
        model_name = self.model_name

        # Look up the actual model ID from the display mapping if it exists
        if hasattr(self.__class__, "_model_display_map") and model_name in self.__class__._model_display_map:
            model_name = self.__class__._model_display_map[model_name]

        output = ChatOpenAI(
            max_tokens=self.max_tokens or None,
            model_kwargs=self.model_kwargs or {},
            model=model_name,  # Use the actual model ID
            base_url=self.nearai_api_base,
            api_key=api_key,
            temperature=self.temperature if self.temperature is not None else 0.1,
            max_retries=self.max_retries,
            request_timeout=self.timeout,
        )

        if self.json_mode:
            output = output.bind(response_format={"type": "json_object"})

        return output

    def _get_exception_message(self, e: Exception):
        """Extracts meaningful messages from OpenAI errors."""
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            return e.body.get("message", "Unknown OpenAI error")
        return None


# Hook to refresh models on module import
try:
    # Try to fetch models using default API key if available
    default_credential_api_key = NearAIModelComponent.get_default_credentials_api_key()
    if default_credential_api_key:
        NearAIModelComponent.fetch_openai_models(api_key=default_credential_api_key)
except Exception:  # noqa: BLE001, S110
    pass
