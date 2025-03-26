import json

import requests
from pydantic import SecretStr

from langflow.base.vectorstores.model import LCVectorStoreComponent
from langflow.custom import Component
from langflow.helpers import docs_to_data
from langflow.inputs import FloatInput, NestedDictInput
from langflow.io import (
    BoolInput,
    DropdownInput,
    IntInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data
from langflow.template import Output


class NearVectorStoreComponent(Component):
    display_name: str = "NEAR Vector DB"
    description: str = "Ingest and search data from NEAR Vector DB"
    name = "NearVectorDB"
    icon: str = "NearAI"

    # Default values
    nearai_api_base = "https://api.near.ai/v1"

    # Store default credentials
    default_credentials = ""
    data_path = "~/.langflow/data"

    @classmethod
    def get_default_credentials_api_key(cls):
        """Generate a API key for model fetching during initialization."""
        if not (cls.default_credentials):
            return None

        authorization_object = SecretStr(cls.near_credentials).get_secret_value()
        return json.dumps(authorization_object["auth"])

    # Define inputs with default models
    inputs = [
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        StrInput(
            name="nearai_api_base",
            display_name="NearAI API Base",
            advanced=True,
            info="The base URL of the NearAI API. Defaults to https://api.near.ai/v1.",
            value=nearai_api_base,
        ),
        StrInput(
            name="data_folder_path",
            display_name="Data Folder Path",
            advanced=True,
            info="The path to the data folder.  Defaults to /data.",
            value=data_path,
        ),
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR credentials",
            info="Credential file info.",
            advanced=False,
            value=default_credentials,
            required=True,
        ),
        IntInput(
            name="max_chunk_size_tokens",
            display_name="Max Chunk Size Tokens",
            info="The maximum chunk token size.",
            advanced=True,
            value=800,
        ),
        IntInput(
            name="chunk_overlap_tokens",
            display_name="Chunk Overlap Tokens",
            info="The chunk overlap token value.",
            advanced=True,
            value=400,
        ),
        *LCVectorStoreComponent.inputs,
        IntInput(
            name="number_of_results",
            display_name="Number of Search Results",
            info="Number of search results to return.",
            advanced=True,
            value=4,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Search type to use",
            options=["Similarity", "Similarity with score threshold", "MMR (Max Marginal Relevance)"],
            value="Similarity",
            advanced=True,
        ),
        FloatInput(
            name="search_score_threshold",
            display_name="Search Score Threshold",
            info="Minimum similarity score threshold for search results. "
            "(when using 'Similarity with score threshold')",
            value=0,
            advanced=True,
        ),
        NestedDictInput(
            name="advanced_search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
        ),
        BoolInput(
            name="autodetect_collection",
            display_name="Autodetect Collection",
            info="Boolean flag to determine whether to autodetect the collection.",
            advanced=True,
            value=True,
        ),
        StrInput(
            name="content_field",
            display_name="Content Field",
            info="Field to use as the text content field for the vector store.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Vector Store Data",
            name="vector_store_data",
            info="The vector store data.",
            method="build_store",
        ),
    ]

    def __init__(self, **data):
        """Initialize the component and trigger model fetch if credentials are provided."""
        super().__init__(**data)

        # Set default credentials at class level if provided in instance
        if hasattr(self, "near_credentials") and self.near_credentials:
            self.__class__.default_credentials = self.default_credentials

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

    def build_store(self) -> Data:
        """Construct the vector store."""
        api_key = self.get_credentials_api_key()

        # Construct the API endpoint URL to get the Vector Store
        vector_store_endpoint = f"{self.nearai_api_base}/vector_stores/{Output.id}/search"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        vector_results = requests.post(
            vector_store_endpoint, headers=headers, json={"query": self.content_field}, timeout=10
        )
        vector_results.raise_for_status()
        return vector_results.json()

    def _get_exception_message(self, e: Exception):
        """Extracts meaningful messages from OpenAI errors."""
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            return e.body.get("message", "Unknown OpenAI error")
        return None

        search_type_mapping = {
            "Similarity with score threshold": "similarity_score_threshold",
            "MMR (Max Marginal Relevance)": "mmr",
        }

        return search_type_mapping.get(self.search_type, "similarity")

    def _build_search_args(self):
        query = self.search_query if isinstance(self.search_query, str) and self.search_query.strip() else None

        if query:
            args = {
                "query": query,
                "search_type": self._map_search_type(),
                "k": self.number_of_results,
                "score_threshold": self.search_score_threshold,
            }
        elif self.advanced_search_filter:
            args = {
                "n": self.number_of_results,
            }
        else:
            return {}

        filter_arg = self.advanced_search_filter or {}
        if filter_arg:
            args["filter"] = filter_arg

        return args

    def search_documents(self, vector_store=None) -> list[Data]:
        vector_store = vector_store or self.build_vector_store()

        self.log(f"Search input: {self.search_query}")
        self.log(f"Search type: {self.search_type}")
        self.log(f"Number of results: {self.number_of_results}")

        try:
            search_args = self._build_search_args()
        except Exception as e:
            msg = f"Error in ._build_search_args: {e}"
            raise ValueError(msg) from e

        if not search_args:
            self.log("No search input or filters provided. Skipping search.")
            return []

        docs = []
        search_method = "search" if "query" in search_args else "metadata_search"

        try:
            self.log(f"Calling vector_store.{search_method} with args: {search_args}")
            docs = getattr(vector_store, search_method)(**search_args)
        except Exception as e:
            msg = f"Error performing {search_method}: {e}"
            raise ValueError(msg) from e

        self.log(f"Retrieved documents: {len(docs)}")

        data = docs_to_data(docs)
        self.log(f"Converted documents to data: {len(data)}")
        self.status = data

        return data

    def get_retriever_kwargs(self):
        search_args = self._build_search_args()

        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
