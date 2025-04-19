import os
import json
import requests
import openai
from pydantic import SecretStr
from langchain.schema import Document
from langflow.base.vectorstores.model import LCVectorStoreComponent
from langflow.custom import Component
from langflow.inputs import (
    SecretStrInput,
    StrInput,
    IntInput,
    FloatInput,
    NestedDictInput,
    DropdownInput,
    BoolInput,
)
from langflow.io import Output
from langflow.schema import Data
from langflow.helpers import docs_to_data


class NearVectorStoreComponent(Component):
    display_name: str = "NEAR Vector DB"
    description: str = "Upload files to NEAR AI and optionally perform a search"
    name = "NearVectorDB"
    icon: str = "NearAI"

    nearai_api_base = "https://api.near.ai/v1"
    default_credentials = ""
    vector_store_id: str = None

    inputs = [
        SecretStrInput(
            name="near_credentials",
            display_name="NEAR credentials",
            info="NEAR AI credentials as JSON",
            required=True,
        ),
        StrInput(
            name="store_name",
            display_name="Vector Store Name",
            value="langflow-store",
            required=False,
        ),
        StrInput(
            name="provided_vector_store_id",
            display_name="Vector Store ID",
            required=False,
            advanced=False,
            info="If provided, will use this existing vector store instead of creating a new one.",
        ),
        BoolInput(
            name="clear_existing_store",
            display_name="Clear Existing Store",
            value=False,
            advanced=False,
            info="If true and a vector store ID is provided, will delete and recreate it.",
        ),
        StrInput(
            name="search_query",
            display_name="Search Query",
            required=False,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            value=4,
            advanced=True,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            options=["similarity", "similarity_score_threshold", "mmr"],
            value="similarity",
            advanced=True,
        ),
        *LCVectorStoreComponent.inputs,
        FloatInput(
            name="score_threshold",
            display_name="Score Threshold",
            value=0.0,
            advanced=True,
        ),
        NestedDictInput(
            name="search_filter",
            display_name="Search Metadata Filter",
            advanced=True,
        ),
        BoolInput(
            name="run_search",
            display_name="Run Search",
            value=False,
            advanced=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Vector Store Info",
            name="vector_store_data",
            info="Vector store ID and upload result.",
            method="build_store",
        ),
        Output(
            display_name="Vector Store ID",
            name="vector_store_id",
            info="Just the vector store ID.",
            method="get_vector_store_id",
        ),
        Output(
            display_name="Search Results",
            name="search_results",
            info="List of matching documents from the vector store.",
            method="search_documents",
        ),
    ]

    def __init__(self, **data):
        super().__init__(**data)

    def get_vector_store_id(self) -> str:
        if not self.vector_store_id:
            self.build_store()
        return self.vector_store_id

    def build_store(self) -> Data:
        provided_id = getattr(self, "provided_vector_store_id", "").strip()
        clear_existing = getattr(self, "clear_existing_store", False)
        store_name = getattr(self, "store_name", "langflow-store")
    
        credentials_str = SecretStr(self.near_credentials).get_secret_value()
        credentials = json.loads(credentials_str)
        auth = credentials.get("auth")
        if not auth:
            raise ValueError("Invalid NEAR credentials provided.")
        api_key = json.dumps(auth)
    
        headers_json = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
        # --- CASE 1: Run search only, skip ingestion ---
        if getattr(self, "run_search", False) and provided_id and not clear_existing:
            print(f"[SEARCH-ONLY] Skipping ingestion, using existing vector store: {provided_id}")
            self.vector_store_id = provided_id
            return {
                "vector_store_id": self.vector_store_id,
                "status": "search_only_used_existing_store",
                "store_name": store_name,
                "uploaded_file_ids": [],
            }
    
        # --- CASE 2: Clear files from existing store without deleting the store ---
        if provided_id and clear_existing:
            print(f"[CLEAR] Detaching all files from vector store: {provided_id}")
            self.vector_store_id = provided_id
    
            # Step 1: List attached files
            list_url = f"{self.nearai_api_base}/vector_stores/{provided_id}/files"
            list_response = requests.get(list_url, headers=headers_json)
    
            if list_response.status_code != 200:
                raise ValueError(f"[CLEAR ERROR] Failed to list files: {list_response.status_code} - {list_response.text}")
    
            list_data = list_response.json()
            file_list = list_data.get("data", [])
    
            print(f"[CLEAR] Found {len(file_list)} files to detach.")
    
            # Step 2: Detach each file
            for file_info in file_list:
                file_id = file_info.get("id")  # <- this is the correct field per API docs
                if not file_id:
                    print(f"[WARN] Skipping invalid file info: {file_info}")
                    continue
    
                detach_url = f"{self.nearai_api_base}/vector_stores/{provided_id}/files/{file_id}"
                detach_response = requests.delete(detach_url, headers=headers_json)
    
                if detach_response.status_code not in (200, 204):
                    print(f"[WARN] Failed to detach file {file_id}: {detach_response.status_code} - {detach_response.text}")
                else:
                    print(f"[DETACH] Successfully detached file {file_id}")
    
        # --- CASE 3: Provided ID but not clearing, validate and fallback to create ---
        elif provided_id:
            print(f"[USE EXISTING] Validating existing vector store ID: {provided_id}")
            check_url = f"{self.nearai_api_base}/vector_stores/{provided_id}"
            check_response = requests.get(check_url, headers=headers_json)
    
            if check_response.status_code == 200:
                self.vector_store_id = provided_id
                print(f"[USE EXISTING] Using verified vector store ID: {self.vector_store_id}")
            elif check_response.status_code == 404:
                print(f"[NOT FOUND] Vector store ID {provided_id} not found. Creating new store: {store_name}")
                create_url = f"{self.nearai_api_base}/vector_stores"
                create_response = requests.post(create_url, headers=headers_json, json={"name": store_name})
                create_response.raise_for_status()
                self.vector_store_id = create_response.json()["id"]
                print(f"[CREATE] Created new vector store ID: {self.vector_store_id}")
            else:
                raise ValueError(f"Error validating vector store {provided_id}: {check_response.status_code} - {check_response.text}")
    
        # --- CASE 4: No ID provided, create new store ---
        elif not provided_id:
            print(f"[CREATE] No vector store ID provided, creating a new one.")
            create_url = f"{self.nearai_api_base}/vector_stores"
            create_response = requests.post(create_url, headers=headers_json, json={"name": store_name})
            create_response.raise_for_status()
            self.vector_store_id = create_response.json()["id"]
            print(f"[CREATE] Created vector store ID: {self.vector_store_id}")
    
        # --- Check if ingestion data is present ---
        if not hasattr(self, "ingest_data") or not self.ingest_data:
            raise ValueError("No data provided for ingestion.")
    
        # --- Upload and attach files ---
        file_ids = []
        for i, data_obj in enumerate(self.ingest_data):
            text = data_obj.data.get("text", "")
            metadata = data_obj.data.get("metadata", {})
            if not text.strip():
                print(f"[SKIP] Chunk {i} is empty.")
                continue
    
            file_type = metadata.get("file_type", "txt").lower()
            ext = {
                "pdf": ".pdf",
                "md": ".md",
                "markdown": ".md"
            }.get(file_type, ".txt")
    
            mime_type = {
                ".pdf": "application/pdf",
                ".md": "text/markdown",
                ".txt": "text/plain"
            }.get(ext, "text/plain")
    
            print(f"[UPLOAD] Uploading file_{i}{ext} to /v1/files as {mime_type}")
    
            files = {
                "file": (f"file_{i}{ext}", text.encode("utf-8"), mime_type),
                "purpose": (None, "assistants")
            }
    
            upload_response = requests.post(
                f"{self.nearai_api_base}/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files
            )
    
            if upload_response.status_code != 200:
                print(f"[ERROR] File upload failed for file_{i}{ext}")
                print(f"[ERROR] Status: {upload_response.status_code}")
                print(f"[ERROR] Response: {upload_response.text}")
                continue
    
            file_id = upload_response.json().get("id")
            if not file_id:
                print(f"[ERROR] No file_id returned for file_{i}{ext}")
                continue
    
            file_ids.append(file_id)
            print(f"[UPLOAD] Uploaded and received file_id: {file_id}")
    
            attach_url = f"{self.nearai_api_base}/vector_stores/{self.vector_store_id}/files"
            attach_response = requests.post(
                attach_url,
                headers=headers_json,
                json={"file_id": file_id}
            )
    
            if attach_response.status_code != 200:
                print(f"[ERROR] Failed to attach file_id {file_id} to vector store.")
                print(f"[ERROR] Response: {attach_response.text}")
                continue
    
            print(f"[LINK] Linked file_id {file_id} to vector store {self.vector_store_id}")
    
        if not file_ids:
            raise ValueError("No files were successfully uploaded or linked.")
    
        return {
            "vector_store_id": self.vector_store_id,
            "status": "created_and_uploaded" if not provided_id else "uploaded_to_existing",
            "store_name": store_name,
            "uploaded_file_ids": file_ids
        }

    def search_documents(self) -> list[Data]:
        if not getattr(self, "run_search", False):
            print("[INFO] Search disabled (run_search=False). Skipping search.")
            return []
    
        # Set vector_store_id if provided
        provided_id = getattr(self, "provided_vector_store_id", "").strip()
        if provided_id:
            self.vector_store_id = provided_id
            print(f"[INFO] Using provided vector store ID: {self.vector_store_id}")
        elif not self.vector_store_id:
            print("[INFO] No vector store ID provided. Attempting to build a new one...")
            store_data = self.build_store()
            self.vector_store_id = store_data["vector_store_id"]
    
        if not self.vector_store_id:
            raise ValueError("No vector store ID available for search.")
    
        print(f"[SEARCH] Running search against vector_store_id = {self.vector_store_id}")
    
        # Prepare API key and client
        credentials_str = SecretStr(self.near_credentials).get_secret_value()
        credentials = json.loads(credentials_str)
        auth = credentials.get("auth")
        if not auth:
            raise ValueError("Missing 'auth' field in NEAR credentials.")
        api_key = json.dumps(auth)
    
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
        search_args = {
            "query": getattr(self, "search_query", ""),
            "search_type": getattr(self, "search_type", "similarity"),
            "k": getattr(self, "number_of_results", 4),
        }
    
        if self.score_threshold > 0:
            search_args["score_threshold"] = self.score_threshold
    
        if getattr(self, "search_filter", None):
            search_args["filter"] = self.search_filter
    
        print(f"[SEARCH] Search Payload:\n{json.dumps(search_args, indent=2)}")
    
        url = f"{self.nearai_api_base}/vector_stores/{self.vector_store_id}/search"
        response = requests.post(url, headers=headers, json=search_args)
    
        print(f"[SEARCH RESPONSE] Status: {response.status_code}")
        print(f"[SEARCH RESPONSE] Body:\n{response.text}")
    
        if response.status_code == 404:
            raise ValueError(f"Vector store not found at {url}. ID: {self.vector_store_id}")
    
        response.raise_for_status()
    
        resp_json = response.json()
        print(f"[SEARCH RESPONSE] JSON:\n{json.dumps(resp_json, indent=2)}")
        
        if isinstance(resp_json, list):
            docs = [
                Document(
                    page_content=item.get("chunk_text", ""),
                    metadata={
                        "file_id": item.get("file_id"),
                        "distance": item.get("distance"),
                    },
                )
                for item in resp_json
            ]
        elif isinstance(resp_json, dict) and "documents" in resp_json:
            docs = resp_json["documents"]  # Should already be Document-like
        else:
            print(f"[ERROR] Unexpected search response format: {resp_json}")
            raise ValueError("Unexpected format in NEAR AI search response.")
        
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        return Data(data={"text": combined_text})


