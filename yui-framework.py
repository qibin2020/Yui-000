"""
title: Yui-000
author: Xuliang (xuliangz@sjtu.edu.cn)
description: OpenWebUI pipe function for Yui-000
version: 0.0.0
licence: MIT
"""

import logging
import io, sys, os
import json
import httpx
import re
import requests
from typing import AsyncGenerator, Callable, Awaitable, Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel("DEBUG")

class LookingGlass(logging.Handler):
    def emit(self, record):
        yield f"<log_assistant_core>\n```log\n{self.format(record)}\n```\n<log_assistant_core>"

glass_handler = LookingGlass()
glass_handler.setLevel(logging.ERROR)
glass_handler.setFormatter(logging.Formatter("%(message)s"))  # Customize as needed

log.addHandler(glass_handler)

class Pipe:
    class Valves(BaseModel):
        """
        Configuration for the pipe
        """
        MODEL_API_BASE_URL: str = Field(
            default=os.getenv("YUI_MODEL_API_BASE_URL", "https://aiapi001.ihep.ac.cn/apiv2"),
            description="语言模型API的基础请求地址",
        )
        MODEL_API_KEY: str = Field(
            default=os.getenv("YUI_MODEL_API_KEY", "api key here"),
            description="用于身份验证的API密钥",
        )
        TASK_MODEL_API_BASE_URL: str = Field(
            default=os.getenv("YUI_TASK_MODEL_API_BASE_URL", "https://aiapi001.ihep.ac.cn/apiv2"),
            description="语言模型API的基础请求地址",
        )
        TASK_MODEL_API_KEY: str = Field(
            default=os.getenv("YUI_TASK_MODEL_API_KEY", "api key here"),
            description="用于身份验证的API密钥",
        )
        BASE_MODEL: str = Field(
            default=os.getenv("YUI_BASE_MODEL", "deepseek-ai/deepseek-r1:671b"),
            description="对话的模型名称",
        )
        TASK_MODEL: str = Field(
            default=os.getenv("YUI_TASK_MODEL", "deepseek-ai/deepseek-v3:671b"),
            description="用于提取搜索提示词等的模型名称",
        )
        VISION_MODEL: str = Field(
            default=os.getenv("YUI_VISION_MODEL", "ark/doubao-vision-pro"),
            description="用于解析图片内容",
        )
        EMBEDDING_API_BASE_URL: str = Field(
            default=os.getenv("YUI_EMBEDDING_API_BASE_URL", "http://127.0.0.1:11434/v1"),
            description="文本嵌入API的基础请求地址",
        )
        EMBEDDING_API_KEY: str = Field(
            default=os.getenv("YUI_EMBEDDING_API_KEY", "api key here"),
            description="用于身份验证的API密钥",
        )
        EMBEDDING_MODEL: str = Field(
            default=os.getenv("YUI_EMBEDDING_MODEL", "bge-m3:latest"),
            description="用于获取文本嵌入的模型名称",
        )
        REALTIME_RAG: bool = Field(
            default=bool(int(os.getenv("YUI_REALTIME_RAG", "1"))),
            description="Realtime seaching Vector DB",
        )
        GENERATE_KEYWORDS_FROM_MODEL: bool = Field(
            default=bool(int(os.getenv("YUI_GENERATE_KEYWORDS_FROM_MODEL", "1"))),
            description="Generate keywords from model",
        )
        RAG_COLLECTION_NAMES: str = Field(
            default=os.getenv("YUI_RAG_COLLECTION_NAMES", "Yui-000-Source, Open WebUI Backend"),
        )
        EMBEDDING_BATCH_SIZE: int = Field(
            default=int(os.getenv("YUI_EMBEDDING_BATCH_SIZE", "2000")),
            description="Batch size for RAG",
        )
        REALTIME_IO: bool = Field(
            default=bool(int(os.getenv("YUI_REALTIME_IO", "1"))),
            description="Realtime Interact with environment and sense environment change",
        )
        USE_DARKSHINE_GUIDE: bool = Field(
            default=bool(int(os.getenv("YUI_USE_DARKSHINE_GUIDE", "1"))),
            title="Use DarkSHINE Guide",
        )
        USE_BESIII_GUIDE: bool = Field(
            default=bool(int(os.getenv("YUI_USE_BESIII_GUIDE", "0"))),
            title="Use BESIII Guide",
        )
        USE_CODE_INTERFACE: bool = Field(
            default=bool(int(os.getenv("YUI_USE_CODE_INTERFACE", "1"))),
        )
        USE_WEB_SEARCH: bool = Field(
            default=bool(int(os.getenv("YUI_USE_WEB_SEARCH", "1"))),
        )
        USE_MAPPING: bool = Field(
            default=bool(int(os.getenv("YUI_USE_MAPPING", "0"))),
        )
        CODE_WORKER_NAME: str = Field(
            default=os.getenv("YUI_CODE_WORKER_NAME", "xuliang/code-worker-v2"),
            description="Code worker model name",
        )
        CODE_WORKER_BASE_URL: str = Field(
            default=os.getenv("YUI_CODE_WORKER_BASE_URL", "http://localhost:42899/apiv2"),
            description="Code worker API base URL",
        )
        FILE_HOLDER_URL: str = Field(
            default=os.getenv("YUI_FILE_HOLDER_URL", "http://cheshire-cache:5678"),
            description="File Holdr URL",
        )
        GOOGLE_PSE_API_KEY: str = Field(
            default=os.getenv("YUI_GOOGLE_PSE_API_KEY", "api key here"),
            title="Google PSE API Key",
        )
        GOOGLE_PSE_ENGINE_ID: str = Field(
            default=os.getenv("YUI_GOOGLE_PSE_ENGINE_ID", "id here"),
            title="Google PSE Engine ID",
        )
        MAX_LOOP: int = Field(
            default=int(os.getenv("YUI_MAX_LOOP", "20")),
            description="Prevent dead loop, 0 for unlimited.",
        )
        THIRD_PARTY_BASE_PATH: str = Field(
            default=os.getenv("YUI_THIRD_PARTY_BASE_PATH", "~/third_party"),
            title="Base path to third-party utils, tools, so on.",
        )

    def __init__(self):
        # Shared Configs
        self.model_id = "Yui-000"
        self.valves = self.Valves()
        self.data_prefix = "data: "

    def pipes(self):
        return [
            {
                "id": self.model_id,
                "name": self.model_id,
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        # 验证配置
        if not self.valves.MODEL_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return
        # Append the thrid party (to easily import in body)
        sys.path.append(os.path.expanduser(self.valves.THIRD_PARTY_BASE_PATH))
        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.MODEL_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            messages = body["messages"]
            assistant_code_block = ""

            log.info("Extracting assistant code block...")
            for i in range(len(messages) - 1, -1, -1):
                content = messages[i]["content"]
                content_text = content
                if isinstance(content, List):
                    for c in content:
                        if c.get("type", "") == "text":
                            content_text = c.get("text", "")
                            break
                core = self.find_assistant_core(content_text)
                if core:
                    if not assistant_code_block:
                        log.info("Found last assistant code block")
                        assistant_code_block = core["content"]

                    if core["attributes"].get("hidden") == "true":
                        log.info("Removing hidden code block from meesage")
                        pattern = r"<(overwrite_assistant_core)(\s+[^>]*)?>(.*?)</\1>"
                        if isinstance(content, List):
                            for c in messages[i]["content"]:
                                if c.get("type", "") == "text":
                                    c["text"] = re.sub(pattern, '', c.get("text", ""), flags=re.DOTALL)
                                    break
                        else:
                            messages[i]["content"] = re.sub(pattern, '', messages[i]["content"], flags=re.DOTALL)

            if not assistant_code_block:
                log.warning("Fallback to load from default_interface.py")
                default_interface_path = os.path.expanduser(f"{self.valves.THIRD_PARTY_BASE_PATH}/assistant_utils/default_interface.py")
                if os.path.exists(default_interface_path):
                    with open(default_interface_path, 'r') as file:
                        assistant_code_block = file.read()
                else:
                    yield "\nError: Default interface file not found"
                    return

            # Add this after extracting assistant_code_block but before executing it
            if assistant_code_block:
                # Apply any edit blocks if they exist
                for i in range(len(messages)):
                    content = messages[i]["content"]
                    content_text = content
                    if isinstance(content, List):
                        for c in content:
                            if c.get("type", "") == "text":
                                content_text = c.get("text", "")
                                break
                    # Apply edit blocks to the assistant code
                    assistant_code_block = self.find_editblock_assistant_core(content_text, assistant_code_block)

            log.info("Evaluating assistant code block")
            namespace = globals().copy()
            # Execute the extracted code within the local_vars context
            exec(assistant_code_block, namespace)
            if "Assistant" not in namespace:
                yield "\nError: Assistant class not found in the assistant interface code"
                return

            Assistant = namespace['Assistant']
            agent = Assistant(self.valves)
            if not agent:
                yield "Error: Failed to create Assistant instance"
                return

            if not callable(agent.interface):
                yield "\nError: No callable function interface() found in the assistant source"
                return

            # Call the interface function with the provided arguments
            result = agent.interface(body, __event_emitter__)
            if isinstance(result, AsyncGenerator):
                async for item in result:
                    yield item
            else:
                yield result

        except Exception as e:
            log.error(f"Yui-Framework-Error: {e}", exc_info=True)
            yield self._format_error("Yui-Framework-Error", str(e))
        
    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"{status_code}: {err_msg}"}, ensure_ascii=False
        )

    def strip_triple_backtick(self, text: str) -> str:
        """
        Strips triple backticks from the text.
        """
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            # Remove the first line and the last line (markdown code block)
            lines = text.splitlines()
            if len(lines) > 1:
                lines = lines[1:-1]
            text = "\n".join(lines)
        return text

    def find_assistant_core(self, content):
        log.info("Finding assistant core...")
        # Define the regex pattern to match the XML tags
        pattern = re.compile(
            r"<(overwrite_assistant_core)(\s+[^>]*)?>(.*?)</\1>",
            re.DOTALL | re.MULTILINE,
        )

        # Find all matches in the content
        matches = pattern.findall(content)

        # If no matches found, return None
        if not matches:
            return None

        for match in matches:
            log.info(f"Match found: {len(matches)}")
            # Extract the tag name, attributes, and content
            tag_name = match[0]
            attributes_str = match[1]
            tag_content = self.strip_triple_backtick(match[2])

            # Extract attributes into a dictionary
            attributes = {}
            for attr in attributes_str.split():
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    value = value.strip("\"'")
                    attributes[key] = value
            
            return {
                "name": tag_name,
                "attributes": attributes,
                "content": tag_content,
            }
        
        return None


    def find_editblock_assistant_core(self, content, assistant_code_block):
        """
        Finds <editblock_assistant_core> tags and applies edit blocks to the assistant_code_block
        """
        log.info("Finding editblock assistant core...")
        # Define the regex pattern to match the XML tags
        pattern = re.compile(
            r"<(editblock_assistant_core)(\s+[^>]*)?>(.*?)</\1>",
            re.DOTALL | re.MULTILINE,
        )

        # Find all matches in the content
        matches = pattern.findall(content)

        # If no matches found, return the original assistant_code_block
        if not matches:
            return assistant_code_block

        # Process each match
        for match in matches:
            log.info("EditBlock match found")
            tag_content = match[2]

            # Parse for edit blocks with search/replace format
            search_replace_pattern = re.compile(
                r'<{5,9} SEARCH\s*\n(.*?)\n={5,9}\s*\n(.*?)\n>{5,9} REPLACE',
                re.DOTALL
            )
            edit_blocks = search_replace_pattern.findall(tag_content)

            for original, updated in edit_blocks:
                try:
                    # Try to apply the edit directly if exact match exists
                    if original in assistant_code_block:
                        assistant_code_block = assistant_code_block.replace(original, updated)
                        log.info("Applied exact match edit")
                    else:
                        # Try fuzzy matching if exact match fails
                        import difflib
                        lines = assistant_code_block.splitlines(True)
                        original_lines = original.splitlines(True)

                        for i in range(len(lines) - len(original_lines) + 1):
                            chunk = ''.join(lines[i:i + len(original_lines)])
                            similarity = difflib.SequenceMatcher(None, chunk, original).ratio()
                            if similarity > 0.8:  # 80% similarity threshold
                                updated_lines = lines[:i] + updated.splitlines(True) + lines[i + len(original_lines):]
                                assistant_code_block = ''.join(updated_lines)
                                log.info(f"Applied fuzzy match edit (similarity: {similarity:.2f})")
                                break
                        else:
                            log.warning("Could not find matching code segment to edit")
                except Exception as e:
                    log.error(f"Error applying edit: {e}")

        return assistant_code_block
