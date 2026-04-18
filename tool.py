import os
from typing import Any, Callable

from dotenv import load_dotenv
from serpapi import SerpApiClient


class ToolExecutor:
    def __init__(self) -> None:
        self.tool_list: dict[str, dict[str, Any]] = {}

    def ToolRegister(self, tool_name: str, description: str, func: Callable) -> None:
        normalized_name = tool_name.strip()
        lookup_key = normalized_name.lower()

        if lookup_key in self.tool_list:
            raise ValueError(f"Tool '{normalized_name}' has already been registered")

        self.tool_list[lookup_key] = {
            "name": normalized_name,
            "description": description,
            "function": func,
        }
        print(f"{normalized_name} has been registered")

    def getTool(self, name: str) -> Callable | None:
        if not name:
            return None

        return self.tool_list.get(name.strip().lower(), {}).get("function")

    def getAvailableTools(self) -> str:
        return "\n".join(
            f"- {info['name']}: {info['description']}"
            for info in self.tool_list.values()
        )

# -------------------------- Tools -------------------------- 
def search(query: str) -> str:
    """基于 SerpAPI 搜索网站、商品或一般网页内容。"""

    load_dotenv(dotenv_path=".env")

    try:
        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            raise ValueError("Missing SERP_API_KEY")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

         # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

if __name__ == "__main__":
    toolExecutor = ToolExecutor()

    search_desc = "基于serpAPI搜索网站/商品等内容"
    toolExecutor.ToolRegister(
        tool_name="Search",
        description=search_desc,
        func=search,
    )
    # 注册检测
    print(toolExecutor.getAvailableTools())

    search_query = "英伟达最新的GPU是什么？"
    tool_name = "Search"

    func = toolExecutor.getTool(name=tool_name)
    if func:
        result = func(search_query)
        print(f"搜索结果:\n{result}")
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
