import re
from dataclasses import dataclass

from llm import HelloAgents
from prompt import REACT_PROMPT_TEMPLATE
from tool import ToolExecutor, search
from datetime import date

ACTION_PATTERN = re.compile(r"([A-Za-z_]\w*)\[(.*)\]", re.DOTALL)
FINISH_PATTERN = re.compile(r"Finish\[(.*)\]", re.DOTALL)


@dataclass
class ParsedStep:
    thought: str | None
    action: str


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgents,
        tool_executor: ToolExecutor,
        max_steps: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: list[str] = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            # 追踪
            current_step += 1
            print(f"这是第{current_step}步:")

            # 初始化prompt
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getAvailableTools(),
                history="\n".join(self.history) or "None",
                question=question,
                today = date.today()
            )
            response = self.llm_client.think(message=[{"role": "user", "content": prompt}])

            if not response:
                print("错误: LLM未能返回有效响应。")
                break
            
            # 解析当前一轮的llm回答
            parsed_step = self._parse_step(response)

            if parsed_step.thought:
                print(f"llm thought: {parsed_step.thought}")

            action = parsed_step.action
            if action.startswith("Finish["):
                final_answer = self._parse_finish(action)
                print(f"最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                raise ValueError(f"The {tool_name} is not in the tool list")

            print(f"tool: {tool_name}\n tool_input: {tool_input}")
            observation = tool_function(tool_input)
            print(f"Next_step: {observation}")

            # Observation only comes from the runtime, not from the model output.
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_step(self, text: str) -> ParsedStep:
        thought_lines: list[str] = []
        action: str | None = None
        state = "idle"
        lines = text.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue

            # thought
            if line.startswith("Thought:"):
                content = line[len("Thought:") :].strip()
                thought_lines = [content] if content else []
                state = "thought"
                index += 1
                continue

            # action
            if line.startswith("Action:"):
                content = self._collect_action_block(lines, index)
                if not content:
                    raise ValueError("Action line is empty")
                action = content
                break

            if state == "thought":
                if self._looks_like_protocol_label(line):
                    state = "idle"
                else:
                    thought_lines.append(line)

            index += 1

        if not action:
            raise ValueError("LLM output did not contain a valid Action line")

        thought = "\n".join(thought_lines).strip() or None
        return ParsedStep(thought=thought, action=action)

    def _parse_finish(self, action_text: str) -> str:
        match = FINISH_PATTERN.fullmatch(self._normalize_action_text(action_text))
        if not match:
            raise ValueError(f"Invalid Finish action format: {action_text}")
        return match.group(1).strip()

    def _parse_action(self, action_text: str) -> tuple[str, str]:
        normalized_action = self._normalize_action_text(action_text)
        match = ACTION_PATTERN.fullmatch(normalized_action)
        if not match:
            raise ValueError(f"Invalid action format: {action_text}")

        tool_name = match.group(1)
        tool_input = match.group(2).strip()
        if not tool_input:
            raise ValueError(f"Tool input is empty for action: {action_text}")
        return tool_name, tool_input

    def _normalize_action_text(self, text: str) -> str:
        normalized = text.strip()
        if normalized.startswith("`") and normalized.endswith("`"):
            normalized = normalized[1:-1].strip()
        return normalized

    def _collect_action_block(self, lines: list[str], start_index: int) -> str:
        first_line = lines[start_index].strip()
        content = first_line[len("Action:") :].strip()
        collected_lines = [content] if content else []

        candidate = "\n".join(collected_lines).strip()
        if self._is_complete_action(candidate):
            return self._normalize_action_text(candidate)

        index = start_index + 1
        while index < len(lines):
            line = lines[index].rstrip()
            stripped = line.strip()

            if stripped and self._looks_like_protocol_label(stripped):
                break

            collected_lines.append(line)
            candidate = "\n".join(collected_lines).strip()
            if self._is_complete_action(candidate):
                return self._normalize_action_text(candidate)
            index += 1

        return self._normalize_action_text("\n".join(collected_lines).strip())

    def _is_complete_action(self, text: str) -> bool:
        normalized = self._normalize_action_text(text)
        if not normalized:
            return False
        return bool(ACTION_PATTERN.fullmatch(normalized))

    def _looks_like_protocol_label(self, line: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*:\s*", line))


if __name__ == "__main__":
    client = HelloAgents()
    tool_executor = ToolExecutor()

    tool_executor.ToolRegister(
        tool_name="Search",
        description="基于serpAPI搜索网站/商品等内容",
        func=search,
    )
    print(tool_executor.getAvailableTools())

    agent = ReActAgent(llm_client=client, tool_executor=tool_executor)
    response = agent.run(
        question="iphone公司最新的手机是什么? 在旧版本的基础上提升/新增了什么功能？"
    )
    print(response)
