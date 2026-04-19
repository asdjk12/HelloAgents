import re
from dataclasses import dataclass

from llm import HelloAgents
from prompt import *
from tool import ToolExecutor, search
from datetime import date
import ast

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

# Plan & Solve
class Planner():
    def __init__(self, client:HelloAgents) -> None:
        self.client = client

    def plan(self, question):
        # 初始化prompt
        prompt = PLANNER_PROMPT_TEMPLATE.format(question = question)

        # 初始化message
        message = [{"role": "user", "content": prompt}]

        
        
        # 调用llm进行提问
        print("正在进行planner的任务拆分~")      # 拆分
        response_text= self.client.think(message=message)
        print("任务拆分完成~")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []
    
class Executor():
    def __init__(self, client:HelloAgents) -> None:
        self.client = client
    
    def execute(self, plan, question):
        history= ""
        response = None

        for step,task in enumerate(plan):   # step: 1/2/3... && task: sub-tasks
            # 追踪记录
            print(f"这是第{step}步")
            
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question = question,
                plan =  plan,
                history = history,
                current_step= task
            )

            message = [{"role":"user", "content": prompt}]
            
            # 运行model
            print("运行模型: ")
            response = self.client.think(message=message)
            print("回答已生成！")

            history += f"步骤 {step+1}: {task}\n结果: {response}\n\n"
            print(f"步骤 {step+1} 已完成，结果: {response}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response
        return final_answer

class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        
        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)
        
        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(plan, question)
        
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

            


if __name__ == "__main__":
    client = HelloAgents()

    """
    # ReAct
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
    """

    # Plan && Solve
    P_S = PlanAndSolveAgent(llm_client=client)
    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    P_S.run(question)
