import os
import uuid
import asyncio
import gradio as gr
import logging
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get device information
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

class PolyThinkAgent:
    def __init__(self, model_name: str, model_path: str, role: str = "solver"):
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        self.role = role
        self.model_path = model_path
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is not set!")
        try:
            login(token=self.hf_token)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=self.hf_token, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, token=self.hf_token, torch_dtype=torch.float16, device_map="auto")
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
        self.specialization = self._determine_specialization()

    def _determine_specialization(self):
        if self.role == "judge":
            return "Evaluation and Consensus Building"
        return {"Phi-2": "Analytical Problem Solving", "Llama 3.2 1b": "Creative Solution Generation"}.get(self.model_name, "General Problem Solver")

    async def solve_problem(self, problem: str) -> Dict[str, Any]:
        prompt = self._construct_solver_prompt(problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        max_new_tokens = 75 if "Phi-2" in self.model_name else 100
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,
            num_return_sequences=1,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution_text = solution.split("SOLUTION:")[1].strip() if "SOLUTION:" in solution else solution.strip()
        if "IMPORTANT" in solution_text:
            solution_text = solution_text.split("IMPORTANT")[0].strip()
        solution_lines = solution_text.split("\n")
        solution_text = "\n".join(sorted(set(solution_lines), key=solution_lines.index))[:3]
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution_text,
            "confidence": self._calculate_confidence(solution_text)
        }

    async def evaluate_solutions(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not solutions:
            return {"judgment": "No solutions provided", "comparative_prompt": "", "winner": None}
        prompt = self._construct_judge_prompt(problem, solutions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=1500, num_return_sequences=1, temperature=0.3)
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_initial_judgment(judgment, problem, solutions)

    async def evaluate_opinions(self, problem: str, solutions: List[Dict[str, Any]], opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self._construct_final_judge_prompt(problem, solutions, opinions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=1500, num_return_sequences=1, temperature=0.3)
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_final_judgment(judgment, solutions)

    async def provide_opinion(self, comparative_prompt: str) -> Dict[str, Any]:
        inputs = self.tokenizer(comparative_prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.5,
            do_sample=True,
            top_p=0.9
        )
        opinion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        opinion_text = opinion.split("OPINION:")[1].strip() if "OPINION:" in opinion else opinion.strip()
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "opinion": opinion_text,
            "confidence": self._calculate_confidence(opinion_text)
        }

    def _construct_solver_prompt(self, problem: str) -> str:
        base_prompt = f"""
        PROBLEM-SOLVING TASK
        PROBLEM: {problem}
        GUIDELINES:
        - Direct answer first (1 sentence)
        - Essential reasoning only (2-3 sentences max)
        - Simple math: show calculation and result
        - Complex problems: brief steps
        - Under 75 words
        SOLUTION:
        """
        if "Phi-2" in self.model_name:
            return base_prompt + "\nIMPORTANT: Be concise, exclude unnecessary text."
        return base_prompt

    def _construct_judge_prompt(self, problem: str, solutions: List[Dict[str, Any]]) -> str:
        prompt = f"""
        SOLUTION EVALUATION TASK
        PROBLEM: {problem}
        You are evaluating solutions.
        SOLUTIONS:
        """
        for i, sol in enumerate(solutions):
            prompt += f"SOLUTION {i+1} from {sol['model_name']} ({sol['specialization']}):\n{sol['solution']}\n\n"
        prompt += """
        YOUR EVALUATION:
        1. COMPARATIVE ANALYSIS:
        - Correctness: Compare accuracy
        - Clarity: Assess explanation clarity
        - Completeness: Check problem coverage
        2. STRENGTHS AND WEAKNESSES:
        - Solution 1: Strengths, Weaknesses
        - Solution 2: Strengths, Weaknesses
        3. COMPARATIVE PROMPT: Write a prompt like:
           "What do you think is a better solution to [problem]?
            [Solution 1]
            [Solution 2]
            Give reasoning."
        """
        return prompt

    def _construct_final_judge_prompt(self, problem: str, solutions: List[Dict[str, Any]], opinions: List[Dict[str, Any]]) -> str:
        prompt = f"""
        FINAL EVALUATION TASK
        PROBLEM: {problem}
        SOLUTIONS:
        """
        for i, sol in enumerate(solutions):
            prompt += f"SOLUTION {i+1} from {sol['model_name']}:\n{sol['solution']}\n\n"
        prompt += "OPINIONS:\n"
        for i, op in enumerate(opinions):
            prompt += f"OPINION {i+1} from {op['model_name']}:\n{op['opinion']}\n\n"
        prompt += """
        INSTRUCTIONS:
        1. Review solutions and opinions
        2. Determine which solution is better based on:
           - Original solutions' quality
           - Strength of reasoning in opinions
        3. Name the winner (Phi-2 or Llama 3.2)
        FINAL JUDGMENT:
        """
        return prompt

    def _parse_initial_judgment(self, judgment_text: str, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        parsed = {"judgment": judgment_text, "comparative_prompt": "", "winner": None}
        if "COMPARATIVE PROMPT:" in judgment_text:
            prompt_section = judgment_text.split("COMPARATIVE PROMPT:")[1].strip()
            parsed["comparative_prompt"] = prompt_section
        else:
            # Default comparative prompt if not explicitly provided
            parsed["comparative_prompt"] = f"""
            What do you think is a better solution to "{problem}"?
            {solutions[0]['model_name']}: {solutions[0]['solution']}
            {solutions[1]['model_name']}: {solutions[1]['solution']}
            Give reasoning.
            OPINION:
            """
        return parsed

    def _parse_final_judgment(self, judgment_text: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        parsed = {"judgment": judgment_text, "winner": None}
        if "WINNER:" in judgment_text:
            winner_section = judgment_text.split("WINNER:")[1].split("\n")[0].strip()
            for sol in solutions:
                if sol['model_name'].lower() in winner_section.lower():
                    parsed["winner"] = sol['model_name']
        elif "FINAL JUDGMENT:" in judgment_text:
            final_section = judgment_text.split("FINAL JUDGMENT:")[1].strip()
            for sol in solutions:
                if sol['model_name'].lower() in final_section.lower():
                    parsed["winner"] = sol['model_name']
        return parsed

    def _calculate_confidence(self, text: str) -> float:
        word_count = len(text.split())
        return min(word_count / 100, 1.0) * 100

class PolyThinkAgentOrchestrator:
    def __init__(self):
        self.solver_agents = [
            PolyThinkAgent("Phi-2", "microsoft/phi-2", role="solver"),
            PolyThinkAgent("Llama 3.2 1b", "meta-llama/llama-3.2-1b", role="solver")
        ]
        self.judge_agent = PolyThinkAgent("DeepSeek R1 1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", role="judge")

    async def get_solver_solutions(self, problem: str) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[agent.solve_problem(problem) for agent in self.solver_agents])

    async def get_initial_judge_evaluation(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.judge_agent.evaluate_solutions(problem, solutions)

    async def get_solver_opinions(self, comparative_prompt: str) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[agent.provide_opinion(comparative_prompt) for agent in self.solver_agents])

    async def get_final_judge_evaluation(self, problem: str, solutions: List[Dict[str, Any]], opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.judge_agent.evaluate_opinions(problem, solutions, opinions)

def create_advanced_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    current_problem = None
    current_solutions = None
    current_initial_judgment = None
    current_opinions = None
    current_final_judgment = None

    def get_solver_solutions(problem):
        nonlocal current_problem, current_solutions
        current_problem = problem
        current_solutions = asyncio.run(orchestrator.get_solver_solutions(problem))
        phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        return [
            phi2_sol['solution'] if phi2_sol else "Error",
            llama_sol['solution'] if llama_sol else "Error",
            phi2_sol['confidence'] if phi2_sol else 0,
            llama_sol['confidence'] if llama_sol else 0
        ]

    def get_initial_judge_evaluation():
        nonlocal current_problem, current_solutions, current_initial_judgment
        if not current_problem or not current_solutions:
            return ["No data", ""]
        current_initial_judgment = asyncio.run(orchestrator.get_initial_judge_evaluation(current_problem, current_solutions))
        return [current_initial_judgment['judgment'], current_initial_judgment['comparative_prompt']]

    def get_solver_opinions():
        nonlocal current_initial_judgment, current_opinions
        if not current_initial_judgment or not current_initial_judgment.get('comparative_prompt'):
            return ["No data", "No data", 0, 0]
        current_opinions = asyncio.run(orchestrator.get_solver_opinions(current_initial_judgment['comparative_prompt']))
        phi2_op = next((o for o in current_opinions if o['model_name'] == "Phi-2"), None)
        llama_op = next((o for o in current_opinions if o['model_name'] == "Llama 3.2 1b"), None)
        return [
            phi2_op['opinion'] if phi2_op else "Error",
            llama_op['opinion'] if llama_op else "Error",
            phi2_op['confidence'] if phi2_op else 0,
            llama_op['confidence'] if llama_op else 0
        ]

    def get_final_judge_evaluation():
        nonlocal current_problem, current_solutions, current_opinions, current_final_judgment
        if not current_problem or not current_solutions or not current_opinions:
            return ["No data", "No clear winner"]
        current_final_judgment = asyncio.run(orchestrator.get_final_judge_evaluation(current_problem, current_solutions, current_opinions))
        return [current_final_judgment['judgment'], current_final_judgment.get('winner', "No clear winner")]

    def generate_final_report():
        if current_final_judgment:
            report = f"## Final Report\n\n**Problem:** {current_problem}\n\n"
            for sol in current_solutions:
                report += f"### {sol['model_name']} Solution\n{sol['solution']}\n\n"
            report += f"### Initial Judge Evaluation\n{current_initial_judgment['judgment']}\n\n"
            report += f"### Comparative Prompt\n{current_initial_judgment['comparative_prompt']}\n\n"
            for op in current_opinions:
                report += f"### {op['model_name']} Opinion\n{op['opinion']}\n\n"
            report += f"### Final Judge Evaluation\n{current_final_judgment['judgment']}\n\n"
            report += f"### Winner: {current_final_judgment['winner']}\n\n"
            return report
        return "Process ongoing..."

    css = """
    .status { background-color: #e6f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; border: 1px solid #007bff; }
    .solution-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #fafafa; }
    .button-primary { background-color: #007bff; color: white; }
    """

    with gr.Blocks(css=css, title="PolyThink Multi-Agent Problem Solver") as demo:
        status_text = gr.Markdown("### 🔄 Status: Ready", elem_classes=["status"])
        gr.Markdown("# PolyThink: Multi-Agent Problem Solving")
        gr.Markdown("Enter a problem and let AI agents solve and evaluate collaboratively!")

        with gr.Row():
            problem_input = gr.Textbox(label="Problem to Solve", placeholder="E.g., What is 5+10?", lines=3)
            solve_button = gr.Button("Solve Problem", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Phi-2 Solution")
                phi2_solution = gr.Textbox(label="Solution", lines=5, elem_classes=["solution-box"])
                phi2_confidence = gr.Number(label="Confidence")
            with gr.Column():
                gr.Markdown("### Llama 3.2 Solution")
                llama_solution = gr.Textbox(label="Solution", lines=5, elem_classes=["solution-box"])
                llama_confidence = gr.Number(label="Confidence")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Initial Judge Evaluation")
                initial_judgment_text = gr.Textbox(label="Judgment", lines=5, elem_classes=["solution-box"])
                comparative_prompt_text = gr.Textbox(label="Comparative Prompt", lines=5, elem_classes=["solution-box"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Phi-2 Opinion")
                phi2_opinion = gr.Textbox(label="Opinion", lines=5, elem_classes=["solution-box"])
                phi2_op_confidence = gr.Number(label="Confidence")
            with gr.Column():
                gr.Markdown("### Llama 3.2 Opinion")
                llama_opinion = gr.Textbox(label="Opinion", lines=5, elem_classes=["solution-box"])
                llama_op_confidence = gr.Number(label="Confidence")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Final Judge Evaluation")
                final_judgment_text = gr.Textbox(label="Judgment", lines=5, elem_classes=["solution-box"])
                winner_text = gr.Textbox(label="Winner")

        final_report = gr.Markdown("### 📜 Final Report\n\nWaiting for completion...")

        solve_button.click(
            fn=lambda: gr.update(value="### 🔄 Status: Generating solutions...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_solver_solutions,
            inputs=[problem_input],
            outputs=[phi2_solution, llama_solution, phi2_confidence, llama_confidence]
        ).then(
            fn=lambda: gr.update(value="### ⚖️ Status: Initial judge evaluation...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_initial_judge_evaluation,
            outputs=[initial_judgment_text, comparative_prompt_text]
        ).then(
            fn=lambda: gr.update(value="### 🔄 Status: Gathering solver opinions...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_solver_opinions,
            outputs=[phi2_opinion, llama_opinion, phi2_op_confidence, llama_op_confidence]
        ).then(
            fn=lambda: gr.update(value="### ⚖️ Status: Final judge evaluation...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_final_judge_evaluation,
            outputs=[final_judgment_text, winner_text]
        ).then(
            fn=generate_final_report,
            outputs=[final_report]
        ).then(
            fn=lambda: gr.update(value="### ✅ Status: Process complete!", elem_classes=["status"]),
            outputs=[status_text]
        )

    return demo

if __name__ == "__main__":
    demo = create_advanced_polythink_interface()
    demo.launch()