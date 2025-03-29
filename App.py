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
            repetition_penalty=1.2  # Added to reduce repetition in Llama
        )
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution_text = solution.split("SOLUTION:")[1].strip() if "SOLUTION:" in solution else solution.strip()
        # Clean up Phi-2 output
        if "IMPORTANT" in solution_text:
            solution_text = solution_text.split("IMPORTANT")[0].strip()
        # Limit Llama repetition
        solution_lines = solution_text.split("\n")
        solution_text = "\n".join(sorted(set(solution_lines), key=solution_lines.index))[:3]  # Keep first 3 unique lines
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution_text,
            "confidence": self._calculate_confidence(solution_text)
        }

    async def evaluate_solutions(self, problem: str, solutions: List[Dict[str, Any]], final_pass: bool = False) -> Dict[str, Any]:
        if not solutions:
            return {"judgment": "No solutions provided", "recommendations": "", "winner": None, "reprompt_needed": False}
        prompt = self._construct_judge_prompt(problem, solutions, final_pass)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=1500, num_return_sequences=1, temperature=0.3)
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_judgment(judgment, solutions)

    async def reprompt_with_context(self, problem: str, solutions: List[Dict[str, Any]], judge_feedback: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._construct_reprompt(problem, solutions, judge_feedback)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_length=1000, num_return_sequences=1, temperature=0.5)
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        revised = solution.split("REVISED SOLUTION:")[1].strip() if "REVISED SOLUTION:" in solution else solution.strip()
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": revised,
            "is_revised": True,
            "confidence": self._calculate_confidence(revised) * 1.2
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

    def _construct_judge_prompt(self, problem: str, solutions: List[Dict[str, Any]], final_pass: bool) -> str:
        prompt = f"""
        SOLUTION EVALUATION TASK
        PROBLEM: {problem}
        You are evaluating solutions. Review carefully.
        SOLUTIONS:
        """
        for i, sol in enumerate(solutions):
            prompt += f"SOLUTION {i+1} from {sol['model_name']} ({sol['specialization']}):\n{sol['solution']}\n\n"
        if not final_pass:
            prompt += """
            YOUR EVALUATION:
            1. COMPARATIVE ANALYSIS:
            - Correctness: Compare accuracy
            - Clarity: Assess explanation clarity
            - Completeness: Check problem coverage
            2. STRENGTHS AND WEAKNESSES:
            - Solution 1: Strengths, Weaknesses
            - Solution 2: Strengths, Weaknesses
            3. INITIAL OPINION: Which is better (Phi-2 or Llama 3.2) and why
            4. REPROMPT NEEDED: YES/NO
            5. RECOMMENDATIONS: If YES, specify improvements
            """
        else:
            prompt += f"""
            FINAL EVALUATION:
            Which is better for "{problem}"?
            - Phi-2: {solutions[0]['solution']}
            - Llama 3.2: {solutions[1]['solution']}
            Give reasons and name the winner (Phi-2 or Llama 3.2).
            FINAL ANSWER:
            """
        return prompt

    def _construct_reprompt(self, problem: str, solutions: List[Dict[str, Any]], judge_feedback: Dict[str, Any]) -> str:
        other_solutions = [s for s in solutions if s['agent_id'] != self.id]
        return f"""
        PROBLEM RE-EVALUATION
        PROBLEM: {problem}
        Your solution: {next((s['solution'] for s in solutions if s['agent_id'] == self.id), "N/A")}
        Other solution: {other_solutions[0]['solution'] if other_solutions else "N/A"}
        Judge feedback: {judge_feedback.get('judgment', 'N/A')}
        Recommendations: {judge_feedback.get('recommendations', 'N/A')}
        INSTRUCTIONS:
        - Revise based on feedback and other solution
        - Address errors or differences
        - Be concise
        REVISED SOLUTION:
        """

    def _parse_judgment(self, judgment_text: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        parsed = {"judgment": judgment_text, "recommendations": "", "winner": None, "reprompt_needed": False}
        if "WINNER:" in judgment_text:
            winner_section = judgment_text.split("WINNER:")[1].split("\n")[0].strip()
            for sol in solutions:
                if sol['model_name'].lower() in winner_section.lower():
                    parsed["winner"] = sol['model_name']
        if "REPROMPT NEEDED: YES" in judgment_text.upper():
            parsed["reprompt_needed"] = True
        if "RECOMMENDATIONS:" in judgment_text:
            parsed["recommendations"] = judgment_text.split("RECOMMENDATIONS:")[1].strip()
        if not parsed["winner"] and "FINAL ANSWER:" in judgment_text:
            final_section = judgment_text.split("FINAL ANSWER:")[1].strip()
            for sol in solutions:
                if sol['model_name'].lower() in final_section.lower():
                    parsed["winner"] = sol['model_name']
        return parsed

    def _calculate_confidence(self, solution: str) -> float:
        word_count = len(solution.split())
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

    async def get_judge_evaluation(self, problem: str, solutions: List[Dict[str, Any]], final_pass: bool = False) -> Dict[str, Any]:
        return await self.judge_agent.evaluate_solutions(problem, solutions, final_pass)

    async def get_revised_solutions(self, problem: str, solutions: List[Dict[str, Any]], judgment: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[agent.reprompt_with_context(problem, solutions, judgment) for agent in self.solver_agents])

def create_advanced_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    current_problem = None
    current_solutions = None
    current_judgment = None
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

    def get_judge_evaluation_initial():
        nonlocal current_problem, current_solutions, current_judgment
        if not current_problem or not current_solutions:
            return ["No data", "", "", gr.update(visible=False)]
        current_judgment = asyncio.run(orchestrator.get_judge_evaluation(current_problem, current_solutions, final_pass=False))
        return [
            current_judgment['judgment'],
            current_judgment.get('winner', "Pending final evaluation"),
            current_judgment.get('recommendations', "N/A"),
            gr.update(visible=current_judgment.get('reprompt_needed', False))
        ]

    def get_judge_evaluation_final():
        nonlocal current_problem, current_solutions, current_final_judgment
        current_final_judgment = asyncio.run(orchestrator.get_judge_evaluation(current_problem, current_solutions, final_pass=True))
        return [
            current_final_judgment['judgment'],
            current_final_judgment.get('winner', "No clear winner"),
            current_final_judgment.get('recommendations', "N/A"),
            gr.update(visible=False)  # Hide reprompt button after final judgment
        ]

    def get_revised_solutions():
        nonlocal current_problem, current_solutions, current_judgment
        if not current_problem or not current_solutions or not current_judgment:
            return ["No data", "No data", 0, 0]
        current_solutions = asyncio.run(orchestrator.get_revised_solutions(current_problem, current_solutions, current_judgment))
        phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        return [
            phi2_sol['solution'] if phi2_sol else "Error",
            llama_sol['solution'] if llama_sol else "Error",
            phi2_sol['confidence'] if phi2_sol else 0,
            llama_sol['confidence'] if llama_sol else 0
        ]

    def generate_final_report():
        if current_final_judgment and not current_judgment.get('reprompt_needed', False):
            report = f"## Final Report\n\n**Problem:** {current_problem}\n\n"
            for sol in current_solutions:
                report += f"### {sol['model_name']} Solution\n{sol['solution']}\n\n"
            report += f"### Initial Judge Evaluation\n{current_judgment['judgment']}\n\n"
            report += f"### Final Judge Evaluation\n{current_final_judgment['judgment']}\n\n"
            report += f"### Winner: {current_final_judgment['winner']}\n\n"
            if 'recommendations' in current_judgment:
                report += f"### Recommendations\n{current_judgment['recommendations']}\n\n"
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
        gr.Markdown("Enter a problem and let AI agents solve it collaboratively!")

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
                gr.Markdown("### Judge Evaluation")
                judgment_text = gr.Textbox(label="Judgment", lines=10, elem_classes=["solution-box"])
                winner_text = gr.Textbox(label="Winner")
                recommendations_text = gr.Textbox(label="Recommendations", lines=3)
                reprompt_button = gr.Button("Get Revised Solutions", visible=False)

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
            fn=get_judge_evaluation_initial,
            outputs=[judgment_text, winner_text, recommendations_text, reprompt_button]
        ).then(
            fn=lambda: gr.update(value="### ⚖️ Status: Final judge evaluation...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_judge_evaluation_final,
            outputs=[judgment_text, winner_text, recommendations_text, reprompt_button]
        ).then(
            fn=generate_final_report,
            outputs=[final_report]
        ).then(
            fn=lambda: gr.update(value="### ✅ Status: Process complete!", elem_classes=["status"]),
            outputs=[status_text]
        )

        reprompt_button.click(
            fn=lambda: gr.update(value="### 🔄 Status: Getting revised solutions...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_revised_solutions,
            outputs=[phi2_solution, llama_solution, phi2_confidence, llama_confidence]
        ).then(
            fn=lambda: gr.update(value="### ⚖️ Status: Re-evaluating with judge...", elem_classes=["status"]),
            outputs=[status_text]
        ).then(
            fn=get_judge_evaluation_initial,
            outputs=[judgment_text, winner_text, recommendations_text, reprompt_button]
        ).then(
            fn=get_judge_evaluation_final,
            outputs=[judgment_text, winner_text, recommendations_text, reprompt_button]
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