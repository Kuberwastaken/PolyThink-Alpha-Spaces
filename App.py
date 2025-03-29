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
        """
        Initialize an agent with specific model capabilities
        
        Args:
            model_name: The display name of the model
            model_path: The Hugging Face path to load the model
            role: Either "solver" or "judge"
        """
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        self.role = role
        
        # Get HF token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            logger.error("HF_TOKEN environment variable is not set!")
            raise ValueError("HF_TOKEN environment variable is not set!")
        
        try:
            # Explicit login before model loading
            login(token=self.hf_token)
            logger.info(f"Successfully logged in to Hugging Face Hub")
            
            logger.info(f"Loading model: {model_name} from {model_path} on {DEVICE}")
            print(f"Time: {datetime.now()} - Loading model: {model_name}")
            
            # Standard handling for models
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=self.hf_token,
                use_fast=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=self.hf_token,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info(f"Successfully loaded model: {model_name}")
            print(f"Time: {datetime.now()} - Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            print(f"Error loading model {model_name}: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            raise
        
        # Agent-specific configuration
        self.specialization = self._determine_specialization()
    
    def _determine_specialization(self):
        """
        Assign a unique problem-solving specialization to the agent
        """
        if self.role == "judge":
            return "Evaluation and Consensus Building"
        
        specialization_map = {
            "Phi-2": "Analytical Problem Solving",
            "Llama 3.2 1b": "Creative Solution Generation"
        }
        return specialization_map.get(self.model_name, "General Problem Solver")
    
    async def solve_problem(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Asynchronous problem-solving method for solver agents
        """
        if self.role == "solver":
            return await self._generate_solution(problem, context)
        else:
            return await self._judge_solutions(problem, context)
    
    async def _generate_solution(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a solution for solver agents
        """
        # Construct specialized prompt
        prompt = self._construct_solver_prompt(problem)
        
        # Generate solution
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs, 
            max_length=1024, 
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the actual solution part
        if "SOLUTION:" in solution:
            solution = solution.split("SOLUTION:")[1].strip()
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution,
            "confidence": self._calculate_confidence(solution)
        }
    
    async def _judge_solutions(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judge existing solutions for judge agents
        """
        if not context or 'solutions' not in context or len(context['solutions']) == 0:
            return {
                "agent_id": self.id,
                "model_name": self.model_name,
                "specialization": self.specialization,
                "judgment": "No solutions provided to evaluate",
                "recommendations": "Please provide solutions to evaluate",
                "confidence": 0
            }
        
        prompt = self._construct_judge_prompt(problem, context['solutions'])
        
        # Generate judgment
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs, 
            max_length=1500, 
            num_return_sequences=1,
            temperature=0.3
        )
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract relevant parts if possible
        judgment_dict = self._parse_judgment(judgment, context['solutions'])
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "judgment": judgment_dict["judgment"],
            "recommendations": judgment_dict["recommendations"],
            "winner": judgment_dict["winner"],
            "confidence": 90.0  # Fixed high confidence for judge
        }
    
    def _construct_solver_prompt(self, problem: str) -> str:
        """
        Create a specialized prompt for solver agents
        """
        return f"""
        PROBLEM-SOLVING TASK
        
        PROBLEM: {problem}
        
        SOLUTION REQUIREMENTS:
        - Provide a clear, step-by-step solution
        - Show your reasoning
        - Explain alternative approaches if relevant
        - Ensure mathematical accuracy
        
        SOLUTION:
        """
    
    def _construct_judge_prompt(self, problem: str, solutions: List[Dict[str, Any]]) -> str:
        """
        Create a specialized prompt for judge agents
        """
        prompt = f"""
        SOLUTION EVALUATION TASK
        
        PROBLEM: {problem}
        
        You are the judge evaluating solutions from different problem-solving agents. 
        Review each solution carefully and determine which one is best.
        
        SOLUTIONS TO EVALUATE:
        """
        
        for i, sol in enumerate(solutions):
            prompt += f"""
        SOLUTION {i+1} from {sol['model_name']} ({sol['specialization']}):
        {sol['solution']}
        """
        
        prompt += f"""
        EVALUATION INSTRUCTIONS:
        1. Analyze each solution for correctness, clarity, and completeness
        2. Identify the strengths and weaknesses of each approach
        3. Select the best solution and explain your reasoning
        4. If neither solution is satisfactory, explain why and recommend improvements
        
        YOUR EVALUATION:
        """
        
        return prompt
    
    def _parse_judgment(self, judgment_text: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the judgment text into structured components
        """
        # Default values
        parsed = {
            "judgment": judgment_text,
            "recommendations": "",
            "winner": ""
        }
        
        # Try to extract structured information if available
        if "WINNER:" in judgment_text:
            winner_section = judgment_text.split("WINNER:")[1].split("\n")[0].strip()
            for sol in solutions:
                if sol['model_name'].lower() in winner_section.lower():
                    parsed["winner"] = sol['model_name']
                    break
        
        if "RECOMMENDATIONS:" in judgment_text:
            parsed["recommendations"] = judgment_text.split("RECOMMENDATIONS:")[1].strip()
        
        # If no winner was explicitly mentioned but one solution is clearly favored
        if not parsed["winner"] and len(solutions) > 0:
            # Count mentions of each model name in a positive context
            mentions = {sol['model_name']: 0 for sol in solutions}
            for sol in solutions:
                if f"correct" in judgment_text.lower() and sol['model_name'].lower() in judgment_text.lower():
                    mentions[sol['model_name']] += 5
                if f"best" in judgment_text.lower() and sol['model_name'].lower() in judgment_text.lower():
                    mentions[sol['model_name']] += 5
                if f"preferred" in judgment_text.lower() and sol['model_name'].lower() in judgment_text.lower():
                    mentions[sol['model_name']] += 3
            
            # Get the most mentioned model
            if any(mentions.values()):
                parsed["winner"] = max(mentions, key=mentions.get)
        
        return parsed
    
    def _calculate_confidence(self, solution: str) -> float:
        """
        Calculate solution confidence based on complexity and detail
        """
        word_count = len(solution.split())
        complexity_factor = min(word_count / 100, 1.0)
        return complexity_factor * 100

class PolyThinkAgentOrchestrator:
    def __init__(self):
        """
        Initialize multi-agent problem-solving system with DeepSeek as judge
        """
        self.solver_agents = [
            PolyThinkAgent("Phi-2", "microsoft/phi-2", role="solver"),
            PolyThinkAgent("Llama 3.2 1b", "meta-llama/llama-3.2-1b", role="solver")
        ]
        
        self.judge_agent = PolyThinkAgent(
            "DeepSeek R1 1.5B", 
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            role="judge"
        )
    
    async def solve_problem_multi_agent(self, problem: str) -> Dict[str, Any]:
        """
        Parallel problem-solving with solvers followed by judgment
        """
        # Step 1: Get solutions from solver agents
        solver_tasks = [agent.solve_problem(problem) for agent in self.solver_agents]
        solutions = await asyncio.gather(*solver_tasks)
        
        # Step 2: Judge the solutions
        judge_context = {"solutions": solutions}
        judgment = await self.judge_agent.solve_problem(problem, judge_context)
        
        return {
            "problem": problem,
            "solver_solutions": solutions,
            "judgment": judgment
        }

def create_advanced_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    
    def solve_problem(problem):
        return asyncio.run(orchestrator.solve_problem_multi_agent(problem))
    
    interface = gr.Blocks(theme=gr.themes.Default())
    
    with interface:
        gr.Markdown("# PolyThink: Multi-Agent Problem Solver with Judgment")
        
        with gr.Row():
            problem_input = gr.Textbox(
                label="Enter your problem",
                lines=4,
                placeholder="Enter your problem or question here..."
            )
            solve_button = gr.Button("Solve with Multiple Agents", variant="primary")
        
        with gr.Accordion("Solver Agents", open=True):
            with gr.Row():
                phi2_output = gr.Textbox(label="Phi-2 Solution", lines=10, interactive=False)
                llama_output = gr.Textbox(label="Llama 3.2 1b Solution", lines=10, interactive=False)
        
        with gr.Accordion("Judge Evaluation", open=True):
            judge_evaluation = gr.Textbox(
                label="DeepSeek Evaluation", 
                lines=12, 
                interactive=False
            )
            
            with gr.Row():
                winner_display = gr.Textbox(
                    label="Winning Solution",
                    interactive=False
                )
                
                recommendations_display = gr.Textbox(
                    label="Improvement Recommendations",
                    lines=4,
                    interactive=False
                )
        
        with gr.Accordion("Confidence Metrics", open=False):
            with gr.Row():
                phi2_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Phi-2 Confidence",
                    interactive=False
                )
                llama_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Llama Confidence",
                    interactive=False
                )
        
        with gr.Accordion("Advanced Features", open=False):
            reprompt_button = gr.Button("Reprompt Agents with Judge Feedback")
            
            detailed_reasoning = gr.Textbox(
                label="Detailed Reasoning",
                lines=3,
                interactive=False
            )
        
        def process_problem(problem):
            result = solve_problem(problem)
            
            phi2_sol = next((s for s in result['solver_solutions'] if s['model_name'] == "Phi-2"), None)
            llama_sol = next((s for s in result['solver_solutions'] if s['model_name'] == "Llama 3.2 1b"), None)
            judgment = result['judgment']
            
            return [
                phi2_sol['solution'] if phi2_sol else "Error retrieving solution",
                llama_sol['solution'] if llama_sol else "Error retrieving solution",
                judgment['judgment'],
                judgment.get('winner', 'No clear winner'),
                judgment.get('recommendations', 'No specific recommendations'),
                phi2_sol['confidence'] if phi2_sol else 0,
                llama_sol['confidence'] if llama_sol else 0
            ]
        
        solve_button.click(
            process_problem,
            inputs=problem_input,
            outputs=[
                phi2_output,
                llama_output,
                judge_evaluation,
                winner_display,
                recommendations_display,
                phi2_confidence,
                llama_confidence
            ]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_advanced_polythink_interface()
    interface.launch(debug=True)