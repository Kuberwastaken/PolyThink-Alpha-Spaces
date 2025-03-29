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
        self.model_path = model_path
        
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
    
    async def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Generate a solution to a problem
        """
        # Construct solver prompt
        prompt = self._construct_solver_prompt(problem)
        
        # Generate solution
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Adjust generation parameters based on model
        max_new_tokens = 150  # Reduced from 200 to limit verbosity
        temperature = 0.5  # Lower temperature for more focused responses
        
        if "Llama" in self.model_name:
            min_new_tokens = 20  # Ensure Llama provides at least this many tokens
        else:
            min_new_tokens = 20  # Ensure Phi-2 provides at least this many tokens
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            top_p=0.9  # Added to focus on more likely tokens
        )
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the actual solution part
        if "SOLUTION:" in solution:
            solution_text = solution.split("SOLUTION:")[1].strip()
        else:
            solution_text = solution.strip()
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution_text,
            "confidence": self._calculate_confidence(solution_text)
        }
    
    async def evaluate_solutions(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate existing solutions and determine a winner
        """
        if not solutions or len(solutions) == 0:
            return {
                "agent_id": self.id,
                "model_name": self.model_name,
                "specialization": self.specialization,
                "judgment": "No solutions provided to evaluate",
                "recommendations": "Please provide solutions to evaluate",
                "winner": None,
                "reprompt_needed": False
            }
        
        prompt = self._construct_judge_prompt(problem, solutions)
        
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
        judgment_dict = self._parse_judgment(judgment, solutions)
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "judgment": judgment_dict["judgment"],
            "recommendations": judgment_dict["recommendations"],
            "winner": judgment_dict["winner"],
            "reprompt_needed": judgment_dict["reprompt_needed"]
        }
    
    async def reprompt_with_context(self, problem: str, solutions: List[Dict[str, Any]], judge_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-solve a problem with knowledge of other solutions and judge feedback
        """
        prompt = self._construct_reprompt(problem, solutions, judge_feedback)
        
        # Generate revised solution
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs, 
            max_length=1500, 
            num_return_sequences=1,
            temperature=0.5
        )
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the revised solution part
        if "REVISED SOLUTION:" in solution:
            revised = solution.split("REVISED SOLUTION:")[1].strip()
        else:
            revised = solution.strip()
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": revised,
            "is_revised": True,
            "confidence": self._calculate_confidence(revised) * 1.2  # Slightly boost confidence for revised solution
        }
    
    def _construct_solver_prompt(self, problem: str) -> str:
        """
        Create a specialized prompt for solver agents
        """
        base_prompt = f"""
        PROBLEM-SOLVING TASK
        
        PROBLEM: {problem}
        
        SOLUTION GUIDELINES:
        - Start with the direct answer in one sentence
        - Provide only essential reasoning (2-3 sentences maximum)
        - For simple math problems, just show the calculation and result
        - For complex problems, use brief step-by-step reasoning
        - Total response should be under 75 words
        
        SOLUTION:
        """
        
        # Add model-specific instructions
        if "Llama" in self.model_name:
            # Llama tends to be too brief, encourage slightly more explanation
            return base_prompt
        else:
            # Phi tends to be verbose, encourage brevity
            return base_prompt + "\nIMPORTANT: Be extremely concise. Avoid unnecessary steps or explanations."
    
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
        3. Select the best solution and explicitly name the winner (Phi-2 or Llama 3.2)
        4. Determine if the solutions significantly disagree on the core answer. If they do, mark "REPROMPT: YES"
        5. If reprompt is needed, explain specifically what the agents should focus on in their revised solutions
        
        YOUR EVALUATION:
        """
        
        return prompt
    
    def _construct_reprompt(self, problem: str, solutions: List[Dict[str, Any]], judge_feedback: Dict[str, Any]) -> str:
        """
        Create a reprompt that includes other solutions and judge feedback
        """
        # Find other solutions (not from this agent)
        other_solutions = [s for s in solutions if s['agent_id'] != self.id]
        
        prompt = f"""
        PROBLEM RE-EVALUATION TASK
        
        PROBLEM: {problem}
        
        Your previous solution:
        {next((s['solution'] for s in solutions if s['agent_id'] == self.id), "No previous solution found")}
        
        Other agent's solution:
        {other_solutions[0]['solution'] if other_solutions else "No other solutions available"}
        
        Judge's feedback:
        {judge_feedback.get('judgment', 'No specific feedback provided')}
        
        Specific recommendations:
        {judge_feedback.get('recommendations', 'No specific recommendations provided')}
        
        INSTRUCTIONS:
        - Reconsider your solution in light of the other agent's approach and the judge's feedback
        - Focus specifically on areas where you may have made errors or where your approach differs
        - Provide a revised solution that addresses the feedback
        - Be explicit about what you're changing and why
        
        REVISED SOLUTION:
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
            "winner": None,
            "reprompt_needed": False
        }
        
        # Check for explicit winner indication
        if "WINNER:" in judgment_text:
            winner_section = judgment_text.split("WINNER:")[1].split("\n")[0].strip()
            for sol in solutions:
                if sol['model_name'].lower() in winner_section.lower():
                    parsed["winner"] = sol['model_name']
                    break
        
        # Check for explicit reprompt need
        if "REPROMPT: YES" in judgment_text.upper() or "REPROMPT:YES" in judgment_text.upper():
            parsed["reprompt_needed"] = True
            
        # Extract recommendations if available
        if "RECOMMENDATIONS:" in judgment_text:
            parsed["recommendations"] = judgment_text.split("RECOMMENDATIONS:")[1].strip()
        
        # If no winner was explicitly mentioned but one solution is clearly favored
        if not parsed["winner"] and len(solutions) > 0:
            # Simple heuristic: check for positive mentions
            mentions = {sol['model_name']: 0 for sol in solutions}
            
            # Look for positive associations
            positive_terms = ["correct", "best", "accurate", "prefer", "better", "winner"]
            for sol in solutions:
                for term in positive_terms:
                    if term in judgment_text.lower() and sol['model_name'].lower() in judgment_text.lower():
                        mentions[sol['model_name']] += 1
            
            # Get the most mentioned model if any
            if any(mentions.values()):
                parsed["winner"] = max(mentions, key=mentions.get)
        
        # If solutions have significantly different answers, we might need a reprompt
        if not parsed["reprompt_needed"] and len(solutions) >= 2:
            # Check if judge mentions disagreement or errors
            disagreement_terms = ["disagree", "conflict", "different", "contradictory", "error", "mistake"]
            if any(term in judgment_text.lower() for term in disagreement_terms):
                parsed["reprompt_needed"] = True
        
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
    
    async def get_solver_solutions(self, problem: str) -> List[Dict[str, Any]]:
        """
        Get solutions from all solver agents in parallel
        """
        solver_tasks = [agent.solve_problem(problem) for agent in self.solver_agents]
        return await asyncio.gather(*solver_tasks)
    
    async def get_judge_evaluation(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get evaluation from judge agent
        """
        return await self.judge_agent.evaluate_solutions(problem, solutions)
    
    async def get_revised_solutions(self, problem: str, solutions: List[Dict[str, Any]], judgment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get revised solutions from solver agents based on judgment
        """
        reprompt_tasks = [
            agent.reprompt_with_context(problem, solutions, judgment) 
            for agent in self.solver_agents
        ]
        return await asyncio.gather(*reprompt_tasks)

def create_advanced_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    
    # State variables to store intermediate results
    current_problem = None
    current_solutions = None
    current_judgment = None
    
    # Step 1: Get solver solutions, then automatically get judgment
    def get_solutions_and_judge(problem):
        global current_problem, current_solutions, current_judgment
        
        # Reset state
        current_problem = problem
        current_solutions = None
        current_judgment = None
        
        # Get solutions
        current_solutions = asyncio.run(orchestrator.get_solver_solutions(problem))
        
        # Format the initial solutions for display
        phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        
        # Automatically get judgment
        current_judgment = asyncio.run(orchestrator.get_judge_evaluation(problem, current_solutions))
        
        # Check if reprompt is needed
        reprompt_needed = current_judgment.get('reprompt_needed', False)
        
        # If reprompt is needed, automatically get revised solutions
        if reprompt_needed:
            revised_solutions = asyncio.run(orchestrator.get_revised_solutions(
                problem, current_solutions, current_judgment
            ))
            
            # Update the current solutions with revised ones
            current_solutions = revised_solutions
            
            # Get updated phi2 and llama solutions
            phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
            llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        
        # Return all results
        return [
            phi2_sol['solution'] if phi2_sol else "Error retrieving solution",
            llama_sol['solution'] if llama_sol else "Error retrieving solution",
            phi2_sol['confidence'] if phi2_sol else 0,
            llama_sol['confidence'] if llama_sol else 0,
            current_judgment['judgment'],
            current_judgment.get('winner', "No clear winner"),
            current_judgment.get('recommendations', "No specific recommendations"),
            gr.update(visible=False)  # Hide judge button since it's automatic
        ]
    
    # Create the interface
    with gr.Blocks(title="PolyThink Multi-Agent Problem Solver") as demo:
        gr.Markdown("# PolyThink: Multi-Agent Problem Solving")
        gr.Markdown("Enter a problem and watch as multiple AI agents collaborate to solve it.")
        
        with gr.Row():
            problem_input = gr.Textbox(
                label="Problem to Solve",
                placeholder="Enter a problem or question here...",
                lines=3
            )
            solve_button = gr.Button("Solve Problem")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Phi-2 Solution")
                phi2_solution = gr.Textbox(label="Solution", lines=10)
                phi2_confidence = gr.Number(label="Confidence")
            
            with gr.Column():
                gr.Markdown("### Llama 3.2 1b Solution")
                llama_solution = gr.Textbox(label="Solution", lines=10)
                llama_confidence = gr.Number(label="Confidence")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Judge Evaluation")
                judgment_text = gr.Textbox(label="Judgment", lines=10)
                winner_text = gr.Textbox(label="Winner")
                recommendations_text = gr.Textbox(label="Recommendations", lines=3)
        
        # Connect the components
        solve_button.click(
            fn=get_solutions_and_judge,
            inputs=[problem_input],
            outputs=[
                phi2_solution, llama_solution, 
                phi2_confidence, llama_confidence,
                judgment_text, winner_text, recommendations_text,
                solve_button  # This is just a placeholder, we'll update visibility
            ]
        )
        
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_advanced_polythink_interface()
    demo.launch()