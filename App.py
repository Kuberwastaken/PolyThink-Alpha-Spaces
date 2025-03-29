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
        max_new_tokens = 200  # Limit token generation to prevent verbose responses
        temperature = 0.7
        
        if "Llama" in self.model_name:
            temperature = 0.5  # Lower temperature for more focused responses
            min_new_tokens = 50  # Ensure Llama provides at least this many tokens
        else:
            min_new_tokens = 30  # Ensure Phi-2 provides at least this many tokens
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True
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
        # Different prompts based on model to control verbosity
        if "Llama" in self.model_name:
            return f"""
            PROBLEM-SOLVING TASK
            
            PROBLEM: {problem}
            
            SOLUTION REQUIREMENTS:
            - Provide a concise, step-by-step solution (maximum 200 words)
            - Show only essential mathematical reasoning
            - Explain your approach briefly
            - Ensure accuracy in your calculations
            - Avoid repetition and unnecessary elaboration
            
            SOLUTION:
            """
        else:
            return f"""
            PROBLEM-SOLVING TASK
            
            PROBLEM: {problem}
            
            SOLUTION REQUIREMENTS:
            - Provide a clear, step-by-step solution (maximum 200 words)
            - Show your mathematical reasoning in detail
            - Explain your approach
            - Ensure accuracy in your calculations
            
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
    
    # Step 1: Get solver solutions
    def get_solutions(problem):
        global current_problem, current_solutions
        current_problem = problem
        current_solutions = asyncio.run(orchestrator.get_solver_solutions(problem))
        
        # Format the results for display
        phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        
        return [
            phi2_sol['solution'] if phi2_sol else "Error retrieving solution",
            llama_sol['solution'] if llama_sol else "Error retrieving solution",
            phi2_sol['confidence'] if phi2_sol else 0,
            llama_sol['confidence'] if llama_sol else 0,
            gr.update(visible=True)  # Make judge button visible
        ]
    
    # Step 2: Get judge evaluation
    def get_judgment():
        global current_problem, current_solutions, current_judgment
        
        if not current_problem or not current_solutions:
            return ["No problem or solutions available", "N/A", "N/A", gr.update(visible=False)]
        
        current_judgment = asyncio.run(orchestrator.get_judge_evaluation(current_problem, current_solutions))
        
        reprompt_needed = current_judgment.get('reprompt_needed', False)
        
        return [
            current_judgment['judgment'],
            current_judgment.get('winner', "No clear winner"),
            current_judgment.get('recommendations', "No specific recommendations"),
            gr.update(visible=reprompt_needed)  # Show reprompt button only if needed
        ]
    
    # Step 3: Get revised solutions if needed
    def get_revised_solutions():
        global current_problem, current_solutions, current_judgment
        
        if not current_problem or not current_solutions or not current_judgment:
            return ["No problem, solutions or judgment available", "No solutions available", 0, 0]
        
        revised_solutions = asyncio.run(orchestrator.get_revised_solutions(
            current_problem, current_solutions, current_judgment
        ))
        
        # Update the current solutions with revised ones
        current_solutions = revised_solutions
        
        # Format the results for display
        phi2_sol = next((s for s in revised_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in revised_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        
        return [
            phi2_sol['solution'] if phi2_sol else "Error retrieving revised solution",
            llama_sol['solution'] if llama_sol else "Error retrieving revised solution",
            phi2_sol['confidence'] if phi2_sol else 0,
            llama_sol['confidence'] if llama_sol else 0
        ]
    
    interface = gr.Blocks(theme=gr.themes.Default())
    
    with interface:
        gr.Markdown("# PolyThink: Multi-Agent Problem Solver with Sequential Evaluation")
        
        with gr.Row():
            problem_input = gr.Textbox(
                label="Enter your problem",
                lines=4,
                placeholder="Enter your problem or question here..."
            )
            solve_button = gr.Button("Get Solutions First", variant="primary")
        
        with gr.Accordion("Solver Agents", open=True):
            with gr.Row():
                phi2_output = gr.Textbox(label="Phi-2 Solution", lines=10, interactive=False)
                llama_output = gr.Textbox(label="Llama 3.2 1b Solution", lines=10, interactive=False)
            
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
        
        with gr.Row():
            judge_button = gr.Button("Evaluate Solutions with DeepSeek", visible=False)
        
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
        
        with gr.Row():
            reprompt_button = gr.Button("Reprompt Agents with Judge Feedback", visible=False)
        
        with gr.Accordion("Revised Solutions", open=True, visible=True):
            with gr.Row():
                phi2_revised = gr.Textbox(label="Phi-2 Revised Solution", lines=10, interactive=False)
                llama_revised = gr.Textbox(label="Llama 3.2 Revised Solution", lines=10, interactive=False)
            
            with gr.Row():
                phi2_revised_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Phi-2 Revised Confidence",
                    interactive=False
                )
                llama_revised_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Llama Revised Confidence",
                    interactive=False
                )
        
        # Connect the buttons to their respective functions
        solve_button.click(
            get_solutions,
            inputs=problem_input,
            outputs=[
                phi2_output,
                llama_output,
                phi2_confidence,
                llama_confidence,
                judge_button
            ]
        )
        
        judge_button.click(
            get_judgment,
            inputs=[],
            outputs=[
                judge_evaluation,
                winner_display,
                recommendations_display,
                reprompt_button
            ]
        )
        
        reprompt_button.click(
            get_revised_solutions,
            inputs=[],
            outputs=[
                phi2_revised,
                llama_revised,
                phi2_revised_confidence,
                llama_revised_confidence
            ]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_advanced_polythink_interface()
    interface.launch(debug=True)