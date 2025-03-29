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
        
        # Extract the actual solution part, ensuring we don't include the prompt
        if "SOLUTION:" in solution:
            solution_parts = solution.split("SOLUTION:")
            # Take the last part after "SOLUTION:" to avoid including any prompt text
            solution_text = solution_parts[-1].strip()
        else:
            # Remove the prompt from the solution
            prompt_text = prompt.strip()
            if solution.startswith(prompt_text):
                solution_text = solution[len(prompt_text):].strip()
            else:
                solution_text = solution.strip()
        
        # Clean up repeating patterns (especially for Llama)
        solution_text = self._clean_repetition(solution_text)
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution_text,
            "confidence": self._calculate_confidence(solution_text)
        }
    
    def _clean_repetition(self, text: str) -> str:
        """
        Clean up repetitive patterns in the generated text
        """
        # Split by lines and remove duplicates while preserving order
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line)
        
        # Check for repeating patterns within the text
        cleaned_text = '\n'.join(unique_lines)
        
        # Handle specific patterns like "5+10 = 15" repeated
        if "=" in cleaned_text:
            equations = cleaned_text.split('\n')
            unique_equations = []
            eq_seen = set()
            for eq in equations:
                eq_stripped = eq.strip()
                if eq_stripped and eq_stripped not in eq_seen:
                    eq_seen.add(eq_stripped)
                    unique_equations.append(eq)
            cleaned_text = '\n'.join(unique_equations)
        
        return cleaned_text
    
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
        
        # Extract the revised solution part, ensuring we don't include the prompt
        if "REVISED SOLUTION:" in solution:
            solution_parts = solution.split("REVISED SOLUTION:")
            # Take the last part after "REVISED SOLUTION:" to avoid including any prompt text
            revised = solution_parts[-1].strip()
        else:
            # Remove the prompt from the solution
            prompt_text = prompt.strip()
            if solution.startswith(prompt_text):
                revised = solution[len(prompt_text):].strip()
            else:
                revised = solution.strip()
        
        # Clean up repeating patterns
        revised = self._clean_repetition(revised)
        
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
        # Use different prompts for different models to better match their capabilities
        if "Llama" in self.model_name:
            # Llama-specific prompt (simpler, more direct)
            return f"""Answer the following problem concisely.
Problem: {problem}
Provide a short, clear answer with minimal explanation.
Answer:"""
        else:
            # Phi-2 specific prompt
            return f"""Answer this problem:
{problem}
Your answer should be:
1. Start with the direct answer
2. Include only essential reasoning
3. Be under 50 words total
Answer:"""
    
    def _construct_judge_prompt(self, problem: str, solutions: List[Dict[str, Any]]) -> str:
        """
        Create a specialized prompt for judge agents
        """
        prompt = f"""You are evaluating solutions to this problem: {problem}

Solution from {solutions[0]['model_name']}:
{solutions[0]['solution']}

Solution from {solutions[1]['model_name']}:
{solutions[1]['solution']}

Analyze both solutions carefully. Determine which solution is better and explain why.
First, state if the solutions are correct or incorrect.
Then, select a winner (either {solutions[0]['model_name']} or {solutions[1]['model_name']}).
Finally, provide recommendations for improvement.

If the solutions significantly disagree, note "REPROMPT: YES" at the end.

Your evaluation:"""
        
        return prompt
    
    def _construct_reprompt(self, problem: str, solutions: List[Dict[str, Any]], judge_feedback: Dict[str, Any]) -> str:
        """
        Create a reprompt that includes other solutions and judge feedback
        """
        # Find other solutions (not from this agent)
        other_solutions = [s for s in solutions if s['agent_id'] != self.id]
        
        prompt = f"""Revise your solution to this problem: {problem}

Your previous solution:
{next((s['solution'] for s in solutions if s['agent_id'] == self.id), "No previous solution found")}

Other solution:
{other_solutions[0]['solution'] if other_solutions else "No other solutions available"}

Judge's feedback:
{judge_feedback.get('judgment', 'No specific feedback provided')}

Create an improved solution that addresses the feedback.
Revised solution:"""
        
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

def create_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    
    # State variables to store intermediate results
    current_problem = None
    current_solutions = None
    current_judgment = None
    
    # Process problem end-to-end with automatic progression
    async def process_problem(problem):
        global current_problem, current_solutions, current_judgment
        
        status_updates = []
        
        # Step 1: Reset state
        current_problem = problem
        current_solutions = None
        current_judgment = None
        
        # Update status
        status_updates.append("🔍 Analyzing problem...")
        yield status_updates[-1], "", "", 0, "", "", 0, "", "", gr.update(visible=False)
        
        # Step 2: Get initial solutions
        status_updates.append("⚙️ Phi-2 and Llama 3.2 are generating solutions...")
        yield status_updates[-1], "", "", 0, "", "", 0, "", "", gr.update(visible=False)
        
        try:
            current_solutions = await orchestrator.get_solver_solutions(problem)
            
            # Format solutions for display
            phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
            llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
            
            phi2_solution = phi2_sol['solution'] if phi2_sol else "Error retrieving solution"
            llama_solution = llama_sol['solution'] if llama_sol else "Error retrieving solution"
            phi2_confidence = phi2_sol['confidence'] if phi2_sol else 0
            llama_confidence = llama_sol['confidence'] if llama_sol else 0
            
            status_updates.append("✅ Initial solutions generated!")
            yield (
                status_updates[-1], 
                phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
                llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
                "", "", gr.update(visible=False)
            )
            
            # Step 3: Get judge evaluation automatically
            status_updates.append("⚖️ DeepSeek judge is evaluating solutions...")
            yield status_updates[-1], phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "", llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "", "", "", gr.update(visible=False)
            
            current_judgment = await orchestrator.get_judge_evaluation(problem, current_solutions)
            
            # Check if reprompt is needed and show button if so
            reprompt_needed = current_judgment.get('reprompt_needed', False)
            
            status_updates.append("✅ Judge evaluation complete!")
            yield (
                status_updates[-1],
                phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
                llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
                current_judgment['judgment'],
                current_judgment.get('winner', "No clear winner"),
                gr.update(visible=reprompt_needed)
            )
            
            # Step 4: If reprompt is needed, proceed automatically
            if reprompt_needed:
                status_updates.append("🔄 Getting revised solutions based on judge feedback...")
                yield (
                    status_updates[-1],
                    phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
                    llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
                    current_judgment['judgment'],
                    current_judgment.get('winner', "No clear winner"),
                    gr.update(visible=False)
                )
                
                revised_solutions = await orchestrator.get_revised_solutions(
                    problem, current_solutions, current_judgment
                )
                
                # Update the current solutions with revised ones
                current_solutions = revised_solutions
                
                # Get updated phi2 and llama solutions
                phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
                llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
                
                phi2_solution = phi2_sol['solution'] if phi2_sol else "Error retrieving solution"
                llama_solution = llama_sol['solution'] if llama_sol else "Error retrieving solution"
                phi2_confidence = phi2_sol['confidence'] if phi2_sol else 0
                llama_confidence = llama_sol['confidence'] if llama_sol else 0
                
                # Get final judgment
                current_judgment = await orchestrator.get_judge_evaluation(problem, current_solutions)
                
                status_updates.append("✅ Process complete! Final solutions and evaluation are ready.")
                yield (
                    "\n".join(status_updates),
                    phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
                    llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
                    current_judgment['judgment'],
                    current_judgment.get('winner', "No clear winner"),
                    gr.update(visible=False)
                )
            else:
                status_updates.append("✅ Process complete! Solutions and evaluation are ready.")
                yield (
                    "\n".join(status_updates),
                    phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
                    llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
                    current_judgment['judgment'],
                    current_judgment.get('winner', "No clear winner"),
                    gr.update(visible=False)
                )
                
        except Exception as e:
            error_message = f"Error: {str(e)}"
            status_updates.append(f"❌ {error_message}")
            yield (
                "\n".join(status_updates),
                f"Error: {str(e)}", 0, "",
                f"Error: {str(e)}", 0, "",
                f"Error occurred during processing: {str(e)}",
                "No winner due to error",
                gr.update(visible=False)
            )
    
    # Manual reprompt function as a backup
    async def manual_reprompt():
        global current_problem, current_solutions, current_judgment
        
        if not current_problem or not current_solutions or not current_judgment:
            return "No problem, solutions or judgment available", "", 0, "", "", 0, "", "", ""
        
        status_updates = ["🔄 Manually getting revised solutions..."]
        yield status_updates[-1], "", 0, "", "", 0, "", "", "", gr.update(visible=False)
        
        # Get revised solutions
        revised_solutions = await orchestrator.get_revised_solutions(
            current_problem, current_solutions, current_judgment
        )
        
        # Update the current solutions with revised ones
        current_solutions = revised_solutions
        
        # Get updated phi2 and llama solutions
        phi2_sol = next((s for s in current_solutions if s['model_name'] == "Phi-2"), None)
        llama_sol = next((s for s in current_solutions if s['model_name'] == "Llama 3.2 1b"), None)
        
        phi2_solution = phi2_sol['solution'] if phi2_sol else "Error retrieving solution"
        llama_solution = llama_sol['solution'] if llama_sol else "Error retrieving solution"
        phi2_confidence = phi2_sol['confidence'] if phi2_sol else 0
        llama_confidence = llama_sol['confidence'] if llama_sol else 0
        
        status_updates.append("⚖️ Getting final judgment...")
        yield (
            "\n".join(status_updates),
            phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
            llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
            current_judgment['judgment'],
            current_judgment.get('winner', "No clear winner"),
            gr.update(visible=False)
        )
        
        # Get final judgment
        current_judgment = await orchestrator.get_judge_evaluation(current_problem, current_solutions)
        
        status_updates.append("✅ Manual reprompt complete!")
        yield (
            "\n".join(status_updates),
            phi2_solution, phi2_confidence, phi2_sol['specialization'] if phi2_sol else "",
            llama_solution, llama_confidence, llama_sol['specialization'] if llama_sol else "",
            current_judgment['judgment'],
            current_judgment.get('winner', "No clear winner"),
            gr.update(visible=False)
        )
    
    # Custom CSS
    custom_css = """
    .status-bar {
        background-color: #f0f7ff;
        border-left: 4px solid #3B82F6;
        padding: 12px 15px;
        margin-bottom: 20px;
        border-radius: 5px;
        font-family: "Courier New", monospace;
        white-space: pre-wrap;
        font-weight: 500;
    }
    
    .agent-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .agent-container:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .agent-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .phi-container {
        border-left: 4px solid #4F46E5;
    }
    
    .llama-container {
        border-left: 4px solid #EC4899;
    }
    
    .judge-container {
        border-left: 4px solid #10B981;
        background-color: #f8fafc;
    }
    
    .confidence-meter {
        height: 8px;
        border-radius: 4px;
        background-color: #e5e7eb;
        margin-top: 5px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background-color: #4F46E5;
    }
    
    .winner-badge {
        display: inline-block;
        background-color: #FBBF24;
        color: #7C2D12;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8em;
        margin-left: 10px;
    }
    
    .title-container {
        background: linear-gradient(90deg, #4F46E5 0%, #EC4899 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    
    .title-container h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .title-container p {
        margin-top: 10px;
        opacity: 0.9;
    }
    """
    
    # Create the interface
    with gr.Blocks(title="PolyThink Multi-Agent Problem Solver", css=custom_css) as demo:
        # Custom header with gradients
        gr.HTML("""
        <div class="title-container">
            <h1>🧠 PolyThink: Multi-Agent Problem Solving</h1>
            <p>Watch multiple AI models collaborate to solve problems with a judge model evaluating their answers</p>
        </div>
        """)
        
        # Status bar at the top (more prominent)
        status_text = gr.Markdown("Ready to solve problems...", elem_classes=["status-bar"])
        
        with gr.Row():
            with gr.Column(scale=4):
                problem_input = gr.Textbox(
                    label="Problem to Solve",
                    placeholder="Enter a problem or question here. For example: 'What is 5+10?' or 'Explain why the sky is blue in simple terms'",
                    lines=3
                )
            with gr.Column(scale=1):
                solve_button = gr.Button("Solve Problem", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("""<div class="agent-container phi-container">
                    <div class="agent-header">
                        <h3>Phi-2 Solution</h3>
                        <div class="model-badge" id="phi-specialization"></div>
                    </div>
                """)
                phi2_specialization = gr.Textbox(label="Specialization", elem_id="phi-specialization")
                phi2_solution = gr.Textbox(label="Solution", lines=5)
                phi2_confidence = gr.Number(label="Confidence")
                gr.HTML("</div>")  # Close container
            
            with gr.Column():
                gr.HTML("""<div class="agent-container llama-container">
                    <div class="agent-header">
                        <h3>Llama 3.2 1B Solution</h3>
                        <div class="model-badge" id="llama-specialization"></div>
                    </div>
                """)
                llama_specialization = gr.Textbox(label="Specialization", elem_id="llama-specialization")
                llama_solution = gr.Textbox(label="Solution", lines=5)
                llama_confidence = gr.Number(label="Confidence")
                gr.HTML("</div>")  # Close container
        
        with gr.Row():
            with gr.Column():
                gr.HTML("""<div class="agent-container judge-container">
                    <div class="agent-header">
                        <h3>DeepSeek Judge Evaluation