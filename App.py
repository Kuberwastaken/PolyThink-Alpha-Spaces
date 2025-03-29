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
    def __init__(self, model_name: str, model_path: str):
        """
        Initialize an agent with specific model capabilities
        """
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        
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
        specialization_map = {
            "Phi-2": "Advanced Analytical Problem Solving",
            "Llama 3.2 1b": "Creative Solution Generation",
            "DeepSeek R1 1.5B": "Consensus and Reasoning"
        }
        return specialization_map.get(self.model_name, "General Problem Solver")
    
    async def solve_problem(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Asynchronous problem-solving method with advanced context handling
        """
        # Construct specialized prompt
        prompt = self._construct_prompt(problem, context)
        
        # Generate solution
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500, num_return_sequences=1)
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "specialization": self.specialization,
            "solution": solution,
            "confidence": self._calculate_confidence(solution)
        }
    
    def _construct_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a context-aware, specialized prompt
        """
        base_prompt = f"""
        ADVANCED PROBLEM-SOLVING PROTOCOL
        Specialization: {self.specialization}
        
        PROBLEM: {problem}
        
        COMPREHENSIVE SOLVING REQUIREMENTS:
        - Provide COMPLETE step-by-step solution
        - Demonstrate FULL computational reasoning
        - Highlight potential alternative approaches
        - Maintain ABSOLUTE mathematical precision
        """
        
        if context and 'previous_solutions' in context:
            base_prompt += "\n\nCONTEXT OF PREVIOUS SOLUTIONS:\n"
            for sol in context['previous_solutions']:
                base_prompt += f"- {sol}\n"
        
        return base_prompt
    
    def _calculate_confidence(self, solution: str) -> float:
        """
        Calculate solution confidence based on complexity and detail
        """
        word_count = len(solution.split())
        complexity_factor = min(word_count / 50, 1.0)
        return complexity_factor * 100

class PolyThinkAgentOrchestrator:
    def __init__(self):
        """
        Initialize multi-agent problem-solving system
        """
        self.agents = [
            PolyThinkAgent("Phi-2", "microsoft/phi-2"),
            PolyThinkAgent("Llama 3.2 1b", "meta-llama/llama-3.2-1b"),
            PolyThinkAgent("DeepSeek R1 1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        ]
    
    async def solve_problem_multi_agent(self, problem: str) -> Dict[str, Any]:
        """
        Parallel problem-solving across multiple agents
        """
        agent_tasks = [agent.solve_problem(problem) for agent in self.agents]
        solutions = await asyncio.gather(*agent_tasks)
        
        consensus_solution = self._determine_consensus(solutions)
        
        return {
            "individual_solutions": solutions,
            "consensus": consensus_solution
        }
    
    def _determine_consensus(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advanced consensus mechanism
        """
        sorted_solutions = sorted(solutions, key=lambda x: x['confidence'], reverse=True)
        
        consensus_confidence = sum(sol['confidence'] for sol in sorted_solutions) / len(sorted_solutions)
        
        return {
            "recommended_solution": sorted_solutions[0],
            "all_solutions": sorted_solutions,
            "consensus_confidence": consensus_confidence
        }

def create_advanced_polythink_interface():
    orchestrator = PolyThinkAgentOrchestrator()
    
    def solve_problem(problem):
        return asyncio.run(orchestrator.solve_problem_multi_agent(problem))
    
    interface = gr.Blocks(theme=gr.themes.Default())
    
    with interface:
        gr.Markdown("# PolyThink: Multi-Agent Intelligent Problem Solver")
        
        with gr.Row():
            problem_input = gr.Textbox(
                label="Enter your problem",
                lines=4,
                placeholder="Paste your homework problem here..."
            )
            solve_button = gr.Button("Solve Problem", variant="primary")
        
        with gr.Accordion("Agent Solutions", open=True):
            with gr.Row():
                phi2_output = gr.Textbox(label="Phi-2 Solution", interactive=False)
                llama_output = gr.Textbox(label="Llama 3.2 1b Solution", interactive=False)
                deepseek_output = gr.Textbox(label="DeepSeek Solution", interactive=False)
            
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
                consensus_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Overall Consensus",
                    interactive=False
                )
        
        with gr.Accordion("Advanced Features", open=False):
            reprompt_button = gr.Button("Reprompt Disagreeing Agents")
            detailed_reasoning = gr.Textbox(
                label="Detailed Reasoning",
                lines=3,
                interactive=False
            )
        
        def process_problem(problem):
            result = solve_problem(problem)
            
            phi2_sol = result['individual_solutions'][0]
            llama_sol = result['individual_solutions'][1]
            deepseek_sol = result['individual_solutions'][2]
            consensus = result['consensus']
            
            return [
                phi2_sol['solution'],
                llama_sol['solution'],
                deepseek_sol['solution'],
                phi2_sol['confidence'],
                llama_sol['confidence'],
                consensus['consensus_confidence']
            ]
        
        solve_button.click(
            process_problem,
            inputs=problem_input,
            outputs=[
                phi2_output,
                llama_output,
                deepseek_output,
                phi2_confidence,
                llama_confidence,
                consensus_confidence
            ]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_advanced_polythink_interface()
    interface.launch(debug=True)