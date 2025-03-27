import os
import uuid
import asyncio
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
from typing import List, Dict, Any

# Use HF Token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set your Hugging Face token.")

class PolyThinkAgent:
    def __init__(self, model_name: str, model_path: str, is_gemma3: bool = False):
        """
        Initialize an agent with specific model capabilities
        """
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        self.is_gemma3 = is_gemma3
        
        try:
            if is_gemma3:
                # Special handling for Gemma 3 models
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    token=HF_TOKEN
                )
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_path,
                    token=HF_TOKEN,
                    device_map="auto"
                ).eval()
                self.tokenizer = self.processor.tokenizer
            else:
                # Standard handling for other models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    token=HF_TOKEN
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    token=HF_TOKEN
                )
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise
        
        # Agent-specific configuration
        self.specialization = self._determine_specialization()
    
    def _determine_specialization(self):
        """
        Assign a unique problem-solving specialization to the agent
        """
        specialization_map = {
            "Gemma 3 4b-it": "Advanced Analytical Problem Solving",
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
        if self.is_gemma3:
            # Gemma 3 specific processing
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": f"You are a specialized problem solver with expertise in {self.specialization}."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            inputs = self.processor(messages, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=500)
            solution = self.processor.decode(outputs[0], skip_special_tokens=True)
        else:
            # Standard processing for other models
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
            PolyThinkAgent("Gemma 3 4b-it", "google/gemma-3-4b-it", is_gemma3=True),  # Updated to Gemma 3
            PolyThinkAgent("Llama 3.2 1b", "meta-ai/llama-3.2-1b"),
            PolyThinkAgent("DeepSeek R1 1.5B", "deepseek-ai/deepseek-coder-1.5b-base")
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
    
    interface = gr.Blocks(theme=gr.themes.Default())  # Changed to Default theme for better visibility
    
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
                gemma_output = gr.Textbox(label="Gemma 3 4b-it Solution", interactive=False)  # Updated label
                llama_output = gr.Textbox(label="Llama 3.2 1b Solution", interactive=False)
                deepseek_output = gr.Textbox(label="DeepSeek Consensus", interactive=False)
            
            with gr.Row():
                gemma_confidence = gr.Slider(
                    minimum=0, maximum=100,
                    label="Gemma Confidence",
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
            
            gemma_sol = result['individual_solutions'][0]
            llama_sol = result['individual_solutions'][1]
            deepseek_sol = result['individual_solutions'][2]  # Added missing DeepSeek solution
            consensus = result['consensus']
            
            return [
                gemma_sol['solution'],
                llama_sol['solution'],
                deepseek_sol['solution'],  # Updated to use DeepSeek instead of consensus
                gemma_sol['confidence'],
                llama_sol['confidence'],
                consensus['consensus_confidence']
            ]
        
        solve_button.click(
            process_problem,
            inputs=problem_input,
            outputs=[
                gemma_output,
                llama_output,
                deepseek_output,
                gemma_confidence,
                llama_confidence,
                consensus_confidence
            ]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_advanced_polythink_interface()
    interface.launch(debug=True)