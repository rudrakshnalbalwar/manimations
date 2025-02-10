import os
import subprocess
import time
import groq
from typing import Optional, Dict, Any

API_KEY = os.getenv("GROQ_API_KEY") or input("Please enter your Groq API key: ").strip()
client = groq.Groq(api_key=API_KEY)

def get_user_input() -> str:
    """Get user input for a math problem."""
    print("\nEnter the math problem (press Enter twice to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except KeyboardInterrupt:
            print("\nInput cancelled.")
            return ""
    return "\n".join(lines)

def get_problem_type(problem: str) -> str:
    """Determine the type of mathematical problem."""
    categories = ["ALGEBRAIC", "CALCULUS", "TRIGONOMETRY", "GEOMETRY", "MATRIX", "COMPLEX", "SEQUENCE", "PROBABILITY"]
    return next((cat for cat in categories if cat in problem.upper()), "ALGEBRAIC")

def generate_math_solution(problem: str) -> Optional[str]:
    """Generate a step-by-step solution emphasizing visualization."""
    problem_type = get_problem_type(problem)
    response = request_with_retry({
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": f"Generate a concise step-by-step solution for a {problem_type} problem, with each step being visualizable. And if possible try to visualise answer using graphs or 2d elements like it should look like you are undersatnding the solution by objects not by steps"},
            {"role": "user", "content": problem}
        ],
        "temperature": 0.2
    })
    if response and response.choices:
        return response.choices[0].message.content.strip()
    return None

def generate_manim_code(math_problem: str, math_solution: str) -> Optional[str]:
    """Generate structured Manim code with smooth step transitions."""
    steps = [step for step in math_solution.split('\n') if step.strip()]
    time_per_step = 25 / max(len(steps), 1)

    manim_code = f"""from manim import *

class MathSolution(Scene):
    def construct(self):
        title = Text("Problem: {math_problem}", font_size=36).to_edge(UP)
        self.play(Write(title), run_time=2)
        self.wait(1)
"""
    
    for i, step in enumerate(steps, 1):
        fade_time = time_per_step * 0.4
        display_time = time_per_step * 0.6
        manim_code += f"""
        step{i} = Text("{step}", font_size=28).move_to(ORIGIN)
        self.play(Write(step{i}), run_time={fade_time:.1f})
        self.wait({display_time:.1f})
        self.play(FadeOut(step{i}), run_time={fade_time:.1f})
"""
    
    manim_code += """
        self.wait(3)  # Pause at the end to review the solution
"""
    return manim_code

def request_with_retry(payload: Dict[str, Any], retries: int = 3, wait: int = 5) -> Optional[Any]:
    """Retry API requests in case of failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            return response
        except Exception as e:
            if attempt < retries - 1:
                print(f"API request failed ({e}). Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Failed after {retries} attempts: {e}")
    return None

def save_manim_script(code: str, output_path: str):
    """Save the generated Manim script."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(code)

def run_manim(script_path: str, output_dir: str) -> bool:
    """Run the Manim script to generate the animation."""
    try:
        command = ["manim", "-pqh", script_path, "MathSolution", "--media_dir", output_dir]
        print("\nGenerating animation... Please wait.")
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating animation: {e}")
        return False

def main():
    """Main function to handle user input and generate animations."""
    try:
        while True:
            print("\nMath Animation Generator")
            math_problem = get_user_input()
            if not math_problem:
                print("No math problem provided.")
                continue
            
            print("\nGenerating step-by-step solution...")
            math_solution = generate_math_solution(math_problem)
            if not math_solution:
                print("Failed to generate a valid solution.")
                continue
            
            print("\nGenerating Manim animation code...")
            manim_code = generate_manim_code(math_problem, math_solution)
            if not manim_code:
                print("Failed to generate valid Manim code.")
                continue
            
            output_dir = "./manim_output"
            script_path = os.path.join(output_dir, "manim_script.py")
            save_manim_script(manim_code, script_path)
            
            if run_manim(script_path, output_dir):
                print(f"\nAnimation generated successfully at {output_dir}/videos/MathSolution.mp4")
            else:
                print("Failed to generate animation.")
            
            if input("\nCreate another animation? (y/n): ").lower() != 'y':
                break
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        exit(0)

if __name__ == "__main__":
    main()
