import os
import subprocess
import groq
from typing import List
from dataclasses import dataclass
from manim import *

@dataclass
class MathSolution:
    problem: str
    steps: List[str]
    problem_type: str

class MathAnimationGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or input("Please enter your Groq API key: ").strip()
        self.client = groq.Groq(api_key=self.api_key)

    def get_problem_input(self) -> str:
        print("\nEnter the math problem (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)

    def classify_problem(self, problem: str) -> str:
        categories = {
            "ALGEBRAIC": ["equation", "solve", "polynomial"],
            "CALCULUS": ["derivative", "integral", "differentiate"],
            "TRIGONOMETRY": ["sin", "cos", "tan"],
            "GEOMETRY": ["triangle", "circle", "area"],
            "PROBABILITY": ["probability", "chance"]
        }
        problem_lower = problem.lower()
        for category, keywords in categories.items():
            if any(keyword in problem_lower for keyword in keywords):
                return category
        return "GENERAL"

    def generate_solution(self, problem: str) -> MathSolution:
        problem_type = self.classify_problem(problem)
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": f"Generate a step-by-step solution for this {problem_type} problem."},
                    {"role": "user", "content": problem}
                ],
                temperature=0.2
            )
            solution_text = response.choices[0].message.content.strip()
            steps = [step.strip() for step in solution_text.split('\n') if step.strip()]
            return MathSolution(problem, steps, problem_type)
        except Exception as e:
            print(f"Error generating solution: {e}")
            return MathSolution(problem, ["Error: Unable to generate solution"], problem_type)

    def generate_manim_script(self, solution: MathSolution) -> str:
        steps_repr = ', '.join([f'"{step}"' for step in solution.steps])
        return f'''
from manim import *

class MathSolutionScene(Scene):
    def construct(self):
        self.setup_layout()
        self.show_problem("{solution.problem}")
        self.animate_solution([{steps_repr}])
    
    def setup_layout(self):
        left_panel = Rectangle(width=config.frame_width/2, height=config.frame_height, fill_color="#2C3E50", fill_opacity=0.1, stroke_width=0).to_edge(LEFT, buff=0)
        right_panel = Rectangle(width=config.frame_width/2, height=config.frame_height, fill_color="#34495E", fill_opacity=0.1, stroke_width=0).to_edge(RIGHT, buff=0)
        self.play(FadeIn(left_panel), FadeIn(right_panel))

    def show_problem(self, problem_text):
        title = Text("Mathematical Solution", font_size=36).to_edge(UP, buff=0.5)
        problem = Text(problem_text, font_size=24).next_to(title, DOWN, buff=0.5)
        self.play(Write(title), Write(problem))

    def create_visualization(self, step_number):
        axes = Axes(x_range=[-5, 5, 1], y_range=[-5, 5, 1], tips=True, axis_config={{"color": BLUE}}).scale(0.7)
        axes.to_edge(RIGHT, buff=1.0)
        return axes

    def animate_solution(self, steps):
        prev_text = None
        prev_vis = None
        for step_index, step in enumerate(steps, start=1):
            step_text = Text(f"Step {{step_index}}: {{step}}", font_size=24).set_width(config.frame_width/2 - 1).to_edge(LEFT, buff=0.5).shift(UP * (1.5 - step_index * 0.8))
            vis = self.create_visualization(step_index)
            if prev_text:
                self.play(FadeOut(prev_text))
            if prev_vis:
                self.play(FadeOut(prev_vis))
            self.play(Write(step_text), Create(vis))
            prev_text = step_text
            prev_vis = vis
            self.wait(2)
        self.wait(3)
'''

    def save_script(self, script: str, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script)

    def render_animation(self, script_path: str, output_dir: str) -> bool:
        try:
            subprocess.run(["manim", "-pqh", script_path, "MathSolutionScene", "--media_dir", output_dir], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Animation rendering failed: {e}")
            return False

    def generate_animation(self):
        problem = self.get_problem_input()
        if not problem:
            print("No problem provided.")
            return False
        print("\nGenerating solution...")
        solution = self.generate_solution(problem)
        if not solution.steps:
            print("Failed to generate solution.")
            return False
        print("Creating animation script...")
        script = self.generate_manim_script(solution)
        output_dir = "./math_animations"
        script_path = os.path.join(output_dir, "math_scene.py")
        self.save_script(script, script_path)
        print("Rendering animation...")
        if self.render_animation(script_path, output_dir):
            print(f"\nAnimation saved to {output_dir}/videos/MathSolutionScene.mp4")
            return True
        print("Failed to render animation.")
        return False

def main():
    generator = MathAnimationGenerator()
    while True:
        generator.generate_animation()
        if input("\nCreate another animation? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
