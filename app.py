import os
import subprocess
import groq
from typing import List, Optional
import re
from dataclasses import dataclass
import logging
import tempfile
import shutil
from manim import *
import time
import numpy as np

@dataclass
class MathSolution:
    problem: str
    steps: List[str]
    problem_type: str
    visualization_details: List[dict]

class MathAnimationGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or input("Please enter your Groq API key: ").strip()
        self.client = groq.Groq(api_key=self.api_key)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Set up directories
        self.media_dir = "media"
        self.output_dir = "output_videos"
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_solution(self, problem: str) -> MathSolution:
        """Generate structured math solution with visualization plan"""
        problem_type = self.classify_problem(problem)
        
        prompt = f"""Generate a detailed step-by-step solution with visualization instructions for:
        {problem}
        Please provide:
        - Numbered steps explaining the solution
        - An equation for each key step (if applicable)
        - A visualization description for each step
        
        Format each step like:
        Step X: [Detailed explanation]
        Equation: [Relevant equation if applicable]
        Visualization: [Description of visual representation]"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": "You're a math tutor creating animated solutions."},
                         {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            return self.parse_solution(response.choices[0].message.content.strip(), problem, problem_type)
        
        except Exception as e:
            self.logger.error(f"Solution generation error: {e}")
            return MathSolution(problem, ["Error generating solution"], problem_type, [])

    def parse_solution(self, solution_text: str, problem: str, problem_type: str) -> MathSolution:
        """Parse LLM response into structured format"""
        steps = []
        visualizations = []
        
        current_step = {}
        for line in solution_text.split('\n'):
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step.get('text', ''))
                    visualizations.append(current_step)
                current_step = {'text': line.strip()}
            elif line.startswith('Equation:'):
                current_step['equation'] = line.replace('Equation:', '').strip()
            elif line.startswith('Visualization:'):
                current_step['visualization'] = line.replace('Visualization:', '').strip()
        
        if current_step:
            steps.append(current_step.get('text', ''))
            visualizations.append(current_step)
            
        return MathSolution(problem, steps, problem_type, visualizations)

    def generate_manim_script(self, solution: MathSolution) -> str:
        """Generate robust Manim script with dynamic layout"""
        # Convert solution data to strings to avoid serialization issues
        problem_tex = self.escape_latex(solution.problem)
        steps_data = []
        
        for i, (step, detail) in enumerate(zip(
            solution.steps or ['No steps available'], 
            solution.visualization_details or [{}]
        )):
            step_dict = {
                'index': i + 1,
                'text': self.escape_latex(step),
                'has_equation': False,
                'equation': '',
                'visualization': ''
            }
            
            if detail and 'equation' in detail:
                step_dict['has_equation'] = True
                step_dict['equation'] = self.escape_latex(detail['equation'])
                
            if detail and 'visualization' in detail:
                step_dict['visualization'] = detail.get('visualization', '')
                
            steps_data.append(step_dict)
        
        # Convert windows path to use forward slashes to avoid escape issues
        media_dir_path = os.path.abspath(self.media_dir).replace('\\', '/')
        
        # Generate script with proper formatting
        script = f'''from manim import *
import numpy as np

config.media_dir = "{media_dir_path}"

class MathSolutionAnimation(Scene):
    def construct(self):
        # Setup scene
        self.camera.background_color = WHITE
        
        # Problem display
        problem_text = Tex(r"\\textbf{{Problem:}} {problem_tex}", 
                         font_size=36, color=BLACK)
        problem_text.to_edge(UP, buff=0.5)
        self.play(Write(problem_text))
        self.wait(1)
        
        # Create step container
        step_group = VGroup(problem_text)
        
'''
        
        # Generate dynamic step code
        for step in steps_data:
            script += f'''
        # Step {step['index']}
        step_text = Tex(r"\\textbf{{Step {step['index']}:}} {step['text']}", 
                      font_size=28, color=BLACK)
        step_text.next_to(step_group, DOWN, buff=0.7, aligned_edge=LEFT)
        
        # Create equation if present
        equation_mobj = None
'''
            
            if step['has_equation']:
                script += f'''
        try:
            equation_mobj = MathTex(r"{step['equation']}", 
                                  color=BLUE, font_size=32)
            equation_mobj.next_to(step_text, DOWN, buff=0.4)
        except Exception as e:
            print(f"Equation error: {{e}}")
'''

            script += f'''
        # Create visualization
        viz_mobj = self.create_visualization("{step['visualization']}")
        if viz_mobj:
            viz_mobj.next_to(equation_mobj or step_text, DOWN, buff=0.7)
        
        # Animate elements
        animations = [
            Write(step_text),
            *([Create(equation_mobj)] if equation_mobj else []),
            *([Create(viz_mobj)] if viz_mobj else [])
        ]
        self.play(
            AnimationGroup(*animations, lag_ratio=0.3),
            run_time=2
        )
        self.wait(1)
        
        # Update step group
        new_group = VGroup(step_text)
        if equation_mobj:
            new_group.add(equation_mobj)
        if viz_mobj:
            new_group.add(viz_mobj)
        step_group.add(new_group)
        
        # Shift elements up if needed
        if step_group.get_bottom()[1] < -3.5:
            self.play(step_group.animate.shift(UP*1.5))
'''

        script += '''
        self.wait(2)
    
    def create_visualization(self, viz_type: str) -> VMobject:
        """Create visualization object based on type"""
        viz_type = viz_type.lower() if viz_type else ''
        
        if 'graph' in viz_type or 'function' in viz_type:
            axes = Axes(
                x_range=[-5,5], y_range=[-5,5],
                axis_config={"color": BLACK},
                x_axis_config={"numbers_to_include": np.arange(-5,6,1)},
                y_axis_config={"numbers_to_include": np.arange(-5,6,1)}
            )
            
            # For the specific quadratic function
            if 'x^2' in viz_type and '-4x' in viz_type and '+3' in viz_type:
                graph = axes.plot(lambda x: x**2 - 4*x + 3, color=RED)
                
                # Draw vertex
                vertex_x = 2
                vertex_y = -1
                vertex_dot = Dot(axes.c2p(vertex_x, vertex_y), color=GREEN)
                vertex_label = Tex("Vertex (2, -1)", font_size=24, color=GREEN).next_to(vertex_dot, UP)
                
                # Draw x-intercepts
                x1, x2 = 1, 3
                x1_dot = Dot(axes.c2p(x1, 0), color=BLUE)
                x2_dot = Dot(axes.c2p(x2, 0), color=BLUE)
                x1_label = Tex("$x=1$", font_size=24, color=BLUE).next_to(x1_dot, DOWN)
                x2_label = Tex("$x=3$", font_size=24, color=BLUE).next_to(x2_dot, DOWN)
                
                return VGroup(axes, graph, vertex_dot, vertex_label, 
                              x1_dot, x2_dot, x1_label, x2_label)
            else:
                # Generic quadratic function
                graph = axes.plot(lambda x: x**2, color=RED)
                return VGroup(axes, graph)
            
        elif '3d' in viz_type:
            return ThreeDAxes()
            
        elif 'geometry' in viz_type:
            circle = Circle(radius=1.5, color=GREEN)
            triangle = Polygon([-1,0,0], [1,0,0], [0,1,0], color=BLUE)
            return VGroup(circle, triangle)
            
        # Return empty VMobject if no visualization matches
        return VGroup()
'''
        return script

    def escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            '\\': r'\\',
            '{': r'\{',
            '}': r'\}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '±': r'\pm',
            '√': r'\sqrt',
            'θ': r'\theta',
            'π': r'\pi',
            '→': r'\rightarrow'
        }
        for char, escape in replacements.items():
            text = text.replace(char, escape)
        return text

    def generate_and_render(self, problem: str) -> tuple[bool, str]:
        """Execute full generation and rendering pipeline"""
        solution = self.generate_solution(problem)
        if not solution.steps or "Error" in solution.steps[0]:
            return False, "Solution generation failed"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                script_path = os.path.join(tmpdir, "animation.py")
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(self.generate_manim_script(solution))
                
                final_output = os.path.join(self.output_dir, f"solution_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                
                # Run manim with standard quality
                result = subprocess.run(
                    ["manim", "-pql", script_path, "MathSolutionAnimation", "-o", "math_solution"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode != 0:
                    self.logger.error(f"Rendering failed: {result.stderr}")
                    return False, "Rendering error"
                
                # Look for the generated file in the media directory
                media_dir = os.path.join(self.media_dir, "videos", "animation")
                if os.path.exists(media_dir):
                    media_files = os.listdir(media_dir)
                    if media_files:
                        latest_file = sorted(media_files)[-1]
                        source_file = os.path.join(media_dir, latest_file)
                        shutil.copy(source_file, final_output)
                        return True, final_output
                
                return False, "No output file generated"
                
            except Exception as e:
                self.logger.error(f"Animation error: {e}")
                return False, str(e)

    def classify_problem(self, problem: str) -> str:
        """Classify the type of math problem"""
        categories = {
            "ALGEBRA": ["equation", "solve", "polynomial", "variable"],
            "CALCULUS": ["derivative", "integral", "limit"],
            "TRIGONOMETRY": ["sin", "cos", "tan", "angle"],
            "GEOMETRY": ["triangle", "circle", "area"],
            "3D_GEOMETRY": ["3d", "volume", "surface"],
            "PROBABILITY": ["probability", "chance"],
            "FUNCTION": ["graph", "function", "plot", "curve"]
        }
        
        problem_lower = problem.lower()
        for category, keywords in categories.items():
            if any(keyword in problem_lower for keyword in keywords):
                return category
        return "GENERAL"

def main():
    generator = MathAnimationGenerator()
    
    while True:
        # Get problem input
        print("\nEnter a math problem (press Enter twice to finish):")
        problem_lines = []
        while True:
            line = input()
            if not line:
                break
            problem_lines.append(line)
        problem = " ".join(problem_lines)
        
        if not problem:
            break
        
        # Generate and render animation
        success, result = generator.generate_and_render(problem)
        
        print(f"\nResult: {result}")
        
        # Ask to continue
        if input("\nCreate another animation? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()