import os
import subprocess
import groq
from typing import List
from dataclasses import dataclass
from manim import *
from multiprocessing import Pool

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
                    {"role": "system", "content": f"Generate a concise step-by-step solution for a {problem_type} problem, with each step being visualizable. And if possible try to visualise answer using graphs or 2d elements like it should look like you are undersatnding the solution by objects not by steps it means you have to solve the question using elements like graphs, digrams etc not by displaying just steps."},
                    {"role": "user", "content": problem}
                ],
                temperature=0.7
            )
            solution_text = response.choices[0].message.content.strip()
            steps = [step.strip() for step in solution_text.split('\n') if step.strip()]
            return MathSolution(problem, steps, problem_type)
        except Exception as e:
            print(f"Error generating solution: {e}")
            return MathSolution(problem, ["Error: Unable to generate solution"], problem_type)

    def generate_manim_script(self, solution: MathSolution) -> str:
        # Escape curly braces in the problem text for f-string
        escaped_problem = (solution.problem
            .replace('"', '\\"')
            .replace('{', '{{')
            .replace('}', '}}')
            .replace('\\', '\\\\'))        
        formatted_steps = []
        for step in solution.steps:
            cleaned_step = (step.strip()
                .replace('"', '\\"')
                .replace('{', '{{')
                .replace('}', '}}')
                .replace('\\', '\\\\'))
            formatted_steps.append(f'"{cleaned_step}"')
        # Generate the steps representation with proper escaping
        steps_repr = ',\n '.join(formatted_steps)
        
        return f'''
from manim import *
import re

class MathSolutionScene(Scene):
    def construct(self):
        self.setup_layout()
        self.problem_text = "{escaped_problem}"
        self.solution_steps = [{steps_repr}]
        self.axes = None
        
        self.show_problem()
        self.determine_problem_type()
        self.animate_solution()
    
    def setup_layout(self):
        # Create a title bar at the top
        title_bar = Rectangle(width=config.frame_width, height=1.0, 
                             fill_color="#2C3E50", fill_opacity=1, stroke_width=0).to_edge(UP, buff=0)
        self.title_text = Text("Mathematical Solution", font_size=36, color=WHITE)
        self.title_text.move_to(title_bar.get_center())
        
        # Split the remaining area into left (text) and right (visualization) panels
        self.left_panel = Rectangle(width=config.frame_width/2, height=config.frame_height-1, 
                                  fill_color="#ECF0F1", fill_opacity=0.2, stroke_width=0)
        self.left_panel.next_to(title_bar, DOWN, buff=0).to_edge(LEFT, buff=0)
        
        self.right_panel = Rectangle(width=config.frame_width/2, height=config.frame_height-1, 
                                   fill_color="#F0F3F4", fill_opacity=0.2, stroke_width=0)
        self.right_panel.next_to(title_bar, DOWN, buff=0).to_edge(RIGHT, buff=0)
        
        self.play(
            FadeIn(title_bar),
            Write(self.title_text),
            FadeIn(self.left_panel),
            FadeIn(self.right_panel)
        )
    
    def show_problem(self):
        # Use MathTex for proper mathematical formatting
        self.problem = MathTex(self.format_math(self.problem_text), font_size=28)
        self.problem.set_width(self.left_panel.width - 1)
        self.problem.next_to(self.title_text, DOWN, buff=0.8).to_edge(LEFT, buff=1.0)
        
        self.play(Write(self.problem))
        self.wait(1)
    
    def determine_problem_type(self):
        """Analyze the problem text to determine what type of visualization to use"""
        problem_text = self.problem_text.lower()
        
        if any(word in problem_text for word in ["graph", "plot", "function", "equation", "f(x)"]):
            self.problem_type = "function"
        elif any(word in problem_text for word in ["triangle", "circle", "rectangle", "polygon", "angle"]):
            self.problem_type = "geometry"
        elif any(word in problem_text for word in ["vector", "matrix", "linear", "transform"]):
            self.problem_type = "linear_algebra"
        elif any(word in problem_text for word in ["probability", "distribution", "random", "expected"]):
            self.problem_type = "probability"
        else:
            self.problem_type = "algebraic"
    
    def animate_solution(self):
        prev_step_text = None
        current_vis_objects = []
        exclude_objects = []
        self.axes = None 
        self.x_label = None
        self.y_label = None
        self.problem_type = self.determine_problem_type()
        
        # Create coordinate system on the right panel if needed
        if self.problem_type in ["function", "geometry", "linear_algebra"]:
            self.axes = Axes(
                x_range=[-5, 5, 1], 
                y_range=[-5, 5, 1], 
                axis_config={{"color": BLUE}},
                x_length=self.right_panel.width - 2,
                y_length=self.right_panel.height - 4
            )
            self.axes.move_to(self.right_panel.get_center())
            
            # Add coordinate labels
            self.x_label = MathTex("x", font_size=24).next_to(self.axes.x_axis.get_end(), RIGHT)
            self.y_label = MathTex("y", font_size=24).next_to(self.axes.y_axis.get_end(), UP)
            
            self.play(Create(self.axes), FadeIn(self.x_label), FadeIn(self.y_label))
            current_vis_objects.extend([self.axes, self.x_label, self.y_label])
        
        for step_index, step in enumerate(self.solution_steps, start=1):
            # Format step text with MathTex for proper mathematical formatting
            step_label = Text(f"Step {{step_index}}:", font_size=24, color=BLUE_D)
            step_label.to_edge(LEFT, buff=1.0).shift(UP * (1.5 - step_index * 1.2))
            
            step_content = MathTex(self.format_math(step), font_size=22)
            step_content.set_width(self.left_panel.width - 2)
            step_content.next_to(step_label, RIGHT, buff=0.2)
            step_content.align_to(step_label, UP)
            
            # Create the appropriate visualization for this step
            new_vis_objects = self.create_visualization_for_step(step, step_index)
            
            # Animate the transition
            if prev_step_text:
                self.play(
                    FadeIn(step_label),
                    FadeIn(step_content),
                    FadeOut(prev_step_text[0]),
                    FadeOut(prev_step_text[1]),
                    *[FadeOut(obj) for obj in current_vis_objects if obj not in [self.axes, self.x_label, self.y_label]],
                    *[Create(obj) if isinstance(obj, VMobject) else FadeIn(obj) for obj in new_vis_objects],
                    run_time=1.5
                )
            else:
                self.play(
                    FadeIn(step_label),
                    FadeIn(step_content),
                    *[Create(obj) if isinstance(obj, VMobject) else FadeIn(obj) for obj in new_vis_objects],
                    run_time=1.5
                )
            
            prev_step_text = (step_label, step_content)
            current_vis_objects = new_vis_objects
            if self.problem_type in ["function", "geometry", "linear_algebra"]:
                current_vis_objects.extend([self.axes, self.x_label, self.y_label])
            
            self.wait(1)
        
        # Final pause at the end
        self.wait(2)
    
    def create_visualization_for_step(self, step, step_number):
        """Create appropriate visualization based on the problem type and current step"""
        try:
            step_text = step.lower()
            objects = []
        
            if self.problem_type == "function":
                # Try to find function definitions or equations
                function_match = re.search(r'f\\(x\\)\\s*=\\s*([^\\n,\\.]+)', step)
                equation_match = re.search(r'y\\s*=\\s*([^\\n,\\.]+)', step)
            
                if function_match:
                    expr = function_match.group(1).strip()
                    try:
                        graph = self.get_function_graph(expr)
                        label = MathTex(f"f(x) = {{expr}}", font_size=24)
                        label.move_to(self.right_panel.get_top() + DOWN * 0.7)
                        objects.extend([graph, label])
                    except:
                        pass
            
                elif equation_match:
                    expr = equation_match.group(1).strip()
                    try:
                        graph = self.get_function_graph(expr)
                        label = MathTex(f"y = {{expr}}", font_size=24)
                        label.move_to(self.right_panel.get_top() + DOWN * 0.7)
                        objects.extend([graph, label])
                    except:
                        pass
            
                # Look for points, intersections, etc.
                point_matches = re.findall(r'\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)', step)
                for x_str, y_str in point_matches:
                    try:
                        x, y = float(x_str), float(y_str)
                        point = Dot(self.axes.c2p(x, y), color=RED)
                        coord_label = MathTex(f"({{x}}, {{y}})", font_size=20)
                        coord_label.next_to(point, UP)
                        objects.extend([point, coord_label])
                    except:
                        pass
        
            elif self.problem_type == "geometry":
                # Look for triangles, circles, etc.
                if "triangle" in step_text:
                    triangle = self.create_triangle_visualization()
                    objects.append(triangle)
            
                elif "circle" in step_text:
                    circle = Circle(radius=2, color=BLUE)
                    circle.move_to(self.right_panel.get_center())
                    objects.append(circle)
            
                # Look for angles
                angle_match = re.search(r'angle\s*=\s*(\d+)', step_text)
                if angle_match:
                    try:
                        angle_value = int(angle_match.group(1))
                        angle_arc = Arc(radius=1.5, angle=angle_value * DEGREES, color=YELLOW)
                        angle_arc.move_to(self.right_panel.get_center())
                        angle_label = MathTex(f"{{angle_value}}^\\circ", font_size=24)
                        angle_label.next_to(angle_arc, RIGHT)
                        objects.extend([angle_arc, angle_label])
                    except:
                        pass
        
            elif self.problem_type == "linear_algebra":
                # Create vector visualization if vectors are mentioned
                if "vector" in step_text:
                    vector_match = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', step)
                    if vector_match:
                        try:
                            x, y = float(vector_match.group(1)), float(vector_match.group(2))
                            vector = Arrow(self.axes.c2p(0, 0), self.axes.c2p(x, y), buff=0, color=GREEN)
                            label = MathTex(f"\\vec{{{{v}}}} = [{{x}}, {{y}}]", font_size=24)
                            label.next_to(vector.get_end(), UP)
                            objects.extend([vector, label])
                        except:
                            pass
        
            elif self.problem_type == "probability":
                # Create probability distribution visualization
                if "probability" in step_text or "distribution" in step_text:
                    bars = self.create_probability_visualization()
                    objects.extend(bars)
        
            # If no specific visualization was created, show a helpful note
            if not objects:
                note = Text("Visualizing this step...", font_size=30, color=GRAY)
                note.move_to(self.right_panel.get_center())
                objects.append(note)
        
            return objects
        except Exception as e:
            print(f"Warning! Failed to create visualization")
            note = Text("Step visualization...", font_size=30, color=GRAY)
            note.move_to(self.right_panel.get_center())
            return [note]
    
    def get_function_graph(self, expr_str):
        """Convert a string expression to a graph"""
        # Replace common mathematical notations with Python syntax
        expr_str = expr_str.replace("^", "**")
        
        # Define the function using the string expression
        def func(x):
            try:
                return eval(expr_str, {{"x": x, "sin": np.sin, "cos": np.cos, "tan": np.tan, 
                                       "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                                       "pi": np.pi, "e": np.e}})
            except:
                return 0
        
        return self.axes.plot(func, color=ORANGE, x_range=[-5, 5, 0.1])
    
    def create_triangle_visualization(self):
        """Create a triangle visualization"""
        vertices = [
            self.axes.c2p(-2, -2),
            self.axes.c2p(2, -2),
            self.axes.c2p(0, 2)
        ]
        triangle = Polygon(*vertices, color=GREEN)
        
        return triangle
    
    def create_probability_visualization(self):
        """Create a simple probability distribution visualization"""
        bar_values = [0.1, 0.2, 0.4, 0.2, 0.1]
        bars = []
        bar_width = 0.8
        
        for i, height in enumerate(bar_values):
            bar = Rectangle(
                width=bar_width,
                height=height * 5,  # Scale for visibility
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=1
            )
            bar.move_to(self.right_panel.get_center() + RIGHT * (i - 2) * bar_width + UP * height * 2.5)
            
            label = MathTex(str(height), font_size=20)
            label.next_to(bar, DOWN)
            
            bars.extend([bar, label])
        
        return bars
    
    def format_math(self, text):
        """Format text for proper LaTeX rendering"""
        # Replace basic formatting
        text = text.replace("^", "^{{").replace("_", "_{{")
        
        # Add closing braces where needed
        opening_count = text.count("^{{") + text.count("_{{")
        closing_count = text.count("}}")
        text += "}}" * (opening_count - closing_count)
        
        return text
'''

    def save_script(self, script: str, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script)

    def render_animation(self, script_path: str, output_dir: str) -> bool:
        try:
            command = ["manim", "-pql","--renderer", "cairo", script_path, "MathSolutionScene", "--media_dir", output_dir]  # Lower quality setting
            print("\nGenerating animation... Please wait.")
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                return True
            print(f"Manim output: {result.stdout}\nErrors: {result.stderr}")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Error generating animation: {e}")
            print(f"Manim output: {e.stdout}\nErrors: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
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