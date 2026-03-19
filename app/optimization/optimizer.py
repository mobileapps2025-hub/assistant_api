import dspy
from dspy.teleprompt import BootstrapFewShot
from app.optimization.dspy_module import RAGModule
from typing import List, Dict

class DSPyOptimizer:
    def __init__(self, teacher_model_name="gpt-4o", student_model_name="gpt-4o-mini"):
        # Configure LM
        # Note: In a real scenario, we might use a cheaper model for the student
        self.lm = dspy.LM(model=teacher_model_name, max_tokens=1000)
        # dspy.settings.configure(lm=self.lm)  # Moved to context manager in compile for async safety
        
        self.module = RAGModule()

    def load_examples(self, examples_data: List[Dict[str, str]]):
        """
        Convert list of dicts to dspy.Example objects.
        Expected format: [{"question": "...", "context": ["..."], "answer": "..."}]
        """
        examples = []
        for ex in examples_data:
            # Ensure context is a list
            context = ex["context"]
            if isinstance(context, str):
                context = [context]
                
            examples.append(dspy.Example(
                question=ex["question"],
                context=context,
                answer=ex["answer"]
            ).with_inputs("question", "context"))
        return examples

    def compile(self, train_examples: List[dspy.Example]):
        """
        Run the BootstrapFewShot optimizer.
        """
        # Simple metric: Exact match is too strict, so we'll use a simple length check or similar for now
        # In reality, we'd use an LLM-based metric (dspy.evaluate.answer_exact_match or similar)
        # For this MVP, we'll assume if it generates *something* reasonable it's okay, 
        # but BootstrapFewShot relies on the metric to select traces.
        
        # Let's use a simple metric that checks if the answer is non-empty and somewhat similar length
        def validate_answer(example, pred, trace=None):
            return len(pred.answer) > 10

        teleprompter = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=4, max_labeled_demos=4)
        
        print("Compiling DSPy program...")
        with dspy.context(lm=self.lm):
            self.compiled_module = teleprompter.compile(self.module, trainset=train_examples)
        print("Compilation complete.")
        
        return self.compiled_module

    def save(self, path: str):
        if hasattr(self, 'compiled_module'):
            self.compiled_module.save(path)
            print(f"Optimized program saved to {path}")
        else:
            print("No compiled module to save.")
