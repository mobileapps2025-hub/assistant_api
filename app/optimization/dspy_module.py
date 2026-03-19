import dspy
from typing import List

class GenerateAnswer(dspy.Signature):
    """
    Answer the user's question based on the retrieved context.
    """
    context = dspy.InputField(desc="Relevant facts and documents retrieved from the knowledge base.")
    question = dspy.InputField(desc="The user's question.")
    answer = dspy.OutputField(desc="A helpful, accurate, and concise answer to the question.")

class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str, context: List[str]):
        # Join context list into a single string
        context_str = "\n\n".join(context)
        prediction = self.generate(context=context_str, question=question)
        return dspy.Prediction(answer=prediction.answer)
