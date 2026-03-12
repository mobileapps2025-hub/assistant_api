import asyncio
import dspy
from dspy.teleprompt import BootstrapFewShot

# Simple module
class MySig(dspy.Signature):
    input = dspy.InputField()
    output = dspy.OutputField()

class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(MySig)
    def forward(self, input):
        return self.prog(input=input)

async def main():
    # Emulate what the app does
    lm = dspy.LM(model="gpt-4o-mini", max_tokens=50)
    
    trainset = [dspy.Example(input="hi", output="hello").with_inputs("input")]
    
    def metric(example, pred, trace=None):
        return True

    teleprompter = BootstrapFewShot(metric=metric, max_bootstrapped_demos=1)
    
    print("Compiling...")
    try:
        with dspy.context(lm=lm):
            compiled = teleprompter.compile(MyModule(), trainset=trainset)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
