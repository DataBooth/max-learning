# Feedback: InferenceSession Terminology in Educational Context

## Background

I'm developing [max-learning](https://github.com/DataBooth/max-learning), an educational repository with progressive examples teaching the MAX Graph API from simple operations to transformers. While learning MAX and creating the first example ([01_elementwise](https://github.com/DataBooth/max-learning/tree/main/examples/python/01_elementwise)), which demonstrates basic element-wise operations like `y = relu(x * 2.0 + 1.0)`, I encountered a terminology question.

## The Question

The example uses `InferenceSession` to execute a simple mathematical operation:

```python
# Example: y = relu(x * 2.0 + 1.0)
session = InferenceSession(devices=[CPU()])
model = session.load(graph)
output = model.execute(input_tensor)
```

**Observation**: We're calling it "inference" even though:
- There's no trained model
- No learned weights
- Just hardcoded mathematical constants
- It's basic computational graph execution

From a traditional ML perspective, "inference" typically means "using a trained model on new data" (as opposed to training). This created a moment of confusion while learning.

## Discussion

**MAX's Perspective** (which makes sense):
- MAX Engine is a graph execution engine
- `InferenceSession` executes *any* computational graph
- Whether it's simple maths, trained models, or preprocessing
- The term aligns with other inference engines (TensorRT, ONNX Runtime, etc.)

**Learner's Perspective**:
- "Inference" has strong ML connotations
- Might expect to see trained models and predictions
- Could cause confusion in educational/introductory contexts
- "I'm not doing inference, I'm just doing maths!"

## Potential Approaches

Rather than suggesting an API change (which would be disruptive), perhaps:

### 1. Documentation Enhancement
Add a clarification in tutorials/docs, such as:
> **Note**: `InferenceSession` is MAX's execution context for running *any* computational graph, not just ML model inference. Whether you're doing simple maths operations or running trained neural networks, you use `InferenceSession` to execute your graph.

### 2. Tutorial/FAQ Addition
A FAQ entry like:
> **Q: Why is it called InferenceSession if I'm just doing basic operations?**  
> A: MAX Engine uses "inference" to mean "graph execution" in general. Any time you run a compiled graph—whether it's a trained model, data preprocessing, or mathematical operations—you're performing "inference" in MAX terminology. This aligns with industry-standard inference engines.

### 3. Terminology Alternatives (for consideration)
If future API versions are considered:
- `ExecutionSession` / `GraphSession` - more general
- `RuntimeSession` - emphasises execution phase
- `ComputeSession` - neutral about use case
- Keep `InferenceSession` but clarify in docs

## Request

Would appreciate the team's perspective on:
1. Whether documentation clarification would be valuable
2. If this is feedback others have raised
3. Any historical reasoning behind the naming that would help explain it in educational materials

## Repository Context

The max-learning repository aims to be a comprehensive learning resource:
- Progressive examples (element-wise → linear layers → transformers)
- Clear documentation of MAX architecture and design decisions
- [Investigation of Mojo Graph API status](https://github.com/DataBooth/max-learning/tree/main/examples/mojo/01_elementwise)
- Acknowledgements of [MAX documentation and inspirations](https://github.com/DataBooth/max-learning/blob/main/docs/ACKNOWLEDGEMENTS.md)

Trying to make MAX as approachable as possible while learning it myself and maintaining technical accuracy.

---

**Feedback provided with respect and appreciation for the MAX platform.** This is about making the framework even more accessible to learners.

---

## Short Version (for Discord)

I'm learning MAX Graph API and building educational examples. Quick terminology question:

In my first example doing simple maths (`y = relu(x * 2.0 + 1.0)`), I use `InferenceSession` even though there's no trained model or ML inference in the traditional sense—just executing a computational graph.

```python
session = InferenceSession(devices=[CPU()])
model = session.load(graph)
output = model.execute(input_tensor)
```

I understand MAX uses "inference" to mean "graph execution" (aligning with TensorRT, ONNX Runtime, etc.), but wondered if docs could clarify this for newcomers? Traditional ML meaning of "inference" = using trained models, which caused a momentary "wait, I'm not doing inference?" moment while learning.

Not suggesting API changes—just wondering if a note in tutorials/docs might help other learners. Something like: *"InferenceSession runs any computational graph, not just ML model inference."*

Repo: https://github.com/DataBooth/max-learning
Example: https://github.com/DataBooth/max-learning/blob/main/examples/python/01_elementwise/elementwise_minimal.py
