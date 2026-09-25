# Vision-Language-Action (VLA) Mini Simulator

A from-scratch learning implementation of a tiny Vision-Language-Action (VLA) pipeline, rebuilt from the beginner-oriented VLA basics.
<img width="807" height="169" alt="image" src="https://github.com/user-attachments/assets/2ac34d06-0062-4482-883d-07fe01036e9a" />


The purpose of this repository is not to reproduce a production robotics VLA. It is to make the complete **vision → language → fusion → action** path small enough that every tensor, transformation, and design choice can be inspected.

It uses a custom 64×64 RGB CartPole-style simulator, a small convolutional vision encoder, a bag-of-words language encoder, feature concatenation, and a two-action policy head. The linked notebook then performs inference directly in the simulator. No dataset, pretrained model, robot hardware, or training loop is required for the original demonstration.

> Learning project: simulation only. No physical robot or hardware interface is required.

---

## 1. What This Project Is

A Vision-Language-Action model receives:

```text
Image / visual observation
        +
Human language instruction
        ↓
Vision encoder + language encoder
        ↓
Fusion
        ↓
Action / policy head
        ↓
Action probabilities
        ↓
Environment
```

In this miniature implementation, the input and output are deliberately tiny:

```text
Vision input:      64 × 64 × 3 RGB image
Language input:    text instruction
Actions:           0 = move left
                   1 = move right
Output:            P(left), P(right)
```

The environment is a simple cart-and-pole scene. The agent sees the scene as an image rather than receiving the underlying simulator state directly.

---


## 2. Components

### 2.1 Custom MiniCartPoleVisionEnv

A small Gymnasium environment supplies the visual observation.

The environment contains two state variables:

```text
x      = horizontal cart position
theta  = pole angle
```

It exposes:

```text
observation_space = Box(0, 255, shape=(64, 64, 3), dtype=uint8)
action_space      = Discrete(2)
```

The two actions are:

```text
0 → move cart left
1 → move cart right
```

The simulator returns an RGB image instead of the raw values `x` and `theta`.

### 2.2 Renderer

The environment creates a 64×64 RGB image with Pillow.

Conceptually:

```text
black background
      |
      +-- yellow rectangle = cart
      |
      +-- blue line        = pole
```

The cart is placed according to `x`, while the pole endpoint depends on `theta`.

This matters because the VLA is supposed to operate on the **image**, not on `theta` directly.

### 2.3 Vision Encoder

The image encoder is a small convolutional neural network:

```text
Input:  64 × 64 × 3
   ↓
Conv2D: 3  → 16 channels, kernel 3, stride 2, padding 1
   ↓
ReLU
   ↓
Conv2D: 16 → 32 channels, kernel 3, stride 2, padding 1
   ↓
ReLU
   ↓
Flatten
   ↓
8192-dimensional vision vector
```

The spatial dimensions change as follows:

```text
64 × 64 × 3
       ↓ first stride-2 convolution
32 × 32 × 16
       ↓ second stride-2 convolution
16 × 16 × 32
       ↓ flatten
32 × 16 × 16 = 8192
```

The network does not attempt object detection, segmentation, or image captioning. It simply learns a compact feature representation of the RGB input.

### 2.4 Language Encoder

The language side is deliberately primitive.

The instruction is converted into a 1,000-element bag-of-words-style vector.

For example:

```text
"keep the pole upright"
```

is split into words, and each word is mapped to one of 1,000 positions using a hash:

```text
word → hash(word) → index in [0, 999]
```

The resulting vector contains word counts at those positions.

The 1,000-dimensional vector is then projected into a 32-dimensional embedding:

```text
1000-D hashed text vector
          ↓
     Linear(1000, 32)
          ↓
      32-D embedding
```


### 2.5 Multimodal Fusion

The two embeddings are concatenated:

```text
Vision:   8192 values
Language:   32 values
              ↓
          concatenate
              ↓
         8224 values
```

Formally:

```text
z = [v ; t]
```

where:

```text
v = vision embedding
t = text embedding
z = fused representation
```

This is the simplest possible form of multimodal fusion.

### 2.6 Policy / Action Head

The fused 8,224-dimensional vector is passed through a small multilayer perceptron:

```text
8224
 ↓
Linear(8224, 64)
 ↓
ReLU
 ↓
Linear(64, 2)
 ↓
2 action logits
 ↓
Softmax
 ↓
P(action=0), P(action=1)
```

The selected action is the index of the largest probability:

```text
action = argmax(probabilities)
```

---

## 3. End-to-End Data Flow

At each simulation step:

```text
1. Environment produces RGB observation

2. Image is converted to a tensor

3. Image values are normalized:
      pixel / 255

4. Image layout changes:
      NHWC → NCHW

5. ConvNet extracts visual features

6. Instruction is converted to a 1000-D bag-of-words vector

7. Linear layer converts text vector to 32-D embedding

8. Vision + language vectors are concatenated

9. Policy network produces 2 logits

10. Softmax converts logits to probabilities

11. Argmax selects left or right

12. Action is applied to the environment

13. Environment returns the next observation and reward

14. Repeat
```



---

## 4. Mathematical View

Let:

```text
I = visual observation
T = text instruction
```

The VLA first creates separate representations:

```text
v = f_vision(I)
t = f_text(T)
```

Then it fuses them:

```text
z = concatenate(v, t)
```

The policy network maps the fused representation to action logits:

```text
l = f_policy(z)
```

The action distribution is:

```text
p(a | I, T) = softmax(l)
```

For the two-action environment:

```text
p(left)  = p(a = 0 | I, T)
p(right) = p(a = 1 | I, T)
```

The inference code then uses:

```text
argmax(p(left), p(right))
```

to choose the action.

---

## 5. Environment Dynamics

The custom environment intentionally uses extremely simple dynamics.

For an action `a`:

```text
if a == 1:
    x ← x + 0.05
else:
    x ← x - 0.05
```

The pole angle is updated with Gaussian random noise:

```text
theta ← theta + Normal(0, 0.02)
```

The reward is:

```text
reward = -|theta|
```

The episode terminates when:

```text
|theta| > 0.5
```

The environment does not define a time-limit truncation in the original implementation:

```text
truncated = False
```

### Why this environment exists

It provides exactly the interface needed to demonstrate embodied interaction:

```text
state → image → model → action → state update
```

It is not intended to be a physically accurate CartPole implementation.

---

## 6. Image Representation

The observation is a NumPy array:

```text
shape = (64, 64, 3)
dtype = uint8
range = 0 ... 255
```

This is the conventional image layout used by the simulator/rendering code:

```text
H × W × C
```

PyTorch convolution layers expect:

```text
N × C × H × W
```

Therefore inference changes:

```text
(1, 64, 64, 3)
        ↓
(1, 3, 64, 64)
```

and scales the values:

```text
image = image / 255.0
```

This conversion is essential because the ConvNet is implemented with `nn.Conv2d`.

---

## 7. Text Representation

a simple hashing scheme instead of maintaining an explicit vocabulary.

Conceptually:

```python
bow = zeros(1000)
for word in text.lower().split():
    index = abs(hash(word)) % 1000
    bow[index] += 1
```

This gives a fixed-size text vector regardless of the number of distinct words.

### Important consequence

Different words can map to the same bucket. This is a hash collision.

For this educational project, that is acceptable because the purpose is to demonstrate the pipeline, not to create a high-quality language representation.

A production system would normally use a tokenizer and a learned language representation rather than raw Python string hashing.

---

## 8. Model Dimensions

The important tensor dimensions are:

| Stage | Shape |
|---|---:|
| RGB input | `64 × 64 × 3` |
| Batched image | `1 × 64 × 64 × 3` |
| After NHWC → NCHW | `1 × 3 × 64 × 64` |
| Conv 1 output | `1 × 16 × 32 × 32` |
| Conv 2 output | `1 × 32 × 16 × 16` |
| Flattened vision vector | `8192` |
| Hashed text vector | `1000` |
| Text embedding | `32` |
| Fused vector | `8224` |
| Hidden policy layer | `64` |
| Action logits | `2` |
| Action probabilities | `2` |

This table should be traceable directly through the code.

---

## 9. Approximate Parameter Count

The miniature network is small enough to calculate manually.

### Vision encoder

```text
Conv1:
3 × 16 × 3 × 3 + 16 = 448

Conv2:
16 × 32 × 3 × 3 + 32 = 4640
```

### Language encoder

```text
1000 × 32 + 32 = 32032
```

### Policy head

```text
8224 × 64 + 64 = 526400
64 × 2 + 2 = 130
```

### Total

```text
448 + 4640 + 32032 + 526400 + 130
= 563650 trainable parameters
```

The majority of the parameters are in the first policy layer because the flattened image representation is 8,192 dimensions.

---


## 10. Dependencies

The source notebook installs:

```text
gymnasium
pillow
torch
```

The visualization cells also use:

```text
matplotlib
```

A simple `requirements.txt` for a learning-oriented reconstruction is:

```text
gymnasium
pillow
matplotlib
torch
```

The original notebook was executed in Google Colab and recorded a Python 3 environment with a T4 GPU runtime. The model is small enough that the conceptual workload does not require a large GPU.

For local PyTorch installation, use the PyTorch installation method appropriate for your CPU/GPU platform rather than assuming the Colab CUDA build.

---

## 11. Setup

### Option A: Google Colab

The original project is designed to be easy to run in Colab.

Open the notebook and execute cells in order.

Typical first cell:

```bash
pip install gymnasium pillow torch
```

The notebook then imports the required packages and builds the simulator.

### Option B: Local Python environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Implementation Order

Rebuild the project in this order. Do not start with the model.

### Step 1 — Build the environment

Implement:

```text
__init__
reset
step
render
```

The first checkpoint is simply:

```python
obs, _ = env.reset()
print(obs.shape)
```

Expected:

```text
(64, 64, 3)
```

### Step 2 — Render the observation

Use Matplotlib:

```python
plt.imshow(obs)
plt.axis("off")
plt.show()
```

You should see the tiny cart and pole.

### Step 3 — Test raw actions

Call both actions manually:

```python
env.step(0)
env.step(1)
```

The purpose is to verify that the environment changes before adding any neural network.

### Step 4 — Build the vision encoder

Create the two convolution layers and verify the output shape.

Expected flattened dimension:

```text
8192
```

### Step 5 — Build text encoding

Convert an instruction such as:

```text
keep the pole upright
```

to the 1,000-element hashed bag-of-words vector.

Then verify:

```text
shape = (1000,)
```

### Step 6 — Build the language projection

Pass the 1,000-D vector through:

```text
Linear(1000 → 32)
```

Verify the output shape:

```text
(32,)
```

### Step 7 — Fuse the modalities

Concatenate:

```text
8192 + 32 = 8224
```

### Step 8 — Build the policy head

Use:

```text
8224 → 64 → 2
```

Then apply softmax.

### Step 9 — Connect model and environment

The inference loop becomes:

```python
obs, _ = env.reset()

for step in range(5):
    image = ...
    text = ...

    probs = model(image, text)
    action = probs.argmax(...)

    obs, reward, done, truncated, info = env.step(action)
```

### Step 10 — Visualize the decision

Show:

```text
left panel  = current RGB observation
right panel = action probabilities
```

This makes the multimodal-to-action pipeline directly observable.

---

## 13. Inference Pipeline in Pseudocode

```text
initialize environment
initialize MiniVLA
set instruction = "keep the pole upright"

convert instruction → bag-of-words vector

repeat for each step:

    get RGB observation
    convert RGB image → float tensor
    normalize pixel values
    convert NHWC → NCHW

    vision_vector = vision_encoder(image)
    text_vector   = text_encoder(bow_instruction)

    fused = concatenate(vision_vector, text_vector)

    logits = policy_head(fused)
    probabilities = softmax(logits)

    action = argmax(probabilities)

    apply action to environment

    observe next image and reward

    stop if episode terminates
```



## 14. Important Fidelity and Environment Notes

There are several details worth understanding when rebuilding the original experiment.

### 14.1 The action does not directly control the pole angle

The original `step()` logic changes `x` according to the chosen action, while `theta` is updated by Gaussian noise.

Therefore:

```text
action → x movement
noise  → theta movement
```

rather than a coupled physical equation such as a real CartPole system.

Consequently, the action cannot truly correct the pole angle through the environment dynamics.

### 14.2 The reward is determined only by pole angle

Because:

```text
reward = -abs(theta)
```

the reward depends on the current angle, not on whether the cart moved in the direction needed for balancing.

### 14.3 The environment has no explicit x bounds

The renderer calculates the cart center from `x`.

Over enough steps, the cart can move outside the 64×64 image area unless additional bounds are introduced.

### 14.4 `hash()` is not a stable vocabulary mapping across Python processes

Python's hash randomization can cause hash values to differ between processes.

For a learning reconstruction, this is acceptable as a demonstration of fixed-size hashing, but it is not a good production feature representation.

A deterministic hash or explicit tokenizer would be more reproducible.

### 14.5 The demo uses a fixed instruction

The notebook repeatedly uses an instruction equivalent to:

```text
keep the pole upright
```

There is no instruction dataset and no language-conditioned training procedure in the original demo.

---


---

## 15. Example Output

A run follows this pattern:

```text
step=0, action=0, reward=-0.100
step=1, action=0, reward=-0.094
step=2, action=0, reward=-0.112
step=3, action=0, reward=-0.114
step=4, action=0, reward=-0.124
```

The exact sequence is not guaranteed because the environment injects random noise into `theta`, and the neural network is not trained.

The useful observation is the complete control loop, not a particular action sequence.

---
