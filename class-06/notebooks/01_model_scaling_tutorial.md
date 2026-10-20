# Model Scaling Tutorial

## 1 Introduction

When planning an AI project of significant size, one of the early questions we need to answer is *what computing resources will we need?*

Though it is not always practical to answer this question prematurely, once we have established our use case, formulated requirements compared to some baseline, determined how much data is required to represent the key attributes of our problem, and have an idea of the type of models we would like to train (perhaps from the results of early, small-scale experiments), we are ready to start planning for the computing resources required to *scale* our models.

Generally, the computing resources we need to train and serve our models will grow as a function of the size of the models we are working with. Assuming end-to-end training, the size of the models will (usually) grow as a function of the size of the dataset. With those factors in mind, the right time to start thinking about scaling is before we start budgeting for large scale experiments (since compute will be a significant cost) and after we have established the factors our dataset must represent.

For example, if we are training an object recognition model, we will want an idea of how many objects we must recognize, from what poses, under what lighting conditions, from what distances, etc. so we can determine how many overall examples we should plan to collect or obtain to define the intended behavior of our model as completely as possible. With this number determined, we can get a rough idea of how large our model needs to be. From there, we can determine how many and what type of GPUs we might need to train, and can start to determine training cost from the cost of these GPUs.

The number and type of GPUs required also will impose requirements on the skills of the team needed for the project. If a significant number of high-end GPUs are required, a dedicated team to set them up and maintain them might be warranted. Very high-end GPUs might have special facilities requirements for power and cooling if you are hosting them yourself.

### 1.1 The Relationship Between Model Size and Computing Requirements

All of these factors will depend on the relationship between model size and computing requirements. While obtaining an exact formula for this relationship is intractable for models of typical size and complexity for many practical applications, several scaling heuristics are available in the literature and used in practice to answer this question. Here we will review some of the scaling rules used in practice for *dense transformer models*, i.e., transformer models designed such that most of the models' neurons are active for any given input (as opposed to sparse models or mixture-of-experts designs where only some neurons are active for a given input by design).

These scaling laws will give us the following approximate working relationships.
* $D(\mu)$, the relationship between data set size, $D$, in *tokens* and model size, $\mu$, in *parameters*.
* $C(\mu)$, the relationship between the computational cost to train, $C$, in floating point operations (FLOPs) and the model size, $\mu$.
* $M(\mu)$, the relationship between the memory required during training and the model size, $\mu$.

We will use these relationships to answer practical questions about the computational cost and memory required to train our models.

* *How many GPUs do I need to train a model?* Given a target model size, $\mu$, we wish to know the computing cost to train it, $C$, and how much memory it will use, $M$ so we can select a set of GPUs that meet these requirements and determine if they will be within our project budget.
* *How large of a model can I train?* Given a number of GPUs available to us and information about these GPUs, we also wish to know the limits they will impose on model sizes we can feasibly train, $\mu$. This is the question above.

The answers to both of these questions are of course also influenced by the time available to us, $\tau$. If we are willing to wait longer to finish training runs, we may be able to train larger models on smaller or fewer GPUs, within memory constraints.

We will start by reviewing some basic forms of these equations from the literature on dense transformer scaling, and then rearrange them to answer our questions above. This will enable us to build up a *scaling calculator* we can use in subsequent project planning.

## 2 Fundamental Scaling Equations

Different network architectures and training regimens will have different scaling laws, and in practice scaling equations will be determined by a combination of literature review of prior work and practical experiments on smaller models. For this scaling exercise, we will assume a dense transformer architecture since this is well studied, enabling us to take our scaling equations from literature for the purposes of the exercise.

The following three papers provide useful insights for estimating the computational requirements of dense transformers.
* *Chinchilla* (Hoffmann *et al.*) provides a relationship between model size, training data, and training compute.
* *Pathways Language Model (PaLM)* provides a useful approximation for the FLOPs required to train a dense transformer.
* *Zero Redundancy Optimizer (ZeRO)* provides a useful approximation for the memory required to train a model.

Each of these papers had different goals. Chinchilla was seeking to empirically establish fundamental relationships between model size, training data required to train it, and training compute available. PaLM was seeking to advance understanding of how model scale affects few shot learning performance. ZeRO itself presents an approach for efficiently distributing models across GPUs with minimal overhead. Here we use the scaling laws they uncovered as the basis to answer our practical questions above.

### 2.1 Chinchilla: Relating Model Size and Training Dataset Size

The Chinchilla scaling analysis investigates the relationship between model size, training data, and training compute. One of its key findings is that, under a fixed training compute budget, the compute-optimal number of training tokens scales approximately linearly with the number of model parameters.

We can express this relationship using a *tokens-per-parameter scaling factor*:

$$
D(\mu)=r\mu
$$

where:

| Symbol | Description | Units |
|---|---|---|
| $D$ | Number of training tokens | [tokens] |
| $\mu$ | Number of model parameters | [parameters] |
| $r$ | Tokens-per-parameter ratio | [tokens/parameter] |

We will keep track of the units throughout our calculations with dimensional analysis. Starting with the basic chinchilla formula:

$$
D\ [\text{tokens}] = r \left[\frac{\text{tokens}}{\text{parameter}}\right]\ \mu\ [\text{parameters}].
$$

A commonly used approximation based on the Chinchilla results is $r \approx 20$ [tokens/parameter].

For example, a model with $\mu=7 \times 10^9$ [parameters] would have a compute-optimal training dataset of approximately

$$
D=20\ \frac{\text{tokens}}{\text{parameter}}\ (7 \times 10^9)\ \text{parameters} = 1.4 \times 10^{11}\ \text{tokens}.
$$

Note that $D$ represents the total tokens processed by the model, which can include repeated passes over the same dataset.

Chinchilla scaling provides a useful heuristic because it establishes a fixed proportional relationship between model size and dataset size. This means for planning purposes, we can readily think in terms of either model size, i.e., "how large of a model do I want to train?" or dataset size, i.e., "I have a dataset size of $D$, how large of a model do I need?" by plugging in the proportional relationship. Note that this is a heuristic rather than a universal optimal ratio. The original Chinchilla analysis fits empirical scaling laws for model loss as a function of model size and training data, subject to a compute budget.

The important result of the Chinchilla analysis is that there is a useful balance between the number of model parameters and the amount of training data processed. Given a fixed computational budget, we can choose to train a smaller model on more tokens or a larger model on fewer tokens. The Chinchilla results suggest that neither extreme is generally optimal for the language models studied.

Note also that the Chinchilla relationship does not tell us how large a model needs to be to solve a particular problem. The model size still depends on the complexity of the task, the desired performance, and the results of experiments. The relationship is useful once we have selected a candidate model size and wish to estimate an appropriate training workload.

### 2.2 PaLM: Training Computational Cost

While the Chinchilla paper provides the relationship between dataset size and model size that we need, the PaLM paper provides a useful approximation for the floating-point operations (FLOPs) cost, $C(\mu)$, required to train a dense transformer model. We will consider two versions, of $C(\mu)$. First, ignoring self-attention corrections, the total training compute is approximately given by

$$
C(\mu)\approx6D\mu.
$$

A more complete approximation, including the computational cost of self-attention, is

$$
C(\mu)\approx D(6\mu+12LSW)
$$

where:

| Symbol | Description | Units |
|---|---|---|
| $C$ | Total training compute | [FLOPs] |
| $\mu$ | Number of model parameters | [parameters] |
| $D$ | Number of training tokens | [tokens] |
| $L$ | Number of Transformer layers | [dimensionless] |
| $S$ | Sequence length (context window) | [tokens] |
| $W$ | Transformer hidden dimension (model width) | [elements] |

The factor of 6 arises from the approximate computational cost of a training step including
approximately 2 [FLOPs/(parameter $\cdot$ token)] for the forward pass and approximately 4 [FLOPs/(parameter $\cdot$ token)] for the backward pass. This is an approximate correction for dense transformers and assumes a fixed sequence length, $S$.

TODO: additional explanation

For the simplified equation, dimensional analysis gives

$$
\begin{aligned}
C(\mu)\ [\text{FLOPs}] &= 6\ \left[\frac{\text{FLOPs}}{\text{parameter} \cdot \text{token}}\right] D\ [\text{tokens}]\ \mu\ [\text{parameters}]
\end{aligned}
$$

The additional term $12LSW$ accounts for the matrix multiplications involved in self-attention. Its contribution becomes increasingly important as the sequence length grows.

Both approximations assume a dense transformer in which nearly all parameters participate in each forward pass. In practice, many state-of-the-art models use techniques such as mixture-of-experts (MoE) so that not all parameters of the model are used to process a given input. Such optimizations are not accounted for here.

### 2.3 Combined Scaling Equations

From Chinchilla scaling, we have

$$
D(\mu)=r\mu.
$$

Substituting into the PaLM equation, $C(\mu) = 6D\mu$, gives

$$
\begin{aligned}
C(\mu) &\approx 6\mu(r\mu)\\
C(\mu) &\approx 6r\mu^2.
\end{aligned}
$$

This reveals the fundamental scaling relationship we will exploit for most of our calculations. *For a fixed tokens-per-parameter ratio, the computational workload grows quadratically with model size*.

We can verify the units using dimensional analysis, treating the factor of 6 as approximately 6 [FLOPs/(parameter $\cdot$ token)]:

$$
C(\mu)\ [\text{FLOPs}] =6\ \left[\frac{\text{FLOPs}}{\text{parameter}\cdot\text{token}}\right]\ r\ \left[\frac{\text{tokens}}{\text{parameter}}\right]\ \mu^2\ [\text{parameters}]^2.
$$

This gives our fundamental workload estimate:

$$
\boxed{C(\mu)\approx6r\mu^2}
$$

We will come back to this expression to answer our computational planning questions.

### 2.4 Model Memory Requirements

Besides computing workload, $C(\mu)$, the next relationship we need to understand is that between model size and GPU memory required to train the model. It will not help to demonstrate that we can handle the computational load of training a model of a given size on a particular GPU if that model does not fit in memory.

We will consider two cases here. First, we will consider training on one GPU, and then consider the case where our model is *sharded* across multiple GPUs. Though will not implement our own sharding tools here, we will use the relationships developed in the ZeRO paper, which itself influenced the development of the Fully Sharded Data Parallel (FSDP) approach used in PyTorch.

Following the mixed-precision Adam memory accounting described by the ZeRO paper, the memory required by the model's training state can be estimated using 16 bytes per parameter.

| Component | Memory per Parameter |
|---|---:|
| BF16 model weights | 2 [bytes/parameter] |
| BF16 gradients | 2 [bytes/parameter] |
| FP32 master weights | 4 [bytes/parameter] |
| FP32 Adam first moment | 4 [bytes/parameter] |
| FP32 Adam second moment | 4 [bytes/parameter] |
| **Total** | **16 [bytes/parameter]** |

Thus, the total memory required for model training states is approximately

$$
M_{\text{train}}(\mu) \approx 16\mu.
$$

Here each symbol is defined:

| Symbol | Description | Units |
|---|---|---|
| $M_{\text{train}}$ | Memory required for training states | [bytes] |
| $\mu$ | Number of model parameters | [parameters] |
| $16$ | Memory per model parameter | [bytes/parameter] |

Note this is the *training-state memory*, which is distinct from the total memory required, which also includes activations, buffers, and additional overhead we will leave reserve memory for.

### 2.5 Available GPU Memory

Not all GPU memory can be used to store the model's training states. Some memory must be reserved for activations, temporary buffers, and runtime overhead.

We estimate the available memory per GPU as

$$
M_{\text{avail}}=M(1-f_R)
$$

where:

| Symbol | Description | Units |
|---|---|---|
| $M_{\text{avail}}$ | Available memory per GPU | [bytes] |
| $M$ | Total memory capacity per GPU | [bytes] |
| $f_R$ | Fraction of memory reserved for overhead | [dimensionless] |

A reserve of $f_R=0.25$ is an initial value that can be used for planning (note this is not an established result of the ZeRO paper).

Assuming the training states can be fully *sharded*, or distributed, across $N$ identical GPUs, the total available memory becomes

$$
M_{\text{avail}}=N_{\text{GPU}} M (1-f_R)\ [\text{bytes}].
$$

Therefore, to fit the model's training states into GPU memory, we require $M_train \leq M_{\text{avail, total}}$, i.e.,

$$
16\ \left[\frac{\text{bytes}}{\text{parameter}}\right]\ \mu\ [\text{parameters}] \leq N_{\text{GPU}}M(1-f_R)\ [\text{bytes}].
$$

This heuristic is given by a ZeRO-3/FSDP-style assumption in which weights, gradients, and optimizer states are all sharded. Different sharding strategies will require different heuristics.

## 3 GPU Throughput

A GPU's computational throughput measures how many floating-point operations it can perform per second. GPU manufacturers typically report peak theoretical throughput in teraFLOPs per second (TFLOPs/s). Note that throughput (FLOPs/s) is distinct from FLOPs and measures how many operations we can perform per second, rather than the total number of operations we need to perform. For a given compute cost $C$ in total operations, i.e., FLOPs, the GPU throughput in FLOPs/s gives us an estimate of how long it will take the GPU to execute those operations.

We will denote a GPU's peak computational throughput as $P$:

| Symbol | Description | Units |
|---|---|---|
| $P$ | Peak computational throughput of one GPU | [FLOPs/s] |

The manufacturer will usually specify a GPU's peak computational throughput in TFLOPs/s. A teraFLOP represents $10^{12}$ floating-point operations. Therefore, a GPU rated at $P_{\text{TF}}$ [TFLOPs/s] has a peak throughput of

$$
P = P_{\text{TF}}\times 10^{12}\ \frac{\text{FLOPs}}{\text{s}}.
$$

The theoretical peak throughput depends on the GPU architecture, numerical precision (e.g., FP32, FP16, BF16), and the hardware used. When planning, if we know what numerical precision we will need (perhaps from early experiments), we can look up the manufacturer specified throughput for that precision. If we are doing early planning, we can run our calculations for multiple precision assumptions to determine what the trade off will be between numerical precision and computing workload.

### 3.1 Model FLOPs Utilization (MFU)

In practice, a GPU rarely achieves its theoretical peak throughput when training a model. Data movement, memory bandwidth, communication between GPUs, and other overheads reduce the effective computational throughput.

We account for these effects using *Model FLOPs Utilization (MFU)*, denoted by $\eta$.

$$
P_{\text{eff}} = \eta P
$$

where:

| Symbol | Description | Units |
|---|---|---|
| $P_{\text{eff}}$ | Effective model computational throughput per GPU | [FLOPs/s] |
| $P$ | Peak computational throughput per GPU | [FLOPs/s] |
| $\eta$ | Model FLOPs utilization, $0<\eta\leq1$ | [dimensionless] |

For example, a GPU with a peak throughput of 300 [TFLOPs/s] operating at an MFU of 40% has an effective computational throughput of

$$
\begin{aligned}
P_{\text{eff}}
&= 0.4 \times 300 \times 10^{12}\ \frac{\text{FLOPs}}{\text{s}} \\
&= 1.2 \times 10^{14}\ \frac{\text{FLOPs}}{\text{s}}.
\end{aligned}
$$

In this approximation, MFU represents the fraction of theoretical peak throughput effectively applied to the model's estimated computational workload. It can be measured experimentally and depends on the model architecture, training configuration, hardware, and parallelization strategy, among other parameters. It would be common to start determining MFU from early scaling experiments before training a large model.

## 4 Training Wall Time

Having estimated the total computational workload $C$ [FLOPs] and the effective GPU throughput $P_{\text{eff}}$ [FLOPs/s], we can estimate the *wall time*, $\tau$, required to train a model, i.e., the elapsed time required to complete training.

### 4.1 Training on a Single GPU

For a single GPU, the training time is

$$
\tau(\mu) = \frac{C(\mu)}{P_{\text{eff}}}.
$$

Substituting the effective throughput equation gives the time in terms of the ideal throughput specified by the manufacturer, accounting for the MFU, $\eta$,

$$
\tau = \frac{C(\mu)}{\eta P}.
$$

With units:

$$
\tau(\mu)\ [\text{s}] = \frac{C(\mu)\ [\text{FLOPs}]}{\eta\ [1]\ P\ \left[\frac{\text{FLOPs}}{\text{s}}\right]}.
$$

### 4.2 Training on Multiple GPUs

If we have $N_{\text{GPU}}$ identical GPUs, we can approximate the total effective computational throughput by multiplying the throughput of a single GPU by the number of GPUs:

$$
P_{\text{total}} = \eta P N_{\text{GPU}}
$$

The wall time required to process a workload of $C$ [FLOPs] is then given by

$$
\tau(\mu) = \frac{C(\mu)}{\eta P N_{\text{GPU}}}.
$$

Substituting the simplified PaLM computational workload $C(\mu)\approx6D\mu$ gives

$$
\tau(\mu) \approx \frac{6D\mu}{\eta P N_{\text{GPU}}}.
$$

Alternatively, including the self-attention correction,

$$
\tau(\mu) \approx \frac{D(6\mu+12LSW)}{\eta P N_{\text{GPU}}}.
$$

These equations provide wall time in seconds when $P$ is expressed in [FLOPs/s].

Note that in multi-GPU calculations, the MFU, $\eta$, must represent the utilization in the multi-GPU setup, which will not be the same as a single GPU MFU measurement.

### 4.3 Converting Wall Time to Hours

There are 3,600 seconds in one hour. We therefore convert training time to hours using

$$
\tau_{\text{hrs}} = \frac{\tau}{3600\ \text{s/hr}}.
$$

Substituting the simplified training workload gives

$$
\tau_{\text{hrs}} \approx \frac{6D\mu}{3600\eta P N_{\text{GPU}}}
$$

With units:

$$
\tau_{\text{hrs}}\ [\text{hrs}] \approx
     \frac{6D\mu\ [\text{FLOPs}]}
          {3600\ \left[\frac{\text{s}}{\text{hrs}}\right]
           \ \eta
           \ [1]
           \ P
           \ \left[\frac{\text{FLOPs}}{\text{s}}\right]
           \ N_{\text{GPU}}\ [1]}
$$

Here, the numerical factor 3600 represents the conversion between seconds and hours.

If we instead express GPU throughput in [TFLOPs/s], the equation becomes

$$
\boxed{
\tau_{\text{hrs}}(\mu) \approx
\frac{6D\mu}{\left(3600\times10^{12}\right) \eta\ P_{\text{TF}}N_{\text{GPU}}}
}
$$

This provides a practical way to estimate the number of hours required to train a model given its parameter count, dataset size, GPU count, peak GPU throughput, and expected utilization, assuming that the workload can be distributed across all GPUs and that the effective aggregate throughput is represented by $N_{\text{GPU}}\eta P$. In practice, GPU communication, synchronization, memory limitations, and workload partitioning can reduce scaling efficiency. These effects must be reflected in the selected MFU value for a realistic estimate.

## 5 Practical Questions

Now we are ready to compute answers to some of our practical questions:

* *How many GPUs do I need to train a model?* Given a target model size, $\mu$, we wish to know the computing cost to train it, $C$, and how much memory it will use, $M$ so we can select a set of GPUs that meet these requirements and determine if they will be within our project budget.
* *How large of a model can I train?* Given a number of GPUs available to us and information about these GPUs, we also wish to know the limits they will impose on model sizes we can feasibly train, $\mu$. This is the question above.

### 5.1 How Long Will Training Take?

We begin with the relationship between computational workload, effective throughput, and elapsed time,

$$
\tau = \frac{C(\mu)}{\eta P N_{\text{GPU}}}.
$$

Substituting the computational workload from our scaling equations,

$$
\begin{aligned}
\tau &\approx \frac{6D\mu}{\eta P N_{\text{GPU}}} \\
  &\approx \frac{6r\mu^2}{\eta P N_{\text{GPU}}}.
\end{aligned}
$$

This provides training time in [s]. To express the result in [hr], we multiply by the conversion factor $1\ \text{hr}/3600\ \text{s}$:

$$
\boxed{
\tau_{\text{hrs}}\approx
\frac{6r\mu^2}{\eta P N_{\text{GPU}}}
\left(\frac{1\ \text{hr}}{3600\ \text{s}}\right)
}
$$

where $\tau_{\text{hrs}}$ is the numerical training time expressed in hours when the conversion is applied, and:

| Symbol | Description | Units |
|---|---|---|
| $\mu$ | Number of model parameters | [parameters] |
| $r$ | Tokens-per-parameter ratio | [tokens/parameter] |
| $N_{\text{GPU}}$ | Number of GPUs | [GPUs] |
| $\eta$ | Model FLOPs utilization | [dimensionless] |
| $P$ | Peak throughput per GPU | [FLOPs/s] |
| $\tau$ | Training wall time | [s] |

We can summarize this relationship as

$$
\tau = k \frac{\mu^2}{N_{\text{GPU}}}
$$

where

$$
k = \frac{6r}{\eta P}.
$$

Note that $\tau$ is expressed in seconds here, and the appropriate conversion factor must be applied if it is to be expressed in hours or days.

With the number of GPUs remaining equal, training time scales quadratically with model size. Inversely, to train a model of a given size in a fixed amount of time, quadratically increasing GPUs are required. Now we are beginning to arrive at the relationships we need to understand what time and computing resources it takes to train a model.

### 5.2 How Many GPUs are Required?

The two factors that drive the number of GPUs we will need are throughput and memory of available GPUs.

#### 5.2.1 Throughput-Limited GPU Count

To address throughput, we can rearrange our compute scaling equation for the number of GPUs driven by compute workload,

$$
N_{\text{compute}}(\mu, \tau) = k \frac{\mu^2}{\tau}, \quad k = \frac{6r}{\eta P}.
$$

The number of GPUs driven by our computing workload is given by the ceiling of this number.

$$
N_{\text{compute}}(\mu, \tau)  = \left\lceil k \frac{\mu^2}{\tau} \right\rceil, \quad k = \frac{6r}{\eta P}.
$$

If we wish to specify P in TOPs:

$$
\boxed{
N_{\text{compute}}(\mu, \tau) =
    \left\lceil k \frac{\mu^2}{\tau} \right\rceil,
    \quad k = \frac{6r}{\eta P_{TF} \times 10^{12}}
}.
$$

If we wish to specify $\tau$ in hours, $\tau_{\text{hrs}}$:

$$
\boxed{
N_{\text{compute}}(\mu, \tau_{\text{hrs}}) =
    \left\lceil k \frac{\mu^2}{3600 \times \tau_{\text{hrs}}} \right\rceil,
    \quad k = \frac{6r}{\eta P_{TF} \times 10^{12}}
}.
$$

Similarly, if we wish to specify $\tau$ in days, $\tau_{\text{days}}$:

$$
\boxed{
N_{\text{compute}}(\mu, \tau_{\text{days}}) =
    \left\lceil k \frac{\mu^2}{86400 \times \tau_{\text{days}}} \right\rceil,
    \quad k = \frac{6r}{\eta P_{TF} \times 10^{12}}
}.
$$

In all of these relationships, $\eta$ is the MFU as a ratio, $r$ is the ratio of tokens to parameter, and $P_{TF}$ is the manufacturer-specified peak throughput in TFLOPs/s.

#### 5.2.2 Memory-Limited GPU Count

Now to address the GPU count imposed by memory limitations, recall first the total memory to store model training states,

$$
M_{\text{train}}(\mu) \approx 16\mu.
$$

Then recall the available memory of a GPU to store model state during training is the total memory, $M$, that GPU has available reduced by a fraction that is reserved for other overhead:

$$
M_{\text{avail}}=M(1-f_R).
$$

Assuming the training states can be fully *sharded*, or distributed, across $N$ identical GPUs, the total available memory becomes

$$
M_{\text{avail}}=N_{\text{GPU}}M(1-f_R)\ [\text{bytes}].
$$

The number of GPUs required as driven by memory needs, $N_{\text{memory}}$ is given by dividing the memory required to train the model, $M_{\text{train}}(\mu)$ by the memory available on each GPU accounting for the reserve, $M_{\text{avail}}$. Since we cannot purchase fractional GPUs, we take the ceiling for a conservative estimate:

$$
\begin{align*}
N_{\text{memory}}(\mu) &= \left\lceil\frac{M_{\text{train}}(\mu)}{M_{\text{avail}}}\right\rceil, \\
    &= \left\lceil\frac{16\mu}{M(1 - f_R)}\right\rceil.
\end{align*}
$$

This provides the GPU count we need accounting for memory requirements.

$$
\boxed{
N_{\text{memory}}(\mu) = \left\lceil\frac{16\mu}{M(1 - f_R)}\right\rceil
}.
$$

So the number of GPUs as driven by memory will grow linearly with respect to the model parameters, $\mu$. Here $M$ is the manufacturer-specified memory per GPU in bytes and $f_R$ is the fraction of memory reserved for overhead (e.g., $f_R = 0.25$).

#### 5.2.3 Combined Compute and Memory Scaling

With expressions for the number of GPUs needed to support our workload, $C(\mu)$ in a given time, $\tau$, and the number of GPUs needed to fit our model state in memory during training, we can now determine the final number of GPUs needed to train our model as the maximum of the two, i.e.,

$$
\boxed{
N_{\text{GPU}}(\mu, \tau) = \max\{
    N_{\text{compute}}(\mu, \tau_{\text{hrs}}),
    N_{\text{memory}}(\mu)
\}
}.
$$

Mathematically, we see that the number of GPUs required to train a model is therefore given as the maximum of a quadratic staircase function of the number of parameters,

$$
\left\lceil\left(\frac{k_C}{\tau}\right)\mu^2\right\rceil,\ k_C = \frac{6r}{\eta P}
$$

and a linear staircase function of the number of parameters,

$$
\left\lceil k_M \mu \right\rceil,\ k_M = \frac{16}{M(1 - f_R)}.
$$

More compactly, we can write a unified mathematical scaling heuristic as

$$
\boxed{
N_{\text{GPU}} = \max\left(
    \left\lceil\left(\frac{k_C}{\tau}\right)\mu^2\right\rceil,
    \left\lceil k_M \mu \right\rceil
\right),\ k_C = \frac{6r}{\eta P},\ k_M = \frac{16}{M(1 - f_R)}
}.
$$

As before, $r$ is the ratio of tokens to parameter, $\eta$ is the MFU as a fraction, $P$ is the manufacturer-specified peak throughput for a given GPU, $M$ is the manufacturer-specified total memory per GPU in bytes, and $f_R$ is the fraction of memory held in reserve.

We can see from these relationships that for fast training runs with a small $\tau$, the quadratic term will dominate the number of GPUs we need. Conversely, if we are willing to wait longer to train a model and accept a large $\tau$, we might still require several GPUs to fit our model in memory, even if our computing workload can in theory be executed on fewer GPUs.

### 5.3 How Large of a Model Can Be Trained?

Similarly, given a number of GPUs and a fixed training time budget, we can rearrange the equations above to ask how large of a model we can feasibly train.

As before, we must consider two constraints: the computational workload we can complete in the available time, and the memory available to store the model's training states.

#### 5.3.1 Compute-Limited Model Size

Recall that the time required to train a model using multiple GPUs is approximately

$$
\tau \approx \frac{6r\mu^2}{\eta P N_{\text{GPU}}}.
$$

We can rearrange this equation to solve for the model size, $\mu$, by first multiplying both sides by $\eta P N_{\text{GPU}}$:

$$
\tau\eta P N_{\text{GPU}} \approx 6r\mu^2.
$$

Dividing by $6r$ and taking the square root gives

$$
\boxed{
\mu_{\text{compute}} \approx
\sqrt{
\frac{\tau\eta P N_{\text{GPU}}}{6r}
}
}.
$$

Here $\mu_{\text{compute}}$ is the maximum model size, in parameters, that can be trained within our time budget under the simplified compute scaling approximation.

If the GPU throughput is specified in TFLOPs/s and training time is specified in days, we can write

$$
\boxed{
\mu_{\text{compute}} \approx
\sqrt{
\frac{
86400\tau_{\text{days}}\eta
P_{\text{TF}}10^{12}N_{\text{GPU}}
}{6r}
}
}.
$$

The factor of 86400 converts days to seconds, while $10^{12}$ converts TFLOPs/s to FLOPs/s.

Notice that the maximum model size grows with the square root of the number of GPUs and the available training time. For example, doubling the number of GPUs does not allow us to double the model size. Instead, it increases the compute-limited model size by a factor of approximately $\sqrt{2}$. To double the model size while keeping the training time fixed, we would need approximately four times as many GPUs, assuming the same MFU and tokens-per-parameter ratio.

#### 5.3.2 Memory-Limited Model Size

Next, we consider the maximum model size that can fit in GPU memory.

Recall that the total memory required to store our model's training states is approximately

$$
M_{\text{train}}(\mu)\approx16\mu.
$$

Assuming full sharding across $N_{\text{GPU}}$ identical GPUs, the total memory available for these training states is

$$
M_{\text{avail,total}} =
N_{\text{GPU}}M(1-f_R).
$$

For the model's training states to fit in memory, we require

$$
16\mu \leq N_{\text{GPU}}M(1-f_R).
$$

Rearranging to solve for $\mu$ gives

$$
\boxed{
\mu_{\text{memory}} =
\frac{N_{\text{GPU}}M(1-f_R)}{16}
}.
$$

Here $\mu_{\text{memory}}$ represents the maximum number of parameters whose estimated training states can fit in the available GPU memory.

Unlike the compute-limited model size, the memory-limited model size grows linearly with the number of GPUs. Doubling the number of GPUs approximately doubles the available aggregate memory and therefore doubles the number of parameters whose training states can be stored. This assumes that model weights, gradients, and optimizer states can be fully sharded across the GPUs, and that the memory reserve adequately accounts for activations and other overhead.

#### 5.3.3 Combined Compute and Memory Limits

To train a model successfully, it must satisfy both the compute and memory constraints. Therefore, the maximum feasible model size is limited by whichever constraint is more restrictive.

We can express this relationship as

$$
\boxed{
\mu_{\text{max}} =
\min\left(
\mu_{\text{compute}},
\mu_{\text{memory}}
\right)
}.
$$

Substituting our expressions gives

$$
\boxed{
\mu_{\text{max}} =
\min\left(
\sqrt{
\frac{\tau\eta P N_{\text{GPU}}}{6r}
},
\frac{N_{\text{GPU}}M(1-f_R)}{16}
\right)
}.
$$

A before, $\tau$ is the training time budget, $\eta$ is the MFU fraction, $N_{\text{GPU}}$ is the number of GPUs available, $r$ is the ratio of tokens to parameters, $M$ is the memory per GPU, and $f_R$ is the fraction of GPU memory held in reserve. This is the inverse of our earlier GPU-count calculation and gives us the model size we can train on a fixed deadline given the number of GPUs available to us.

As with the previous calculation, if our training is compute-limited, adding more GPUs or allowing more training time will enable us to train larger models. If our training is memory-limited, increasing the available training time will not help us fit a larger model into memory.

## 6 Summary of Scaling Relationships

The fundamental scaling relationships are summarized below for convenience.

| Quantity | Approximate Relationship | Scaling with Model Size, $\mu$ |
|---|---|---|
| Training tokens | $D(\mu) = r\mu$ | Linear |
| Training compute | $C(\mu) = 6r\mu^2$ | Quadratic |
| Training-state memory | $M(\mu)_{\text{train}} = 16\mu$ | Linear |
| Training wall time | $\tau(\mu) = \frac{6r\mu^2}{\eta P N_{\text{GPU}}}$ | Quadratic |
| Compute-limited GPU count | $N_{\text{compute}}(\mu, \tau) = \left\lceil\frac{6r\mu^2}{\tau\eta P}\right\rceil$ | Quadratic |
| Memory-limited GPU count | $N_{\text{memory}}(\mu) = \left\lceil\frac{16\mu}{M(1-f_R)}\right\rceil$ | Linear |

The total number of GPUs required is the maximum of the compute-limited and memory-limited GPU counts:

$$
\boxed{
N_{\text{GPU}} =
\max\left(
N_{\text{compute}},
N_{\text{memory}}
\right)
}.
$$

Conversely, given a fixed number of GPUs and an available training time, the maximum model size is

$$
\boxed{
\mu_{\text{max}} =
\min\left(
\mu_{\text{compute}},
\mu_{\text{memory}}
\right)
}.
$$

### 6.1 Observations

From these relationships, we can make a few helpful observations that can guide our thinking about model scaling going forward.

First, we observe that *training compute grows quadratically with model size.* Under the Chinchilla scaling heuristic, doubling the number of parameters also doubles the number of training tokens, resulting in approximately four times the computational workload. This means there will likely be an inflection point in model scale where training a given size model quickly becomes out of our reach on a fixed budget.

Next, we observe that although training compute grows quadratically, *training-state memory grows linearly with model size.* Doubling the number of parameters approximately doubles the memory required to store the model's weights, gradients, and optimizer states. This means that even if we have enough compute to handle our training workload, we might be limited by available memory. This can be the main limiting factor in longer training runs. We can wait longer for a model to finish training to train a larger model on fewer GPUs, but still might be limited by available memory.

When determining how long it may take to train a model, we observe that under the ideal scaling laws we have been using, *doubling the number of GPUs halves the time required to complete a fixed computational workload*. In practice, however, communication and other overheads limit this improvement. Theoretical peak throughput is not the same as the throughput achieved during training. MFU provides a way to account for the difference when estimating training time and GPU requirements, but in practice additional complexities need to be accounted for.

### 6.2 Limitations

As with any *back-of-the-envelope* calculation, these relationships are approximations we use for planning purposes and are not exact predictions of training performance. The heuristics come with a number of disclaimers and limitations we need to remain aware of so we know how to modify them when needed. Our estimates assume dense transformer models, a fixed tokens-per-parameter ratio, approximately 16 bytes of training-state memory per parameter, and effective GPU throughput represented by MFU. We also assume that training states can be sharded across multiple GPUs when necessary.

Actual resource requirements will depend on the model architecture, training configuration, numerical precision, sequence length, GPU interconnect, and parallelization strategy. Furthermore, the equations estimate the resources required for a training run, not the performance or accuracy of the resulting model. Selecting an appropriate model size still requires experiments and an understanding of the problem being solved.

For early project planning, however, these relationships provide a useful way to estimate whether a proposed training workload is feasible and to compare the computing resources required for different model sizes and GPU architectures.

## 7 Example: 7-Billion-Parameter Model

To illustrate how to conduct the analysis we built up here, we will conclude with an example. Suppose we wish to train a dense transformer model with 7 billion parameters in 30 days. We have access to GPUs with the following specifications:

| Parameter | Value |
|---|---|
| Model size, $\mu$ | $7 \times 10^9$ parameters |
| Tokens per parameter, $r$ | 20 |
| GPU peak throughput, $P_{\text{TF}}$ | 300 TFLOPs/s |
| GPU memory, $M$ | 80 GB |
| Model FLOPs utilization, $\eta$ | 0.4 |
| Memory reserve, $f_R$ | 0.25 |
| Training time, $\tau_{\text{days}}$ | 30 days |

(These GPU specifications and utilization assumptions are illustrative.)

**Step 1: Estimate Training Tokens**

Using the Chinchilla relationship,

$$
D = r\mu = 20(7\times10^9)
= \boxed{1.4\times10^{11}\text{ tokens}}.
$$

We estimate that training will process approximately 140 billion tokens.

**Step 2: Estimate Computational Workload**

Using the simplified PaLM approximation,

$$
\begin{aligned}
C &\approx 6r\mu^2\\
  &= 6(20)(7\times10^9)^2\\
  &= \boxed{5.88\times10^{21}\text{ FLOPs}}.
\end{aligned}
$$

**Step 3: Estimate Compute-Limited GPU Count**

We can calculate the number of GPUs needed to complete this workload within 30 days:

$$
N_{\text{compute}} =
\left\lceil
\frac{6r\mu^2}
{86400\tau_{\text{days}}\eta P_{\text{TF}}10^{12}}
\right\rceil.
$$

Substituting our values,

$$
\begin{aligned}
N_{\text{compute}}
&= \left\lceil
\frac{5.88\times10^{21}}
{86400(30)(0.4)(300\times10^{12})}
\right\rceil\\
&= \boxed{19\text{ GPUs}}.
\end{aligned}
$$

**Step 4: Estimate Memory-Limited GPU Count**

Using the 16-byte-per-parameter training-state estimate and reserving 25% of GPU memory for overhead,

$$
N_{\text{memory}} =
\left\lceil
\frac{16\mu}{M(1-f_R)}
\right\rceil.
$$

Substituting our values, with GPU memory expressed in bytes,

$$
\begin{aligned}
N_{\text{memory}}
&= \left\lceil
\frac{16(7\times10^9)}
{80\times10^9(1-0.25)}
\right\rceil\\
&= \boxed{2\text{ GPUs}}.
\end{aligned}
$$

**Step 5: Determine the Required GPU Count**

The final GPU requirement is the larger of the compute and memory requirements:

$$
\begin{aligned}
N_{\text{GPU}}
&= \max(N_{\text{compute}},N_{\text{memory}})\\
&= \max(19,2)\\
&= \boxed{19\text{ GPUs}}.
\end{aligned}
$$

In this example, training is compute-limited rather than memory-limited. Although the model's estimated training states can fit across just two GPUs, completing the training workload within 30 days requires approximately 19 GPUs given the specifications we have used.

If we were willing to wait longer for training to complete, we could use fewer GPUs, but we would still need at least two GPUs to satisfy our required memory estimate under the assumed sharding strategy.

As with all the calculations presented here, these estimates assume that the model can be distributed efficiently across the GPUs and that the assumed MFU and memory reserve remain valid. In practice, MFU and memory reserve required can vary based on model architecture and implementation details of the model and computing cluster it is trained on.

## 8 References

1. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). https://arxiv.org/abs/2203.15556
2. Chowdhery et al. (2022). *PaLM: Scaling Language Modeling with Pathways*. https://arxiv.org/abs/2204.02311
3. Rajbhandari et al. (2020). *ZeRO: Memory Optimizations
   Toward Training Trillion Parameter Models*.
   https://arxiv.org/abs/1910.02054
4. PyTorch. *Fully Sharded Data Parallel (FSDP) Tutorial*.
   https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html
