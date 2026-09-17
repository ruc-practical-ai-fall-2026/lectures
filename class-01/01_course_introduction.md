# Course Introduction: Practical AI

## Introduction

Hello and welcome to Practical AI!

This is a one-semester advanced course on the practical aspects of designing, developing, evaluating, and deploying artificial intelligence (AI) applications. The course material will cover and apply to a broad span of AI methods and a similarly broad span of applications, with an in-depth focus on deep-learning-enabled perception methods, i.e., methods for using deep neural networks to detect, classify, and track patterns or objects in data, and an introduction to generative and foundation models.

The theory of some common deep learning architectures will be briefly reviewed, but as we will see throughout the semester, knowing the theory of a specific AI technique or architecture in isolation is not enough to enable us to design, develop, and deploy robust systems that perform reliably in the real world. Practical AI systems must operate under changing data distributions, environments, users, and requirements. They must also be evaluated appropriately, integrated with other software and hardware, monitored after deployment, and designed with an understanding of the limitations and failure modes of the models they contain.

See the course syllabus for information on the course outline, expectations, schedule, and topics. Here we will review some motivating examples that show why it is important to consider the practical aspects of implementing an AI system. Some of these examples may be familiar and some may be new. By the end of this course, however, we will have built the concepts needed to understand all of these examples in depth and to implement mitigations for many of them in the practical AI systems we will design.

We will start with a discussion of the relevance of AI to modern life. We will then look at a few key definitions of technical terms related to AI before moving on to discuss some example challenges that AI systems need to address to perform reliably in the real world.

## Real-World Context and Motivation

Due to the rapidly increasing prevalence of AI systems, we often hear about the significant impacts, both positive and negative, that the technology may have on our daily lives and on broader society. Investment in AI has accelerated dramatically, and established technology companies compete alongside newer entrants to develop increasingly capable models, computing infrastructure, and AI-enabled products. At the same time, AI is becoming less of a specialized feature and more of a general-purpose component of modern software systems. Products that we have been familiar with for decades, such as cell phones and cars, are increasingly augmented with AI capabilities. AI systems are now routinely incorporated into search engines, productivity software, programming tools, scientific workflows, consumer electronics, and industrial systems. Generative AI systems can produce text, images, audio, video, and software, and AI systems are also increasingly able to interact with external software and tools rather than simply producing predictions or responses.

Meanwhile, AI-enabled systems are increasingly employed in safety-critical industries and applications, such as medicine, defense, transportation, and cybersecurity, with critical stakes for individuals, groups, and societies. AI has reportedly been employed in recent global conflicts, while world powers compete for leadership in the technology and adopt different policies governing its development and use. These differences raise important technical, political, and ethical considerations. In medicine, for example, AI systems have demonstrated useful performance on narrow tasks such as medical-image analysis and disease screening, and AI-enabled medical devices are already used across a growing range of applications. However, the tendency for prominent AI techniques to produce the correct answers for the wrong reasons still presents significant risks. Even with progress in *explainable AI*, it is not always clear why an algorithm produces the result it does, which makes it difficult to predict if strong performance on a benchmark dataset will translate to strong performance in real-world applications where data distributions shift, edge cases present themselves, and emergent behaviors arise in large systems. Generative AI or AI agents may produce plausible but incorrect information, expose sensitive information, interact incorrectly with external tools, or be manipulated through adversarial inputs such as prompt injection.

Autonomous vehicles provide another useful example. These systems must accurately detect and interpret vehicles, pedestrians, road markings, signs, and other features of a constantly changing environment. Their sensors and algorithms must continue operating under variations in lighting, rain, fog, road conditions, construction, unusual objects, and human behavior. The engineering considerations that arise in the design of these systems often bring philosophical questions once considered a classroom exercise (such as variations on the famous trolley problem in ethics) to the heart of engineering design discussions for the specialists who design these systems. Designers must consider uncertainty, redundancy, failure detection, degraded operating conditions, appropriate fallback behavior, and how the complete system responds when individual components fail.

As with any new technology, the utility of each AI capability introduced to the world must be balanced with the unintended side effects it creates, and hype around the technology must be tempered with the reality of its limitations. A system that works in a demonstration environment is not necessarily ready for the complexity and unpredictability of the real world! Similarly, as with other technological advances that impacted critical applications, such as the prevalence of software in modern devices, the invention of new modes of travel (e.g., sea and air travel), and even the use of electricity itself, the maturation of AI from a field of experimental study to a mature practice suitable for real-world deployment is happening now, as practitioners adopt and implement a rich repertoire of safeguards and practices for evaluating, deploying, monitoring, and maintaining AI systems, similar to the standards of practice that now exist for software engineering, aviation, maritime transportation, and codes governing electrical systems. Practitioners will need to adopt these safeguards both proactively, to prevent foreseeable incidents involving AI technology, and reactively, to ensure lessons learned from previous implementations are not forgotten.

For AI specifically, the National Institute of Standards and Technology (NIST) AI Risk Management Framework (AI RMF) provides a widely applicable, voluntary framework for managing risks associated with the design, development, deployment, evaluation, and use of AI systems. NIST has also published additional guidance addressing risks specific to generative AI. We will return to these frameworks throughout the course as we encounter practical examples of reliability, robustness, security, evaluation, and responsible deployment.

For a growing list of incidents in which AI systems deployed to the real world have caused harms to individuals, groups, and societies, we will continually reference the AI Incident Database, which maintains a living list of such instances. The incidents documented range from annoyances for users of systems, to harmful injustices, and tragic outcomes with injuries and lives lost from those who were not using an AI system directly. It is not necessary to read or know all these references to proceed with this course (except for those interested!) but it is necessary to realize the role that we all play as AI practitioners in adopting responsible practices as the AI field matures, to prevent recurrence of past AI incidents and proactively prevent future AI incidents.

Let's take some time to look around the [AI Incident Database](https://incidentdatabase.ai/).

### References

* National Institute of Standards and Technology (NIST), [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://airc.nist.gov/)

* National Institute of Standards and Technology (NIST), [Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile (NIST AI 600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)

* [AI Incident Database](https://incidentdatabase.ai/)

### Further Reading

AI algorithms are implemented in software, and the practices of responsible software engineering are both essential to the development of AI systems and an important example of the evolution of a critical field into a rigorous engineering discipline that runs much of the modern world! For a description of the prevalence of software in our modern world and the important responsibility that all software craftspeople have to produce maintainable, and reliable code through the adoption of hard-won principles and practices, see Robert C. Martin's *Clean Code* lectures and book.

## Introduction to Artificial Intelligence and Machine Learning

Before we discuss some challenges to the implementation of AI systems that perform well in the real world and provide a preview to how this course will help us address these challenges, we need to define some key terms in AI and related fields.

### Artificial Intelligence

In this class, we will use the definition of *Artificial Intelligence (AI)* from *AI - A Modern Approach*, by Stuart Russell and Peter Norvig. Russell and Norvig define AI as the study of *intelligent agents* that receive *percepts* from the environment and perform *actions*. In this context, an intelligent agent refers to a machine that can compute how to act effectively to achieve a goal in a given environment. A percept in this context is the visual input an intelligent agent receives, which due to limited capacity, sensor errors, noise, perspective, and many other real-world considerations, will not be the same as the true object of perception. In this course, we further emphasize that useful intelligent agents should act both effectively and safely, i.e., achieve their intended goals without causing unacceptable harm or unintended consequences.

Consider what happens when many humans observe the same event (e.g., a car accident) and then provide differing accounts of the event. This is due to the differences between their percepts and the true object of perception. In some cases these differences are due to bias, which can happen in computers as well as humans, but in many cases, they are due to more fundamental ambiguities. Humans and machines alike are limited (in different ways) and not all-knowing or all-perceiving. Objects can become occluded, weather can impair our ability to see them, they can become so small they are hard to see, etc. and all these effects vary based on the specifics of the situation at hand. Accounting for the differences between percepts and objects of perception is at the heart of this course and will be further discussed below.

This course further focuses on the software aspects of the study of intelligent agents. In our studies, we will be using standard computers (local laptops, servers, and cloud resources) as our machines, and will be making them act as *intelligent agents* by programming them with carefully designed *algorithms*, i.e., programmed steps to solve a problem or achieve a goal. We will not be studying the hardware aspects of building robots that act in the world. For this reason, we will also be more focused on the perception and information-processing tasks that agents must perform rather than *physical* action tasks, though much of the course material is general and broadly applicable to these systems as well.

### Machine Learning

*Machine learning (ML)* refers to a subclass of AI algorithms which *learn from data* rather than needing to be explicitly programmed to perform a task. While ML algorithms themselves are still programmed, the behavior of the learned algorithm, e.g., the patterns it recognizes or generates, is largely defined by training data and runtime inputs. ML algorithms are provided with datasets as input, then learn to solve a problem from the dataset. The output of a learning algorithm is a *model* which is fit to the input dataset such that it recognizes the desired patterns or generates the desired outputs. In comparison, conventional algorithms are designed by programming the explicit steps that a human designer thinks need to be taken to solve the problem. The problem itself may be defined by a set of requirements or expected input-output test cases. The act of using a dataset to generate a *model* is often called *learning* or *fitting* while the act of using the model learned from the data to perform tasks on new inputs is often called *inference* or *prediction*. It is worth emphasizing that learning and inference represent distinct algorithms in most learning paradigms. The learning process accepts a dataset as input, carries out the learning algorithm, and then outputs the *parameters* or, in many neural network models, *weights*, which define the model. The inference process uses the weights and any runtime inputs as inputs to an algorithm having behavior defined by these parameters, to produce runtime outputs. In general, learning a model from data requires possessing or collecting training and evaluation data that is representative of the real-world conditions an algorithm will ultimately be deployed to. The main motivations for using ML to generate models are:

1. Faster development: explicitly programming algorithms is often time consuming and error prone. When applied well, ML can enable tuning algorithms faster than developing them by hand.
2. More accurate algorithms: often ML algorithms will be able to uncover patterns that human experts may not have observed. This is one of the most powerful benefits of ML, but also one of the main reasons ML requires care in application. Algorithm designers applying ML must be careful to ensure it learns the right result for the right reason. More powerful ML algorithms can make this difficult to determine, leading to a tradeoff between accuracy and interpretability that will be discussed often in this course. More accurate algorithms are often (though not always) less interpretable, and designers must determine which is higher priority based on their use cases. This can be difficult since critical applications often require both accuracy and interpretability.
3. Faster update times: for any algorithms, ML or explicitly programmed, updates will be required due to changes to the data they receive (*data drift*), changes to the relationship between input and target output data (*concept drift*), changes to the system in which they reside (*system evolution*), changes to the way users want to employ them, and routine maintenance. When the reasons for an update can be defined by data, ML algorithms can often be updated faster than conventionally programmed algorithms by acquiring data from the environment and using the data to train and update the algorithm. Since the software remains the same, this process can be less prone to *software-specific* bugs introduced by manually updating program logic. Since the new data is acquired from the real world, when proper ML practice is followed this process can also effectively account for real-world effects which can otherwise break algorithms.

A key theme in this course will be understanding the best practices that must be applied to realize these benefits in real-world applications. Learning algorithms from data introduces new risks and the need for continuous evaluation of algorithms throughout their lifecycle. Risks which must be managed to realize the benefits of ML algorithms in practice include:

* *Label noise*: errors in the labels which must be corrected or, if not corrected, handled with an appropriate learning strategy.
* *Data impairments and corruption*: noise, corruption, or human errors in collected data can reduce training or inference performance.
* *Data leakage*: - leaking of information between the training and evaluation datasets can cause overly-optimistic performance evaluations, shortcut learning, and other issues. This can happen in unexpected ways that are less intuitive than "training on the test data," such as choosing the wrong sampling or data splitting strategies, or artifacts from data collection sensors, instruments, or processes leaving clues as to the source of a dataset in the data.
* *Data drift*: - changes in the distributions of input data can lead to reduced performance or unexpected behavior.
* *Concept drift*: - changes in the relationship between input and target output data can lead to unexpected outputs and drive a need to update ML algorithms.
* *Catastrophic forgetting*: - for learning paradigms that require sequentially training and retraining, training on new data can lead to forgetting of previously correct input-output relationships.

Examples of ML algorithms, model architectures, and types of models that we will discuss include:

* Regression
* Perceptrons
* Multi-layer perceptrons
* Support vector machines
* Decision trees
* Random forests
* Gradient-boosted trees
* Deep neural networks (DNNs)
* Recurrent neural networks (RNNs)
* Long short-term memory networks (LSTMs)
* Transformers
* Large language models (LLMs)
* Large foundation models for vision, language and other tasks
* Ensembles and systems comprised of multiple of the above components

It is not necessary to know exactly what each of these are and how they work at this stage in the course (though that background is helpful), but you may have encountered them in other classes, or even daily life, and it is helpful to begin associating them with the course material.

## Key Terms

### Artificial Intelligence and Machine Learning

#### Artificial Intelligence (AI)
In this course, *artificial intelligence* refers both to (1) the study and engineering of systems that perceive, reason, learn, and act in ways that support the achievement of goals, and (2) the broad set of algorithms that perform perceive and act in environments or upon information.

#### Machine Learning (ML)
*Machine learning* refers to the subset of AI algorithm which learn parameters of a model from data or experience rather than requiring that all behavior be explicitly programmed.

#### Deep Learning
*Deep learning* refers to the further subset of machine learning approaches which employs models composed of multiple, often many, layers of learned representations. Modern deep learning is most commonly implemented using multilayer neural networks, i.e., deep neural networks (DNNs).

#### Algorithm
An *algorithm* is a finite sequence of computational steps used to solve a problem.

#### Learning Algorithm
A *learning algorithm* is an algorithm that uses data or experience to produce or update a model.

#### Model
A *model* is a mathematical or computational representation learned or fit to data for use in performing a task, such as predicting an output from an input.

#### Training, Learning, or Fitting
In this course, *training*, *learning*, and *fitting* are all synonyms for the process by which a learning algorithm uses data and/or experience to compute or update the parameters and/or structure of a model.

#### Inference
*Inference* refers to the process of applying a trained model to inputs to produce predictions, decisions, representations, or other outputs at runtime.

#### Generalization
*Generalization* refers to the ability of a learned model to maintain strong performance suitable for its application on data or in environments that were not used to train the model.

### Systems Terminology

#### System
A *system* is a collection of interacting components organized to perform one or more functions within an environment.

#### Component
A *component* is a constituent part of a larger system that performs a particular function and interacts with other parts of the system or the environment through defined interfaces. An *AI component* can refer to a component of a system that employs AI algorithms to perform its functions.

#### Subsystem
A system that is a component of a larger system is called a *subsystem* of the top-level system.

#### AI System
A system containing one or more AI components whose outputs contribute to the behavior or decisions of the overall system can be referred to as an *AI system*. An AI system may also contain conventional software, hardware, data pipelines, users, operators, and organizational processes.

#### Socio-Technical System
A system having behavior dependent on interactions between technical components and people, organizations, processes, and the environment in which the system operates is called a *socio-technical* system. NIST treats AI systems as socio-technical systems rather than purely computational artifacts.

#### System Boundary
The *system boundary* is the conceptual boundary that specifies what is considered part of a system and what is considered part of its external environment. In practice, a system boundary can evolve over time as new components are added and interaction with the environment changes.

#### Environment
Everything outside a system that can affect the system or be affected by it is considered part of the *environment* the system operates in.

#### Interface
An interface is a specified point of interaction between components, systems, users, or the system and its environment.

#### Input, Output, and State
Information, signals, or actions which are supplied to a system, subsystem, or component, are called the *inputs* of the system. Information, signals, or actions produced by a system, subsystem, or component are called *outputs* of the system. Information which describes the condition of a system at a particular moment in time is called the *system state*. Inputs can, but do not always, change system state. Outputs can, but do not always, reveal system state. Some parameters of the system state are often unobservable outside the system.

#### Requirement
A statement of a capability, behavior, constraint, or performance measure that a system is expected to satisfy is called a *requirement*. Requirements can specify functions the system must possess (*functional* requirements) or performance metrics it must meet (*performance* requirements). In traditional systems engineering, requirements are often formulated as *shall* statements, e.g., "the sensor system shall detect the presence of vehicles from 50 meters slant range".

#### Context of Use and Operational Design Domain
The conditions under which a system is intended to operate, including its users, environment, purpose, assumptions, and constraints is called a system's *context of use*. Similarly, the term *Operational Design Domain* is commonly used in autonomous driving and other fields developing autonomous systems to refer to the specific set of operating conditions, environments, and boundaries under which an automated system is designed to function. Context of use and operational design domains are especially important when evaluating AI risks and whether observed performance is sufficient deployment in the specific production application.

### System Behavior and Analysis

#### White-Box Systems and Methods
A *white-box system* is one whose internal subsystems and components are known and available for analysis. Typically, this would mean an equation or software program can be written to model the system. *White-box methods* are those which rely on having a *white-box view* or *whitebox access* to a system.

#### Black-Box Systems and Methods
A *black-box system* has known inputs and outputs which can be collected by experimentation or probing, but unknown internals. A *black-box method* makes no assumptions about the internals of a system.

#### Gray-Box Systems and Methods
A *gray-box system* is one having partially known internals. For example, some structure might be known but a precise equation or the precise parameters of that equation might be unknown.

#### Linear System
A *linear system* has output directly proportional to its input, such that the relationship between all outputs and inputs can be described as linear (e.g., straight lines) functions. Linear systems obey the superposition principle, i.e., providing the sum of two inputs will yield an output equal to the sum of the outputs that would have resulted from each input individually. We will review linearity more thoroughly in later material.

#### Nonlinear System
A *nonlinear system* has output which cannot be described as a linear function of its input, and therefore does not obey superposition. Most real-world systems and many machine learning models are nonlinear.

#### Deterministic System
A *deterministic system* is a system where the same system state and inputs produce the same outputs and subsequent state under identical initial conditions.

#### Non-Deterministic System
A *non-deterministic system* has behavior not completely determined by its observable current state and inputs. Non-determinism may arise from randomness, unobserved variables, measurement noise, or other factors. Non-deterministic systems are often modeled as *stochastically*, i.e., using random variables.

#### Robustness
In this course, *robustness* refers to the ability of a system maintain acceptable performance when inputs, operating conditions, domain, or assumptions vary within some pre-defined range.

#### Resilience
In this course, *resilience* refers to the ability of a system to withstand, recover from, or adapt to failures, disruptions, unexpected conditions, or attacks. NIST treats resilience as broader than ordinary robustness and includes unexpected or adversarial use. In this course, we will aim to create systems that are resilient to environmental and adversarial conditions.

### Types of Learning

There are three basic types of learning we will encounter in this course.

#### Supervised Learning
*Supervised learning* refers to learning a dataset for which labels, or a mapping from inputs to target outputs, are available.

#### Unsupervised Learning
*Unsupervised learning* refers to extracting useful structure, representations, distributions, or relationships from data *without* externally provided target outputs or labels.

#### Reinforcement Learning
*Reinforcement learning* is done through *interaction with an environment* so as to maximize a measure of cumulative reward.

#### Additional Types of Learning

We will also encounter all of the following types of learning.

* *Self-Supervised Learning*: Learning in which target values are generated automatically from the data itself rather than supplied by an external annotator. The training objective is typically supervised, but the supervision is constructed from otherwise unlabeled data.
* *Semi-Supervised Learning*: Learning from a dataset containing both labeled and unlabeled examples.
* *Weakly Supervised Learning*: Learning from supervision that is incomplete, imprecise, noisy, indirect, or otherwise weaker than fully labeled training examples.
* *Active Learning*: A learning approach in which the learning algorithm selects examples or observations for which obtaining additional labels or information is expected to be particularly useful.
* *Transfer Learning*: Using knowledge or representations learned for one problem, dataset, or domain to improve learning or performance on another.
* *Continual Learning / Lifelong Learning*: Learning in which a system continues to acquire or update knowledge over time as new data or tasks become available, ideally without losing previously acquired capabilities.

Though often referred to as different learning paradigms, these learning types are built from the fundamental elements of the basic three: supervised, unsupervised, and reinforcement learning.

### Data Terminology

#### Sample
A *sample*, or synonymously, *example*, or *instance* is an observation supplied to or used by a learning algorithm, commonly consisting of an input and, in supervised learning, an associated label or target.

#### Feature / Attribute
A *feature* or *attribute* is a measurable or computable property of an example that is used as an input to a model.

#### Target / Label
A *target* or *label* is a desired output associated with a supervised learning example. The term label is generally used for categorical data while the term target is used more generally, e.g., for regression tasks.

#### Dataset
A *dataset* is a collection of examples used for training, evaluating, or operating a machine learning system.

#### Training, Validation, and Testing Sets
The *training* dataset is the portion of a dataset used to fit the model. The *validation* dataset is used during development to make choices about models, hyperparameters, stopping criteria, or other aspects of the learning procedure. A *test* dataset is held apart from training and model selection entirely and used to estimate the performance of the final selected model on completely unseen examples.

### Learning Terminology

#### Parameter
A *parameter* is a value within a model that is learned or otherwise determined during training.

#### Hyperparameter
A *hyperparameter* is a setting that controls the model architecture or learning procedure and is normally chosen outside the ordinary parameter-fitting process. Often the difference between parameters and hyperparameters is the time-scales on which they change.

#### Loss and Objective Functions
A *loss function* measures the cost or error of a model's output. Learning algorithms aim to increase performance by reducing loss functions. More broadly, *objective functions* can include additional terms, such as regularization terms, which guide the learning process toward broader objectives beyond simply minimizing loss, often with the aim of yielding a more robust, resilient, or general model. Loss and objective functions are minimized through *optimization* methods.

#### Overfit and Underfit Models
A model which fits the training data well but fails to generalize to new data is referred to as overfit. A model which is insufficiently trained to capture important patterns and structure in the training data is referred to as underfit.

### Common AI Tasks

Though we will not have a chance to perform hands-on assignments with all of these tasks, we will cover and gain familiarity with many of the following common tasks an AI component can perform in a system.

* *Classification*: Assigning an input to one or more discrete categories.
* *Regression*: Predicting a continuous numerical quantity from an input.
* *Detection*: Determining whether specified objects, events, conditions, or patterns are present, often including information about their locations or times.
* *Segmentation*: Partitioning an input into meaningful regions or elements, often by assigning a class or other label to individual pixels, samples, tokens, or other components.
* *Ranking*: Ordering a set of items according to a learned criterion or predicted relevance.
* *Clustering*: Grouping examples so that examples within the same group are similar according to some criterion, without requiring predefined class labels.
* *Dimensionality Reduction*: Representing data using fewer variables while preserving information or structure considered important for subsequent goals or tasks.
* *Representation Learning*: Learning features or representations of data rather than requiring all useful features to be designed manually.
* *Anomaly Detection*: Identifying inputs or events that differ significantly from patterns regarded as normal or previously observed.
* *Generation*: Producing new data, such as text, images, audio, or other structured outputs, based on a learned distribution or model.
* *Prediction*: Estimating an unknown or future quantity from available information.
* *Control*: Selecting actions or inputs that cause a dynamic system to behave in a desired manner.
* *Planning*: Determining a sequence of actions intended to achieve a goal, usually using some representation of possible states, actions, and consequences.
