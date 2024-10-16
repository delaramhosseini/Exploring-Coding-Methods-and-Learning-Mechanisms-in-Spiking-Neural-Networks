# Spiking Neural Networks: Information Coding and Learning Mechanisms

## Overview
### Unsupervised and Reinforcement Learning in Spiking Neural Networks

#### 1. Unsupervised Learning in Spiking Neural Networks
Unsupervised learning in Spiking Neural Networks (SNNs) involves learning patterns from input data without requiring any labels or predefined outputs. This form of learning is inspired by biological mechanisms and is often used for learning useful representations or recognizing patterns from spike trains.

1.1. Spike-Timing-Dependent Plasticity (STDP)
STDP is a learning rule that adjusts synaptic weights based on the timing of spikes from pre- and post-synaptic neurons:
- If a **pre-synaptic neuron spikes just before a post-synaptic neuron**, the synapse is **strengthened** (Long-Term Potentiation, LTP).
- If a **pre-synaptic neuron spikes after the post-synaptic neuron**, the synapse is **weakened** (Long-Term Depression, LTD).

This allows neurons to form connections that reflect frequently encountered patterns in the input data.

1.2. Unsupervised Learning Methods in SNNs
1. **Flat-STDP**: 
   - A simplified version of STDP where weight changes depend on the temporal difference between pre- and post-synaptic spikes.
   - This method helps neurons adapt to input spike patterns without supervision.

2. **Self-Organizing Maps (SOMs) in SNNs**: 
   - SOMs can be implemented using spiking neurons to cluster data into different categories based on similarities in spike patterns. Neurons in the map compete to become the "winner" neuron that best represents the input.

3. **Hebbian Learning in SNNs**: 
   - This general form of learning can be expressed as "cells that fire together, wire together." In SNNs, synaptic changes happen based on correlations between spikes, helping the network learn input features without labeled data.

1.3. Example Use Case
In an SNN, sensory input (e.g., images or sounds) represented as spikes can be used to learn useful features or patterns (e.g., edge detection in images) by adjusting synaptic weights using STDP, without requiring labeled data.

---

### 2. Reinforcement Learning in Spiking Neural Networks
Reinforcement learning (RL) in SNNs involves training based on rewards or punishments received after interacting with an environment. The network learns to adjust its behavior to maximize cumulative reward over time. RL in SNNs is often implemented using **Reward-Modulated STDP (R-STDP)**.

2.1. Reward-Modulated STDP (R-STDP)
R-STDP combines the temporal learning properties of STDP with a reinforcement signal (reward or punishment). The synaptic weights are updated based on spike timing, but the update magnitude is modulated by the reward signal:
- **Positive reward**: Synapses that contributed to the correct output are **strengthened**.
- **Negative reward**: Synapses are **weakened**.

2.2. Reinforcement Learning Methods in SNNs
1. **R-STDP (Reward-Modulated STDP)**:
   - Modifies traditional STDP by introducing a reward signal that adjusts learning to favor actions leading to desired outcomes.
   
2. **Q-Learning with SNNs**: 
   - Combines SNNs with traditional Q-learning, where Q-values (representing the value of actions) are encoded using spike rates or timings.

3. **Actor-Critic Methods with SNNs**:
   - Involves an **actor** (responsible for decision making) and a **critic** (evaluating the actions taken). Both components learn through spike-based encoding, with the actor updating through R-STDP.

#### 2.3. Example Use Case
An SNN-based robot navigating a maze can learn to make decisions (e.g., turning left or right) based on sensory input spikes representing the environment. By receiving rewards for reaching the goal and punishments for hitting obstacles, the robot adjusts its synaptic weights and learns the optimal path through the maze using R-STDP.

---

#### 3. Comparison: Unsupervised vs Reinforcement Learning in SNNs

| Feature                                  | Unsupervised Learning (e.g., STDP)          | Reinforcement Learning (e.g., R-STDP)          |
|------------------------------------------|---------------------------------------------|------------------------------------------------|
| **Learning Paradigm**                    | Learns from patterns in input data          | Learns from interactions with the environment  |
| **Primary Learning Rule**                | STDP (Spike-Timing-Dependent Plasticity)    | R-STDP (Reward-Modulated STDP)                 |
| **Feedback**                             | No feedback (no labels or rewards)          | Feedback through rewards and punishments       |
| **Goal**                                 | Discover input patterns and representations | Maximize cumulative reward                     |
| **Application**                          | Feature extraction, clustering, pattern recognition | Decision making, control tasks         |
| **Neural Activity**                      | Spiking patterns learned based on timing correlations | Spiking patterns modulated by rewards|

---


## Project Objectives
1. **Familiarity with Information Coding Methods**: Understand different ways of encoding information and converting input stimuli into spikes.
2. **Understanding Unsupervised and Reinforcement Learning**: Explore learning paradigms such as unsupervised learning and reinforcement learning.
3. **Understanding Hebbian Learning (Hebb's Law)**: Gain insights into the Hebbian learning process.

## Activities

### **Part One:** Implementing Different Coding Methods

- **Goal**: Implement various methods to transform input stimuli into spikes.

- **Coding Methods**:
  - **Time-to-First-Spike coding**
  - **Numerical Value coding**
  - **Poisson Distribution-based coding**

- **Task**: 
  - Assume the input stimuli last for `T` milliseconds. 
  - Apply the three coding methods and generate spikes, then plot the resulting spikes as raster plots.
  
- **Analysis**: 
  - Compare the results from the different coding methods.
  - Discuss the differences between each method.

### **Part Two:** Implementing Unsupervised Learning with STDP

- **Goal**: Implement the Spike-Timing-Dependent Plasticity (STDP) rule or a simplified version, Flat-STDP.

- **Network Structure**:
  - One input layer and one output layer with two excitatory neurons.
  - All neurons in the input layer are connected to all neurons in the output layer.

- **Task**:
  - Create two distinct activity patterns using Poisson distribution for input neuron activity (represented as blue and yellow).
  - Randomly activate one of the patterns as input and allow the output neurons to learn using the implemented learning rule.
  - Plot the synaptic weight changes for both output neurons during the learning process.

- **Analysis**:
  - After learning, create a cosine similarity plot between output and input neurons' weights.
  - Determine if the output neurons have learned the distinct patterns.
  - Analyze why or why not the neurons learned the patterns.

- **Further Experimentation**:
  - Introduce a silent neuron (no spikes) in both the input and output layers and analyze changes in synaptic weights.
  - Add homeostasis mechanisms and compare the weight changes with the previous setup.
  - Experiment with different parameter values, repeat the process, and analyze how parameter variations affect learning outcomes.

### **Part Three:** Implementing Reinforcement Learning with R-STDP

- **Goal**: Implement the Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP) rule or a simplified version, Flat-R-STDP.

- **Task**:
  - Repeat the same experiments as in the STDP activity but using the R-STDP learning rule.

- **Analysis**:
  - Compare the results obtained with R-STDP to those from STDP.
  - Discuss the differences in the learning process and how parameter variations impact results.
