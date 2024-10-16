# Spiking Neural Networks: Information Coding and Learning Mechanisms

## Objectives
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
