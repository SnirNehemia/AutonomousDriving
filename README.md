# AutonomousDriving

This repository contains a deep reinforcement learning framework designed to train autonomous driving agents using [Gymnasium](https://gymnasium.farama.org/) and [HighwayEnv](https://github.com/eleurent/highway-env).

The project supports both single-run training and highly configurable ablation studies to test different hyperparameters, architectures, and algorithms.

## Implemented Algorithms & Concepts

Below is an overview of the key Reinforcement Learning (RL) concepts and algorithms implemented in this codebase, along with the mathematical formulas that power them.

### 1. The Reinforcement Learning Objective
In RL, an agent interacts with an environment over a sequence of discrete time steps. At each step $t$, the agent observes a state $S_t$, takes an action $A_t$, and receives a reward $R_t$. 

The goal of the agent is to maximize the expected **Discounted Return** ($G_t$), which is the sum of future rewards decayed by a discount factor $\gamma \in [0, 1)$:

$$ G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

The discount factor ensures that immediate rewards are prioritized over distant, uncertain rewards, and prevents infinite sums in continuous tasks.

---

### 2. REINFORCE (Vanilla Policy Gradient)
The `REINFORCEAgent` implements the Monte Carlo Policy Gradient algorithm. It parameterizes a policy $\pi_\theta(A_t | S_t)$ using a neural network and updates the network weights $\theta$ by taking gradient ascent on the expected return.

To compute the gradient, we use the **Policy Gradient Theorem**:

$$ \nabla_\theta J(\theta) = \mathbb{E} \left[ G_t \nabla_\theta \log \pi_\theta(A_t | S_t) \right] $$

Because PyTorch optimizers perform gradient *descent*, we minimize the negative loss:

$$ \mathcal{L}_{REINFORCE} = - \frac{1}{T} \sum_{t=1}^{T} G_t \log \pi_\theta(A_t | S_t) $$

**Key Characteristics:**
* **Monte Carlo:** It must wait until the end of a full episode to calculate $G_t$ before making an update.
* **High Variance:** Because $G_t$ is highly dependent on the exact sequence of actions taken during the episode, training can be noisy and unstable.

---

### 3. Advantage Actor-Critic (A2C)
The `A2CAgent` mitigates the high variance of REINFORCE by introducing a **Critic** network. Instead of waiting for the end of the episode to compute the exact return $G_t$, the Critic network estimates the Value of being in a state: $V_\phi(S_t)$.

We use this value to calculate the **Advantage** ($A_t$), which measures how much better an action was than the average expected action in that state:

$$ A_t = G_t - V_\phi(S_t) $$

The A2C algorithm relies on three distinct loss components combined into a single objective:

#### A. Actor Loss
The actor updates its policy in the direction of actions that yielded a positive advantage.

$$ \mathcal{L}_{Actor} = - \frac{1}{N} \sum_{t=1}^{N} A_t \log \pi_\theta(A_t | S_t) $$

#### B. Critic Loss
The critic updates its value predictions to be closer to the actual returns received using Mean Squared Error (MSE).

$$ \mathcal{L}_{Critic} = \frac{1}{N} \sum_{t=1}^{N} \left( V_\phi(S_t) - G_t \right)^2 $$

#### C. Total Combined Loss
The total loss optimized by the agent combines the Actor and Critic losses. (Note: Entropy is also subtracted, as explained in section 5).

$$ \mathcal{L}_{Total} = \mathcal{L}_{Actor} + c_{critic} \mathcal{L}_{Critic} $$
Where $c_{critic}$ is a coefficient (e.g., `0.5`) to balance the learning rates of the two networks.

---

### 4. Generalized Advantage Estimation (GAE)
Estimating the return $G_t$ using only a 1-step lookahead (TD-error) is low variance but highly biased by the Critic's current inaccuracy. Using the full Monte Carlo return is unbiased but highly variant. 

**GAE** balances bias and variance using a new hyperparameter $\lambda \in [0, 1]$. It computes an exponentially weighted average of $n$-step advantage estimators.

First, we define the 1-step Temporal Difference (TD) error $\delta_t$:

$$ \delta_t = R_t + \gamma V_\phi(S_{t+1}) - V_\phi(S_t) $$

The GAE advantage is then defined recursively as:

$$ \hat{A}_t^{GAE} = \delta_t + \gamma \lambda \hat{A}_{t+1}^{GAE} $$

Which expands to:

$$ \hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$

* If $\lambda = 1$: It reduces to the Monte Carlo advantage (high variance, low bias).
* If $\lambda = 0$: It reduces to the 1-step TD advantage (low variance, high bias).
* Typically, $\lambda = 0.95$ provides an optimal tradeoff.

When GAE is enabled (`use_gae=True`), the critic's target return for $\mathcal{L}_{Critic}$ is explicitly replaced with $\hat{A}_t^{GAE} + V_\phi(S_t)$.

---

### 5. Entropy Regularization
In continuous or highly repetitive tasks, RL agents often prematurely converge to a sub-optimal, deterministic policy (e.g., the car learns to just hit the gas and never switch lanes). 

To combat this, we add an **Entropy Bonus** to the loss function to explicitly encourage exploration. Entropy $H$ measures the unpredictability of the policy:

$$ H(\pi_\theta(\cdot | S_t)) = - \sum_{a} \pi_\theta(a | S_t) \log \pi_\theta(a | S_t) $$

By subtracting entropy from the total loss, the optimizer is forced to keep the action probabilities somewhat spread out unless it is absolutely certain.

$$ \mathcal{L}_{Total} = \mathcal{L}_{Actor} + c_{critic} \mathcal{L}_{Critic} - c_{entropy} H(\pi) $$

Where $c_{entropy}$ is the entropy coefficient (e.g., `0.01` to `0.05`). Over time, as the agent masters the environment, the actor loss overpowers the entropy penalty, and the agent converges.

---

## Running the Code

You can control the behavior of the framework using `config.yaml`.

* **Single Mode:** Trains a single agent and visualizes the results. `python train.py` (with `mode: single` in config)
* **Ablation Mode:** Trains multiple agents sequentially with differing hyperparameters (e.g., sweeping `entropy_coef` from `0.01` to `0.05`), then tests them against the exact same environment seeds to compare performance. `python train.py` (with `mode: ablation` in config)
* **Rendering:** Generates videos with real-time PyTorch network activation visualizations. `python render.py`
