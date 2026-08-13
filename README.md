# Pirate Intelligent Agent: Deep Q-Learning for Pathfinding

**Eric Shadwick** | CS 370: Current and Emerging Trends in Computer Science | Southern New Hampshire University

A deep Q-learning agent that learns to navigate an 8x8 treasure hunt maze. The agent plays a pirate NPC whose goal is to reach the treasure before the human player, which makes this a pathfinding problem solved through reinforcement learning rather than through a classical search algorithm.

**Result:** The agent reached a 100% win rate at epoch 422 of a possible 999 in 8.04 minutes, and passed an exhaustive check confirming it wins from every reachable cell on the board.

---

## The work I did on this project

### What I was given

The environment and supporting infrastructure were provided as course starter code:

- **`TreasureMaze.py`** defines the maze as an 8x8 matrix and implements the environment: the reset behavior, the legal action set, the state representation (the full board flattened into 64 floating point values, with blocked cells at 0.0, free cells at 1.0, and the pirate's position marked at 0.5), the reward function (-0.04 per legal move, -0.25 for revisiting a cell, -0.75 for attempting a blocked move, +1.0 at the treasure, and a floor that terminates hopeless games), and the game status logic.
- **`GameExperience.py`** implements the experience replay buffer, including the target network handling and the batch sampling used for training.
- **Notebook scaffolding**, including `build_model()`, `play_game()`, `completion_check()`, `show()`, `format_time()`, and the `qtrain()` function signature with its setup, epsilon schedule, and progress reporting.

### What I wrote

**The Q-training loop inside `qtrain()`.** This is the core of the assignment and the part that makes the agent learn. Each epoch resets the environment at a randomly chosen free cell, then repeats a cycle of:

1. Capturing the current environment state
2. Selecting an action, either randomly according to the exploration rate or by taking the highest predicted value from the network
3. Executing that action and observing the resulting state, reward, and game status
4. Storing the transition in the experience buffer
5. Sampling a batch from stored experience and training the network on it
6. Synchronizing the target network on its scheduled interval
7. Tracking wins and losses and breaking out when the game ends

I also made three smaller changes:

- Added `tf.config.set_visible_devices([], 'GPU')` after the TensorFlow import. The Codio virtual lab environment raised `InternalError: cudaSetDevice() on GPU:0 failed` on the first Adam optimizer construction, and forcing CPU-only execution resolved it.
- Modified the two verification cells to capture and print the return values of `completion_check()` and `play_game()`. As provided, both cells discarded those booleans because `show(qmaze)` was the last expression, so the notebook rendered maze images without ever stating whether the tests passed.
- Added markdown cells interpreting the training and testing results.

### How I know it works

Training output alone is not proof. Three signals together are:

| Signal | Start of training | End of training |
|---|---|---|
| Win rate (rolling, last 32 games) | 0.000 | 1.000 |
| Average episode length | ~127 moves | ~20 moves |
| Exhaustive completion check | Fails | Passes |

The falling episode length is the strongest evidence, because a wandering agent that stumbles onto the treasure still records a win, while an agent finishing in close to the minimum number of moves has learned which direction carries value from any position.

The most instructive detail is the gap between the second and third rows. The win rate first reached 1.000 at epoch 267, but training continued for 155 more epochs because `completion_check` kept returning False. Of the 156 epochs from 267 onward, 124 showed a perfect win rate while the agent still could not win from every free cell on the board. A metric that looks solved and an exhaustive check that it is solved are different claims, and this run demonstrated the distance between them.

Worth noting what is *not* evidence here: training loss did not decline, averaging 0.0018 across the first twenty epochs and 0.0022 across the last twenty. That is expected rather than a defect. In deep Q-learning the training targets are generated from the network's own bootstrapped value estimates, so the target distribution shifts as the policy improves. A flat loss alongside a rising win rate means the network is tracking a moving target, not that learning has stalled.

---

## Connecting this to the field

### What do computer scientists do, and why does it matter?

Computer scientists translate problems that people care about into forms a machine can actually operate on, and then judge honestly whether the result is trustworthy. The translation step is where most of the work lives. In this project the algorithm was general purpose. Nothing in the training loop knows anything about mazes. What made the pirate agent solve *this* maze was the reward function: a step cost that makes short paths preferable, a revisit penalty that stops the agent from pacing, and a floor that terminates games that cannot teach anything. Remove any one of those and the agent still runs, but it learns the wrong behavior.

That matters because the same pattern scales up to decisions with real consequences. An 8x8 grid is simple enough to specify exhaustively, so a correct reward function is achievable. Clinical objectives are not like that. A system optimizing for length of stay or a lab value while omitting patient burden will optimize that proxy faithfully and look correct on every metric derived from it. The work of specifying the objective, not the work of choosing the architecture, is where the outcome is actually determined.

### How do I approach a problem as a computer scientist?

Read before reasoning, and verify before trusting. I read the two provided Python files in full before writing a line of the training loop, specifically to get the real method signatures rather than assuming a version of this assignment I had seen elsewhere. That turned out to matter: this starter passes a target model into `GameExperience` and adds a target update frequency, so the training loop has to maintain the target network itself, which an assumed implementation would have missed.

The same habit caught problems throughout the course. In Module 3, a `Conv2D` layer using deprecated positional arguments silently built the wrong architecture, visible only by comparing parameter counts before and after. Also in Module 3, a Keras data generator was quietly exhausting mid-epoch and reporting stale metrics for roughly half of the specified training epochs. In Module 6, a starter notebook was missing an `import sys` while four exception handlers called `sys.exit(1)`. None of those failures announced themselves. Each one was found by running the code and inspecting the actual output rather than reading it and assuming it did what it appeared to do.

The corollary is knowing what a given piece of evidence can and cannot support. A bracketed execution number in Jupyter does not mean a cell succeeded, because failed cells get numbers too. A 100% win rate does not mean a solved maze. Being specific about what a measurement actually proves is most of the discipline.

### What are my ethical responsibilities to the end user and the organization?

The responsibility I take most seriously is not overstating what a system can do. Machine learning systems are opaque by construction: a network cannot explain which of its weights produced an output, which means the people relying on it depend entirely on the honesty of whoever reports its performance. Selecting the flattering metric is not lying, and it is exactly how systems get deployed past their competence.

The consequences are documented rather than hypothetical. Buolamwini and Gebru's audit of commercial facial analysis systems found error rates roughly forty times higher for darker-skinned women than for lighter-skinned men, a gap invisible in aggregate accuracy. NIST found the same pattern across 189 algorithms. Obermeyer and colleagues showed a healthcare risk-prediction algorithm affecting millions of patients was systematically underestimating illness in Black patients, because it used historical cost as a stand-in for need, and cost reflects access rather than sickness. In each case the code worked as written. The specification was wrong, and no one caught it because the headline number looked fine.

I work in healthcare technology and intend to build clinical machine learning systems, so this is not abstract to me. My obligations are to state limitations as clearly as capabilities, to test for the failure modes that averages conceal rather than only the ones a demo would surface, to treat patient data as something entrusted rather than owned, and to keep a human in the loop for decisions a model has not earned the right to make alone. To the organization, the same honesty is the actual service: a system that quietly fails on a subgroup is a liability, and the person best positioned to find that before a patient does is the engineer who built it.

---

## Environment

Python 3.11.9, TensorFlow, Keras, NumPy, Matplotlib. Developed and run in the SNHU Codio virtual lab.

The notebook requires `TreasureMaze.py` and `GameExperience.py`, which are SNHU-provided course files and are not redistributed in this repository.

---

## AI Use Acknowledgment

I used Anthropic's Claude to assist with this project. Specifically, it helped me review the provided starter code and environment files, implement and debug the Q-training loop, diagnose the Codio GPU initialization error, and organize this README. All training runs were executed and verified by me, and the analysis and reflections here are my own.

Anthropic. (2026, August 2). *CS370 Project* [Generative AI chat]. Claude. https://claude.ai/share/95d98f29-91e7-4cac-8d83-6d6ac0312366

Anthropic. (2026, August 13). *GitHub Help* [Generative AI chat]. Claude. [<SHARE LINK FOR THIS CHAT>](https://claude.ai/share/23544750-4dcf-4ca1-a6bf-d26287a9cef8)
