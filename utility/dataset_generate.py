import json
import openai

# Replace with your GPT-3 API key
openai.api_key = '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'


def generate_code_using_gpt3(algorithm_name):
    # Create a prompt to ask GPT-3 to generate the code for the algorithm
    prompt = f"Generate Python code for the {algorithm_name} algorithm."

    # Call the GPT-3 API to generate the code
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200  # Adjust max_tokens based on the desired code length
    )

    # Extract the generated code from the API response
    generated_code = response.choices[0].text.strip()
    return generated_code


def generate_golfed_code_using_gpt3(original_code):
    # Create a prompt to ask GPT-3 to golf the code and reduce the number of lines
    prompt = f"Reduce the number of lines in the given Python code:\n\n{original_code}"

    # Call the GPT-3 API to generate the golfed version of the code
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100  # Adjust max_tokens based on the desired code length
    )

    # Extract the generated golfed code from the API response
    golfed_code = response.choices[0].text.strip()
    return golfed_code


# Algorithm names from the list

algorithms_list = [
    "Binary Search", "Linear Search", "Selection Sort", "Bubble Sort", "Insertion Sort",
    "Quick Sort", "Merge Sort", "Heap Sort", "Radix Sort", "Bucket Sort",
    "Counting Sort", "Topological Sort", "Breadth-First Search (BFS)", "Depth-First Search (DFS)",
    "Dijkstra's Algorithm", "Prim's Algorithm", "Kruskal's Algorithm", "Floyd-Warshall Algorithm", "Bellman-Ford Algorithm",
    "A* Search Algorithm", "Trie Data Structure", "Hashing (various techniques)", "Bloom Filter", "Dynamic Programming",
    "Knapsack Problem", "Longest Common Subsequence (LCS)", "Edit Distance", "Fibonacci Sequence (various algorithms)",
    "Matrix Chain Multiplication", "Coin Change Problem", "Shortest Supersequence", "Greedy Algorithms",
    "Fractional Knapsack Problem", "Huffman Coding", "Activity Selection", "Kruskal's Minimum Spanning Tree Algorithm",
    "Max Flow - Ford Fulkerson Algorithm", "Longest Increasing Subsequence", "Graph Coloring", "Johnson's Algorithm",
    "Rabin-Karp Algorithm", "Z-Algorithm", "Boyer-Moore Algorithm", "KMP Algorithm", "Manacher's Algorithm",
    "Suffix Array", "Suffix Tree", "Tarjan's Strongly Connected Components Algorithm", "Minimum Spanning Tree (MST)",
    "Articulation Points and Bridges", "Eulerian Path and Circuit", "Chinese Postman Problem", "Minimum Cut in Graphs",
    "Hopcroft-Karp Algorithm", "Bipartite Matching", "Ford-Fulkerson Algorithm", "Edmonds-Karp Algorithm", "Hungarian Algorithm",
    "Viterbi Algorithm", "Expectation-Maximization Algorithm", "Principal Component Analysis (PCA)", "K-nearest neighbors (KNN)",
    "Support Vector Machines (SVM)", "Random Forest", "K-means Clustering", "Hierarchical Clustering", "DBSCAN Algorithm",
    "Apriori Algorithm", "Expectation Maximization (EM) Algorithm", "PageRank Algorithm", "Simulated Annealing",
    "Ant Colony Optimization", "Particle Swarm Optimization", "Genetic Algorithm", "Tabu Search", "Bat Algorithm",
    "Firefly Algorithm", "Cuckoo Search", "Grey Wolf Optimizer", "Harmony Search Algorithm", "Artificial Bee Colony Algorithm",
    "Bacterial Foraging Optimization Algorithm", "Biogeography-Based Optimization", "Differential Evolution Algorithm",
    "CMA-ES (Covariance Matrix Adaptation Evolution Strategy)", "PSO (Particle Swarm Optimization)", "Cat Swarm Optimization",
    "Spiral Optimization Algorithm", "Hill Climbing", "Tabu Search", "Local Search Algorithms", "Scatter Search",
    "Variable Neighborhood Search", "Large Neighborhood Search", "Dynamic Time Warping", "Levenshtein Distance",
    "Needleman-Wunsch Algorithm", "Smith-Waterman Algorithm", "Hirschberg's Algorithm", "Bitonic Sorting Network",
    "Booth's Algorithm", "Cooley-Tukey FFT Algorithm", "Miller-Rabin Primality Test", "RSA Cryptosystem",
    "ElGamal Encryption", "Diffie-Hellman Key Exchange", "Merkle-Hellman Knapsack Cryptosystem", "Feistel Cipher",
    "Blowfish Cipher", "Rijndael (AES) Cipher", "RC4 Algorithm", "Shamir's Secret Sharing", "Verifiable Random Function",
    "BLS Signature Scheme", "Schnorr Signature", "Paillier Cryptosystem", "RSA Digital Signature", "Elliptic Curve Cryptography",
    "Fisher-Yates Shuffle", "Birthday Paradox", "Monte Carlo Method", "Las Vegas Algorithm", "Simulated Trading",
    "Cross-Entropy Method", "Tabular Q-Learning", "Deep Q-Network (DQN)", "A3C (Asynchronous Advantage Actor-Critic)",
    "PPO (Proximal Policy Optimization)", "Dueling DQN", "TD-learning", "SARSA (State-Action-Reward-State-Action)",
    "Temporal Difference (TD) Learning", "Backpropagation", "Reservoir Sampling", "Skip Lists", "Locality-Sensitive Hashing",
    "R-tree", "B*-tree", "AVL tree", "Red-Black tree", "Splay tree", "KD-tree", "Suffix Automaton", "Van Emde Boas tree",
    "Bloomier Filter", "Skip Graphs", "Lossy Counting", "HyperLogLog", "Quicksort Variants (3-way, Dual-pivot, etc.)",
    "Strassen's Algorithm", "Karatsuba Algorithm", "Coppersmith-Winograd Algorithm", "Floyd's Cycle Detection Algorithm",
    "Moser's Circle Finding Algorithm", "Erdős–Gallai Theorem", "Ruzsa–Szemerédi Graph Partitioning", "Spelling Correction Algorithms",
    "Wheel Factorization", "AKS Primality Test", "Miller's Algorithm", "Pohlig-Hellman Algorithm", "Baby-step Giant-step Algorithm",
    "Pollard's Rho Algorithm", "Schoof's Algorithm", "Silver-Pohlig-Hellman Algorithm", "Coppersmith's Attack",
    "Fermat's Factorization Method", "Index Calculus Algorithm", "Big Number Factorization Algorithms",
    "Pollard's p − 1 Algorithm", "Pell's Equation Solving Algorithm", "Continued Fraction Factorization Method",
    "Adleman-Pomerance-Rumely Method", "Dixon's Factorization Method", "Elliptic Curve Factorization", "Rho Pollard Integer Factorization",
    "Sublinear Algorithms", "Streaming Algorithms", "Online Algorithms", "Approximation Algorithms", "Randomized Algorithms",
    "Parallel Algorithms", "Distributed Algorithms", "Quantum Algorithms", "Metaheuristic Algorithms", "Evolutionary Algorithms",
    "Nature-Inspired Algorithms", "Multi-objective Optimization Algorithms", "Pareto Optimization Algorithms",
    "Constraint Satisfaction Algorithms", "Automated Theorem Proving", "Proof Complexity Algorithms",
    "Decision Tree Learning Algorithms", "Ensemble Learning Algorithms", "Transfer Learning Algorithms", "Reinforcement Learning Algorithms",
    "Batch Reinforcement Learning", "Model-based Reinforcement Learning", "Model-free Reinforcement Learning", "Value Iteration",
    "Policy Iteration", "Temporal-Difference Methods", "Actor-Critic Methods", "Deep Reinforcement Learning", "Inverse Reinforcement Learning",
    "Multi-Armed Bandit Algorithms", "Regret Minimization Algorithms", "Online Convex Optimization", "Meta-learning Algorithms",
    "Adversarial Learning Algorithms", "Generative Adversarial Networks (GANs)", "CycleGAN", "Pix2Pix", "StyleGAN", "DCGAN",
    "Variational Autoencoders (VAEs)", "Restricted Boltzmann Machines (RBMs)", "Hopfield Networks", "Boltzmann Machines",
    "Kohonen Networks", "Liquid State Machines", "Echo State Networks", "Neuroevolution", "Neural Turing Machines",
    "Memory Networks", "Attention Mechanisms", "Transformer Networks", "Capsule Networks", "Self-Organizing Maps (SOMs)",
    "Growing Neural Gas (GNG)", "NeuroFuzzy Systems", "Radial Basis Function Networks (RBFNs)", "Fuzzy Logic Systems",
    "Expert Systems", "Bayesian Networks", "Markov Decision Processes (MDPs)", "Q-learning", "Monte Carlo Tree Search (MCTS)",
    "Temporal-Difference Learning", "Upper Confidence Bound (UCB)", "Thompson Sampling", "Bootstrap Aggregating (Bagging)",
    "Boosting Algorithms (AdaBoost, Gradient Boosting)", "Random Forests", "Extreme Gradient Boosting (XGBoost)",
    "LightGBM", "CatBoost", "Stochastic Gradient Descent (SGD)", "Mini-batch Gradient Descent", "AdaGrad",
    "RMSprop", "Adam Optimizer", "L-BFGS", "LBFGS-B", "Limited-memory BFGS (L-BFGS)", "Conjugate Gradient Descent",
    "Coordinate Descent", "Nelder-Mead Algorithm", "Trust Region Optimization", "Gauss-Newton Algorithm", "Levenberg-Marquardt Algorithm",
    "Non-linear Conjugate Gradient Method", "Powell's Method", "Golden Section Search", "Simulated Annealing",
    "Genetic Algorithms for Optimization", "Ant Colony Optimization", "Particle Swarm Optimization",
    "Differential Evolution Optimization", "Harmony Search Optimization", "Firefly Algorithm", "Cuckoo Search Optimization",
    "Bat Algorithm", "Artificial Bee Colony Optimization", "Bacterial Foraging Optimization", "Grey Wolf Optimizer",
    "Biogeography-Based Optimization", "Symbiotic Organisms Search", "Krill Herd Algorithm", "Hill Climbing",
    "Tabu Search", "Local Search Algorithms", "Stochastic Hill Climbing", "Random Restart Hill Climbing",
    "First-choice Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "Evolutionary Strategies",
    "Evolutionary Programming", "Genetic Programming", "Cultural Algorithms", "Co-evolutionary Algorithms",
    "Multi-objective Evolutionary Algorithms", "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)",
    "NSGA-II (Non-dominated Sorting Genetic Algorithm II)", "SPEA2 (Strength Pareto Evolutionary Algorithm 2)",
    "IBEA (Indicator-Based Evolutionary Algorithm)", "GDE3 (Generalized Differential Evolution 3)", "MOEA/D-ACO",
    "MOEA/D-SO", "Pareto Archived Evolution Strategy (PAES)", "Multi-objective Particle Swarm Optimization",
    "NSPSO (Non-dominated Sorting Particle Swarm Optimization)", "MOPSO (Multi-Objective Particle Swarm Optimization)",
    "PESA-II (Pareto Envelope-based Selection Algorithm II)", "RVEA (Reference Vector Guided Evolutionary Algorithm)",
    "VEGA (Vector Evaluated Genetic Algorithm)", "NSGA-III (Non-dominated Sorting Genetic Algorithm III)",
    "SMPSO (Speed-constrained Multi-objective Particle Swarm Optimization)",
    "MOEA/D-GWO (Grey Wolf Optimizer-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-CMA (Covariance Matrix Adaptation Evolutionary Algorithm)",
    "MOEA/D-DE (Differential Evolution-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-PSO (Particle Swarm Optimization-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-SA (Simulated Annealing-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-ABC (Artificial Bee Colony-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-FS (Firefly Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-CS (Cuckoo Search-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HS (Harmony Search-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-BFO (Bacterial Foraging Optimization-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-BBO (Biogeography-Based Optimization-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-SOS (Symbiotic Organisms Search-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-KHA (Krill Herd Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-MSP (Multi-Strategy Pareto-based Algorithm)",
    "MOEA/D-HHGA (Hybrid Harmony Search and Genetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-MAEA (Multi-Algorithm Evolutionary Algorithm)",
    "MOEA/D-HHSA (Hybrid Harmony Search and Simulated Annealing-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-MS (Multi-Strategy-based Algorithm)",
    "MOEA/D-HHH (Hybrid Harmony Search Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HSGA (Hybrid Harmony Search and Genetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HSDE (Hybrid Harmony Search and Differential Evolution Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HBBO (Hybrid Harmony Search and Biogeography-Based Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-MAM (Memetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HMAB (Hybrid Metaheuristic Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HMABC (Hybrid Metaheuristic Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HPSO (Hybrid Particle Swarm Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HBFO (Hybrid Bacterial Foraging Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HCS (Hybrid Cuckoo Search Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HSA (Hybrid Simulated Annealing Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HCA (Hybrid Cultural Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HCSA (Hybrid Co-evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HDA (Hybrid Differential Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HNFEA (Hybrid Neuro-Fuzzy Evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HDGEA (Hybrid Differential Grouping Evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HDDG (Hybrid Differential Dynamics Group-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HSDA (Hybrid Self-Adaptive Differential Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HIS (Hybrid Immune System Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HPSA (Hybrid Particle Swarm Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HCOA (Hybrid Cultural Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
    "MOEA/D-HMOA (Hybrid Metaheuristic Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)"]

algorithms_list = list(set(algorithms_list))

# Create an empty list to store the dataset
dataset = []

# Loop through each algorithm and generate code and golfed version
for algorithm_name in algorithms_list:
    # Generate code for the algorithm using GPT-3
    algorithm_code = generate_code_using_gpt3(algorithm_name)

    # Generate a golfed version of the code using GPT-3
    golfed_code = generate_golfed_code_using_gpt3(algorithm_code)

    # Create a dataset entry
    dataset_entry = {
        'prompt': algorithm_code, #.strip(),
        'completion': golfed_code  #.strip()
    }
    print(algorithm_name, "completed")
    # Append the dataset entry to the list
    dataset.append(dataset_entry)

# Save the dataset to a f               ile
with open('golf_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print("Dataset generation completed and saved to 'code_golf_dataset.json'.")






#
# import asyncio
# import json
# from openai import AsyncOpenAI
#
# # Replace with your GPT-3 API key
# api_key = "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"
#
#
# async def generate_code_using_gpt3(algorithm_name, client):
#     prompt = f"Generate Python code for the {algorithm_name} algorithm."
#     response = await client.completions.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=200
#     )
#     return response.choices[0].text.strip()
#
#
# async def generate_golfed_code_using_gpt3(original_code, client):
#     prompt = f"Reduce the number of lines in the given Python code:\n\n{original_code}"
#     response = await client.completions.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=100
#     )
#     return response.choices[0].text.strip()
#
#
# async def generate_and_save_dataset(api_key):
#     client = AsyncOpenAI(api_key=api_key)
#     algorithms_list = [
#         "Binary Search", "Linear Search", "Selection Sort", "Bubble Sort", "Insertion Sort",
#         "Quick Sort", "Merge Sort", "Heap Sort", "Radix Sort", "Bucket Sort",
#         "Counting Sort", "Topological Sort", "Breadth-First Search (BFS)", "Depth-First Search (DFS)",
#         "Dijkstra's Algorithm", "Prim's Algorithm", "Kruskal's Algorithm", "Floyd-Warshall Algorithm",
#         "Bellman-Ford Algorithm",
#         "A* Search Algorithm", "Trie Data Structure", "Hashing (various techniques)", "Bloom Filter",
#         "Dynamic Programming",
#         "Knapsack Problem", "Longest Common Subsequence (LCS)", "Edit Distance",
#         "Fibonacci Sequence (various algorithms)",
#         "Matrix Chain Multiplication", "Coin Change Problem", "Shortest Supersequence", "Greedy Algorithms",
#         "Fractional Knapsack Problem", "Huffman Coding", "Activity Selection",
#         "Kruskal's Minimum Spanning Tree Algorithm",
#         "Max Flow - Ford Fulkerson Algorithm", "Longest Increasing Subsequence", "Graph Coloring",
#         "Johnson's Algorithm",
#         "Rabin-Karp Algorithm", "Z-Algorithm", "Boyer-Moore Algorithm", "KMP Algorithm", "Manacher's Algorithm",
#         "Suffix Array", "Suffix Tree", "Tarjan's Strongly Connected Components Algorithm",
#         "Minimum Spanning Tree (MST)",
#         "Articulation Points and Bridges", "Eulerian Path and Circuit", "Chinese Postman Problem",
#         "Minimum Cut in Graphs",
#         "Hopcroft-Karp Algorithm", "Bipartite Matching", "Ford-Fulkerson Algorithm", "Edmonds-Karp Algorithm",
#         "Hungarian Algorithm",
#         "Viterbi Algorithm", "Expectation-Maximization Algorithm", "Principal Component Analysis (PCA)",
#         "K-nearest neighbors (KNN)",
#         "Support Vector Machines (SVM)", "Random Forest", "K-means Clustering", "Hierarchical Clustering",
#         "DBSCAN Algorithm",
#         "Apriori Algorithm", "Expectation Maximization (EM) Algorithm", "PageRank Algorithm", "Simulated Annealing",
#         "Ant Colony Optimization", "Particle Swarm Optimization", "Genetic Algorithm", "Tabu Search", "Bat Algorithm",
#         "Firefly Algorithm", "Cuckoo Search", "Grey Wolf Optimizer", "Harmony Search Algorithm",
#         "Artificial Bee Colony Algorithm",
#         "Bacterial Foraging Optimization Algorithm", "Biogeography-Based Optimization",
#         "Differential Evolution Algorithm",
#         "CMA-ES (Covariance Matrix Adaptation Evolution Strategy)", "PSO (Particle Swarm Optimization)",
#         "Cat Swarm Optimization",
#         "Spiral Optimization Algorithm", "Hill Climbing", "Tabu Search", "Local Search Algorithms", "Scatter Search",
#         "Variable Neighborhood Search", "Large Neighborhood Search", "Dynamic Time Warping", "Levenshtein Distance",
#         "Needleman-Wunsch Algorithm", "Smith-Waterman Algorithm", "Hirschberg's Algorithm", "Bitonic Sorting Network",
#         "Booth's Algorithm", "Cooley-Tukey FFT Algorithm", "Miller-Rabin Primality Test", "RSA Cryptosystem",
#         "ElGamal Encryption", "Diffie-Hellman Key Exchange", "Merkle-Hellman Knapsack Cryptosystem", "Feistel Cipher",
#         "Blowfish Cipher", "Rijndael (AES) Cipher", "RC4 Algorithm", "Shamir's Secret Sharing",
#         "Verifiable Random Function",
#         "BLS Signature Scheme", "Schnorr Signature", "Paillier Cryptosystem", "RSA Digital Signature",
#         "Elliptic Curve Cryptography",
#         "Fisher-Yates Shuffle", "Birthday Paradox", "Monte Carlo Method", "Las Vegas Algorithm", "Simulated Trading",
#         "Cross-Entropy Method", "Tabular Q-Learning", "Deep Q-Network (DQN)",
#         "A3C (Asynchronous Advantage Actor-Critic)",
#         "PPO (Proximal Policy Optimization)", "Dueling DQN", "TD-learning", "SARSA (State-Action-Reward-State-Action)",
#         "Temporal Difference (TD) Learning", "Backpropagation", "Reservoir Sampling", "Skip Lists",
#         "Locality-Sensitive Hashing",
#         "R-tree", "B*-tree", "AVL tree", "Red-Black tree", "Splay tree", "KD-tree", "Suffix Automaton",
#         "Van Emde Boas tree",
#         "Bloomier Filter", "Skip Graphs", "Lossy Counting", "HyperLogLog",
#         "Quicksort Variants (3-way, Dual-pivot, etc.)",
#         "Strassen's Algorithm", "Karatsuba Algorithm", "Coppersmith-Winograd Algorithm",
#         "Floyd's Cycle Detection Algorithm",
#         "Moser's Circle Finding Algorithm", "Erdős–Gallai Theorem", "Ruzsa–Szemerédi Graph Partitioning",
#         "Spelling Correction Algorithms",
#         "Wheel Factorization", "AKS Primality Test", "Miller's Algorithm", "Pohlig-Hellman Algorithm",
#         "Baby-step Giant-step Algorithm",
#         "Pollard's Rho Algorithm", "Schoof's Algorithm", "Silver-Pohlig-Hellman Algorithm", "Coppersmith's Attack",
#         "Fermat's Factorization Method", "Index Calculus Algorithm", "Big Number Factorization Algorithms",
#         "Pollard's p − 1 Algorithm", "Pell's Equation Solving Algorithm", "Continued Fraction Factorization Method",
#         "Adleman-Pomerance-Rumely Method", "Dixon's Factorization Method", "Elliptic Curve Factorization",
#         "Rho Pollard Integer Factorization",
#         "Sublinear Algorithms", "Streaming Algorithms", "Online Algorithms", "Approximation Algorithms",
#         "Randomized Algorithms",
#         "Parallel Algorithms", "Distributed Algorithms", "Quantum Algorithms", "Metaheuristic Algorithms",
#         "Evolutionary Algorithms",
#         "Nature-Inspired Algorithms", "Multi-objective Optimization Algorithms", "Pareto Optimization Algorithms",
#         "Constraint Satisfaction Algorithms", "Automated Theorem Proving", "Proof Complexity Algorithms",
#         "Decision Tree Learning Algorithms", "Ensemble Learning Algorithms", "Transfer Learning Algorithms",
#         "Reinforcement Learning Algorithms",
#         "Batch Reinforcement Learning", "Model-based Reinforcement Learning", "Model-free Reinforcement Learning",
#         "Value Iteration",
#         "Policy Iteration", "Temporal-Difference Methods", "Actor-Critic Methods", "Deep Reinforcement Learning",
#         "Inverse Reinforcement Learning",
#         "Multi-Armed Bandit Algorithms", "Regret Minimization Algorithms", "Online Convex Optimization",
#         "Meta-learning Algorithms",
#         "Adversarial Learning Algorithms", "Generative Adversarial Networks (GANs)", "CycleGAN", "Pix2Pix", "StyleGAN",
#         "DCGAN",
#         "Variational Autoencoders (VAEs)", "Restricted Boltzmann Machines (RBMs)", "Hopfield Networks",
#         "Boltzmann Machines",
#         "Kohonen Networks", "Liquid State Machines", "Echo State Networks", "Neuroevolution", "Neural Turing Machines",
#         "Memory Networks", "Attention Mechanisms", "Transformer Networks", "Capsule Networks",
#         "Self-Organizing Maps (SOMs)",
#         "Growing Neural Gas (GNG)", "NeuroFuzzy Systems", "Radial Basis Function Networks (RBFNs)",
#         "Fuzzy Logic Systems",
#         "Expert Systems", "Bayesian Networks", "Markov Decision Processes (MDPs)", "Q-learning",
#         "Monte Carlo Tree Search (MCTS)",
#         "Temporal-Difference Learning", "Upper Confidence Bound (UCB)", "Thompson Sampling",
#         "Bootstrap Aggregating (Bagging)",
#         "Boosting Algorithms (AdaBoost, Gradient Boosting)", "Random Forests", "Extreme Gradient Boosting (XGBoost)",
#         "LightGBM", "CatBoost", "Stochastic Gradient Descent (SGD)", "Mini-batch Gradient Descent", "AdaGrad",
#         "RMSprop", "Adam Optimizer", "L-BFGS", "LBFGS-B", "Limited-memory BFGS (L-BFGS)", "Conjugate Gradient Descent",
#         "Coordinate Descent", "Nelder-Mead Algorithm", "Trust Region Optimization", "Gauss-Newton Algorithm",
#         "Levenberg-Marquardt Algorithm",
#         "Non-linear Conjugate Gradient Method", "Powell's Method", "Golden Section Search", "Simulated Annealing",
#         "Genetic Algorithms for Optimization", "Ant Colony Optimization", "Particle Swarm Optimization",
#         "Differential Evolution Optimization", "Harmony Search Optimization", "Firefly Algorithm",
#         "Cuckoo Search Optimization",
#         "Bat Algorithm", "Artificial Bee Colony Optimization", "Bacterial Foraging Optimization", "Grey Wolf Optimizer",
#         "Biogeography-Based Optimization", "Symbiotic Organisms Search", "Krill Herd Algorithm", "Hill Climbing",
#         "Tabu Search", "Local Search Algorithms", "Stochastic Hill Climbing", "Random Restart Hill Climbing",
#         "First-choice Hill Climbing", "Simulated Annealing", "Genetic Algorithm", "Evolutionary Strategies",
#         "Evolutionary Programming", "Genetic Programming", "Cultural Algorithms", "Co-evolutionary Algorithms",
#         "Multi-objective Evolutionary Algorithms",
#         "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)",
#         "NSGA-II (Non-dominated Sorting Genetic Algorithm II)", "SPEA2 (Strength Pareto Evolutionary Algorithm 2)",
#         "IBEA (Indicator-Based Evolutionary Algorithm)", "GDE3 (Generalized Differential Evolution 3)", "MOEA/D-ACO",
#         "MOEA/D-SO", "Pareto Archived Evolution Strategy (PAES)", "Multi-objective Particle Swarm Optimization",
#         "NSPSO (Non-dominated Sorting Particle Swarm Optimization)",
#         "MOPSO (Multi-Objective Particle Swarm Optimization)",
#         "PESA-II (Pareto Envelope-based Selection Algorithm II)",
#         "RVEA (Reference Vector Guided Evolutionary Algorithm)",
#         "VEGA (Vector Evaluated Genetic Algorithm)", "NSGA-III (Non-dominated Sorting Genetic Algorithm III)",
#         "SMPSO (Speed-constrained Multi-objective Particle Swarm Optimization)",
#         "MOEA/D-GWO (Grey Wolf Optimizer-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-CMA (Covariance Matrix Adaptation Evolutionary Algorithm)",
#         "MOEA/D-DE (Differential Evolution-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-PSO (Particle Swarm Optimization-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-SA (Simulated Annealing-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-ABC (Artificial Bee Colony-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-FS (Firefly Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-CS (Cuckoo Search-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HS (Harmony Search-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-BFO (Bacterial Foraging Optimization-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-BBO (Biogeography-Based Optimization-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-SOS (Symbiotic Organisms Search-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-KHA (Krill Herd Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-MSP (Multi-Strategy Pareto-based Algorithm)",
#         "MOEA/D-HHGA (Hybrid Harmony Search and Genetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-MAEA (Multi-Algorithm Evolutionary Algorithm)",
#         "MOEA/D-HHSA (Hybrid Harmony Search and Simulated Annealing-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-MS (Multi-Strategy-based Algorithm)",
#         "MOEA/D-HHH (Hybrid Harmony Search Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HSGA (Hybrid Harmony Search and Genetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HSDE (Hybrid Harmony Search and Differential Evolution Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HBBO (Hybrid Harmony Search and Biogeography-Based Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-MAM (Memetic Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HMAB (Hybrid Metaheuristic Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HMABC (Hybrid Metaheuristic Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HPSO (Hybrid Particle Swarm Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HBFO (Hybrid Bacterial Foraging Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HCS (Hybrid Cuckoo Search Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HSA (Hybrid Simulated Annealing Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HCA (Hybrid Cultural Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HCSA (Hybrid Co-evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HDA (Hybrid Differential Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HNFEA (Hybrid Neuro-Fuzzy Evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HDGEA (Hybrid Differential Grouping Evolutionary Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HDDG (Hybrid Differential Dynamics Group-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HSDA (Hybrid Self-Adaptive Differential Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HIS (Hybrid Immune System Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HPSA (Hybrid Particle Swarm Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HCOA (Hybrid Cultural Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)",
#         "MOEA/D-HMOA (Hybrid Metaheuristic Optimization Algorithm-based Multi-Objective Evolutionary Algorithm)"]
#
#     algorithms_list = list(set(algorithms_list))
#     dataset = []
#
#     for algorithm_name in algorithms_list:
#         algorithm_code = await generate_code_using_gpt3(algorithm_name, client)
#         golfed_code = await generate_golfed_code_using_gpt3(algorithm_code, client)
#
#         dataset_entry = {
#             'prompt': algorithm_code.strip(),
#             'completion': golfed_code.strip()
#         }
#         dataset.append(dataset_entry)
#         print(algorithm_name, "completed")
#
#     # Save the dataset to a file
#     with open('golf_dataset.json', 'w') as f:
#         json.dump(dataset, f, indent=4)
#
#     print("Dataset generation completed and saved to 'code_golf_dataset.json'.")
#
#
# # Run the asynchronous code with provided API key
# asyncio.run(generate_and_save_dataset(api_key))
