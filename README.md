# Graph Neural Networks for Multi-Agent Reinforcement Learning in Supply Chain Inventory Control

[![Paper](https://img.shields.io/badge/Paper-Computers%20&%20Chemical%20Engineering-blue)](https://www.sciencedirect.com/science/article/pii/S0098135425001152)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Ray](https://img.shields.io/badge/Ray-RLlib-orange)](https://docs.ray.io/en/latest/rllib/)

This repository contains the implementation of a Multi-Agent Reinforcement Learning (MARL) framework with Graph Neural Networks (GNNs) for inventory control in supply chains, as presented in our paper published in *Computers & Chemical Engineering*.

## 📖 Abstract

Modern supply chains face increasing challenges from disruptive shocks, complex dynamics, uncertainties, and limited collaboration. Traditional inventory control methods with static parameters struggle to adapt to changing environments. This work proposes a MARL framework with GNNs for state representation that addresses these limitations by:

- **Redefining the action space** by parameterizing heuristic inventory control policies into adaptive, continuous forms
- **Leveraging graph structure** of supply chains to enable agents to learn system topology
- **Implementing centralized learning, decentralized execution** for collaborative learning while overcoming information-sharing constraints
- **Incorporating regularization techniques** to enhance performance in complex, decentralized environments

![Supply Chain MARL Framework](figures/gp-mappo2.png)
*Figure: Graph-enhanced Multi-Agent PPO framework for supply chain inventory control*

## 🔗 Paper

**[Leveraging graph neural networks and multi-agent reinforcement learning for inventory control in supply chains](https://www.sciencedirect.com/science/article/pii/S0098135425001152)**

*Niki Kotecha, Antonio del Rio Chanona*

Published in: Computers & Chemical Engineering, Volume 199, August 2025

## 🚀 Key Features

- **Multi-Agent PPO (MAPPO)** with centralized critic architecture
- **Graph Convolutional Networks** for supply chain topology learning
- **Parameterized action spaces** for continuous policy adaptation
- **Multiple supply chain configurations** (6, 12, 18, 24 agents)
- **Comprehensive evaluation framework** with training visualizations
- **Noise regularization** for improved robustness

## 🏗️ Repository Structure

```
├── env3rundiv.py           # Multi-agent supply chain environment
├── model.py                # GNN model implementations
├── ccmodel.py              # Centralized critic model
├── runMARL.py              # Main MARL training script
├── trainingfig.py          # Training visualization and analysis
├── execute.py              # Execution and evaluation scripts
├── data/                   # Supply chain configuration files
├── figures/                # Generated plots and visualizations
├── ray_results/            # Training results and checkpoints
└── Checkpoint/             # Model checkpoints
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/marl-gnn-supply-chain.git
   cd marl-gnn-supply-chain
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

- Ray[rllib] >= 2.0
- PyTorch >= 1.8
- PyTorch Geometric
- Gymnasium
- NumPy
- Matplotlib
- Pandas
- SciPy

## 🎯 Usage

### Training

Run the main training script for different supply chain configurations:

```bash
# Train on 6-agent supply chain
python runMARL.py --config data/g1_6.json

# Train with noise regularization
python runMARL.py --config data/g1_18.json --noise True
```

### Evaluation

Evaluate trained models:

```bash
python execute.py --checkpoint Checkpoint/your_model --config data/g1_18.json
```

### Visualization

Generate training curves and performance analysis:

```bash
python trainingfig.py
```

## 📊 Algorithms Implemented

- **IPPO**: Independent PPO agents
- **MAPPO**: Multi-Agent PPO with centralized critic
- **G-MAPPO**: Graph-enhanced MAPPO
- **P-GCN-MAPPO**: Parameterized Graph Convolutional Network MAPPO
- **Reg-P-GCN-MAPPO**: Regularized P-GCN-MAPPO with noise injection

## 🔬 Experimental Setup

The framework is evaluated on four supply chain configurations:
- **6 agents**: Simple linear supply chain
- **12 agents**: Medium complexity network
- **18 agents**: Complex multi-echelon structure  
- **24 agents**: Large-scale supply network

Each configuration tests the scalability and performance of the proposed approach under different network topologies and agent densities.

## 📈 Results

Our approach demonstrates:
- **Superior performance** compared to traditional MARL methods
- **Improved scalability** with increasing number of agents
- **Enhanced collaboration** through graph-based state representation
- **Robustness** to supply chain disruptions and uncertainties

![Training Results](figures/train_new_final2.png)
*Training performance comparison across different supply chain configurations*

![Performance Analysis](figures/bar_chart_error_bars.png)
*Computational efficiency analysis across different numbers of agents*

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kotecha2025leveraging,
  title={Leveraging graph neural networks and multi-agent reinforcement learning for inventory control in supply chains},
  author={Kotecha, Niki and del Rio Chanona, Antonio},
  journal={Computers \& Chemical Engineering},
  volume={199},
  pages={109111},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compchemeng.2025.109111}
}
```

## 👥 Authors

- **Niki Kotecha** - Imperial College London
- **Antonio del Rio Chanona** - Imperial College London

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

For questions about the research or implementation, please contact:
- Niki Kotecha: [nk3118@ic.ac.uk](mailto:nk3118@ic.ac.uk)

## 🙏 Acknowledgments

- Imperial College London for computational resources
- The Ray team for the excellent RLlib framework
- PyTorch Geometric community for GNN implementations

---

**Keywords:** Inventory Control, Supply Chain Optimization, Multi-Agent Reinforcement Learning, Graph Neural Networks, Decentralized Decision Making
