<div align="center">
  <h1>🧠 Dynamic Optimization with VFI</h1>
</div>

This repository contains a Python implementation for solving dynamic optimization problems using the Value Function Iteration (VFI) method. It includes both deterministic and stochastic versions of the classic cake-eating problem.

---

## ✨ Features

✅ **Generic Solver**: A `ValueFunctionIterator` class that can solve any model inheriting from `BellmanModel`.
✅ **Vectorized Operations**: High-performance NumPy-based calculations that avoid slow Python loops.
✅ **Stochastic & Deterministic Models**: Includes implementations for both certain and uncertain environments.
✅ **Clear Structure**: The code is organized into logical modules for models, solver, and presentation.
✅ **Progress Tracking**: Uses `tqdm` to show real-time progress during model solving.

---

## 📂 Repository Structure

The project is organized as follows:

```
├── presentation/      # 📊 Main scripts to run models and generate plots
├── src/
│   ├── models/        # 🧠 Dynamic programming model definitions
│   │   ├── bellman.py
│   │   └── stochastic.py
│   ├── solver.py      # ⚙️ The Value Function Iteration solver
│   └── ...
└── requirements.txt   # 📦 Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd bellman-opt
    ```
2.  Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

To run the models, visualize the results, and see the analysis, open and run the Jupyter Notebook located in the `presentation/` directory.

A convenient way to do this is by using the official **Jupyter extension for Visual Studio Code**.

1.  **Install the Extension**: If you haven't already, install the Jupyter extension from the VS Code Marketplace.
2.  **Open the Notebook**: Open the `.ipynb` file from the `presentation/` folder directly within VS Code.
3.  **Select a Kernel**: VS Code will prompt you to select a Python interpreter (kernel). Choose the one where you installed the project's dependencies.
4.  **Run the Cells**: You can now run cells individually to step through the analysis or run all cells at once.
