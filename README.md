# GRID-TIN Navigation Mesh Generator

An efficient Python algorithm for generating optimized Navigation Meshes from heightmaps (heightfields) in complex outdoor environments. 

The system implements an incremental **GRID to TIN (Triangulated Irregular Network)** strategy that transforms raw height matrices into highly optimized polygonal networks. Unlike traditional methods, this approach provides a much more accurate terrain representation for intelligent agents in video games and simulations by dynamically optimizing triangle density based on slope error.

## Key Features

* **GRID-TIN Conversion:** Transforms height matrices into optimized polygonal meshes.
* **Incremental Strategy:** Utilizes a custom greedy algorithm for mesh refinement.
* **Geometric Optimization:** Implements **Delaunay Triangulation** to guarantee well-conditioned triangles and prevent the formation of "slivers".
* **Semantic Error Analysis:** Includes comparative metrics (RMSE, MAE, and Maximum Error) evaluating both height discrepancy and angular slope error across different generation algorithms.
* **Visual Debugging:** Built-in visualization tools to render terrain differences and angular errors.

## Tech Stack

* **Language:** Python 3
* **Core Libraries:** * `NumPy` (Matrix operations and data handling)
  * `SciPy` (Delaunay triangulation and numerical computing)
  * `Matplotlib` (Terrain rendering and visual analysis)

## Project Structure

* `src/` - Core algorithms for NavMesh generation (includes both height-error and angular-error approaches).
* `data/` - Test matrices (`.npy` files) including real-world topography data (e.g., Pyrenees Bassiero peak 1500x1500) and smaller 4x4 test matrices.
* `media/` - Rendered outputs and video demonstrations of the algorithmic error analysis.

## Getting Started

### Prerequisites
Ensure you have Python 3 installed. You can install the required dependencies using:
```bash
pip install numpy scipy matplotlib
