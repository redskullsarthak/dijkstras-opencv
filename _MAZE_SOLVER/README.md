
# Maze Solver

This project solves a maze using Python and visualizes the solution.

## Files
- `maze.png`: The input maze image.
- `solve.py`: Python script to solve the maze.
- `solution.png`: Output image showing the solved maze path.
- `README.md`: Project documentation.

## How to Use
1. Place your maze image as `maze.png` in the project folder.
2. Run `solve.py` to solve the maze:
	 ```bash
	 python solve.py
	 ```
3. The solution will be saved as `solution.png`.

## Requirements
- Python 3.x
- Required packages (install with pip):
	- OpenCV (`opencv-python`)
	- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

## Description
The script reads the maze image, finds the start and end points, solves the maze using pathfinding algorithms, and outputs the solution image.


