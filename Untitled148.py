#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import deque

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u in self.adj_list:
            self.adj_list[u].append(v)
        else:
            self.adj_list[u] = [v]

    def bfs(self, start):
        visited = set()
        queue = deque()

        queue.append(start)
        visited.add(start)

        while queue:
            node = queue.popleft()
            print(node, end=' ')

            if node in self.adj_list:
                neighbors = self.adj_list[node]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
# Create a graph
graph = Graph()

# Add edges to the graph
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 3)
graph.add_edge(1, 4)
graph.add_edge(2, 5)
graph.add_edge(2, 6)

# Perform BFS starting from node 0
print("BFS:")
graph.bfs(0)
print()


# In[2]:


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u in self.adj_list:
            self.adj_list[u].append(v)
        else:
            self.adj_list[u] = [v]

    def dfs(self, start):
        visited = set()
        self._dfs_helper(start, visited)

    def _dfs_helper(self, node, visited):
        visited.add(node)
        print(node, end=' ')

        if node in self.adj_list:
            neighbors = self.adj_list[node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    self._dfs_helper(neighbor, visited)
# Create a graph
graph = Graph()

# Add edges to the graph
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 3)
graph.add_edge(1, 4)
graph.add_edge(2, 5)
graph.add_edge(2, 6)

# Perform DFS starting from node 0
print("DFS:")
graph.dfs(0)
print()


# In[3]:


from collections import deque

class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def count_nodes_at_level(root, target_level):
    if root is None:
        return 0
    
    queue = deque()
    queue.append((root, 0))  # Store the node and its level
    count = 0

    while queue:
        node, level = queue.popleft()

        if level == target_level:
            count += 1
        
        if node.left:
            queue.append((node.left, level + 1))
        
        if node.right:
            queue.append((node.right, level + 1))

    return count
# Create a binary tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

# Define the target level
target_level = 2

# Count the number of nodes at the target level
count = count_nodes_at_level(root, target_level)
print("Number of nodes at level", target_level, ":", count)


# In[4]:


class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_list = [[] for _ in range(num_nodes)]

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def count_trees_in_forest(self):
        visited = [False] * self.num_nodes
        count = 0

        for node in range(self.num_nodes):
            if not visited[node]:
                if self.is_tree(node, visited):
                    count += 1

        return count

    def is_tree(self, node, visited):
        stack = [(node, -1)]  # Store the node and its parent
        visited[node] = True

        while stack:
            current, parent = stack.pop()

            for neighbor in self.adj_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append((neighbor, current))
                elif neighbor != parent:
                    return False

        return True
# Create a graph representing the forest
num_nodes = 8
forest = Graph(num_nodes)

# Add edges to the graph
forest.add_edge(0, 1)
forest.add_edge(1, 2)
forest.add_edge(3, 4)
forest.add_edge(4, 5)
forest.add_edge(5, 3)
forest.add_edge(6, 7)

# Count the number of trees in the forest
count = forest.count_trees_in_forest()
print("Number of trees in the forest:", count)


# In[5]:


class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_list = [[] for _ in range(num_nodes)]

    def add_edge(self, u, v):
        self.adj_list[u].append(v)

    def is_cyclic(self):
        visited = [False] * self.num_nodes
        rec_stack = [False] * self.num_nodes

        for node in range(self.num_nodes):
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True

        return False

    def is_cyclic_util(self, node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in self.adj_list[node]:
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[node] = False
        return False
# Create a directed graph
num_nodes = 4
graph = Graph(num_nodes)

# Add edges to the graph
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 0)
graph.add_edge(2, 3)

# Detect if the graph contains a cycle
has_cycle = graph.is_cyclic()
if has_cycle:
    print("The graph contains a cycle")
else:
    print("The graph does not contain a cycle")


# In[6]:


def solve_n_queens(n):
    board = [['.'] * n for _ in range(n)]
    solutions = []

    def is_safe(row, col):
        # Check if no queen threatens the current position

        # Check row
        for i in range(col):
            if board[row][i] == 'Q':
                return False

        # Check upper diagonal
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        # Check lower diagonal
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        return True

    def backtrack(col):
        # Base case: All queens have been placed
        if col == n:
            solution = [''.join(row) for row in board]
            solutions.append(solution)
            return

        # Try placing a queen in each row of the current column
        for row in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(col + 1)
                board[row][col] = '.'

    # Start backtracking from the first column
    backtrack(0)

    return solutions
# Solve the 4-Queens problem
n = 4
solutions = solve_n_queens(n)

# Print the solutions
for solution in solutions:
    for row in solution:
        print(row)
    print()


# In[ ]:




