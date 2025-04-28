import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 Ultimate Best-First Search Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Controls ---
st.sidebar.title("🔧 Controls")
mode = st.sidebar.radio("Mode:", ["📚 Explanation", "🔍 Interactive Demo"])

# --- Helper: Build Default Graph ---
@st.cache_data
def load_default_graph():
    G = nx.Graph()
    heuristics = {chr(65+i): 14-i for i in range(7)}  # A=14, B=13, ..., G=8 (adjust as needed)
    heuristics['G'] = 0
    for n, h in heuristics.items():
        G.add_node(n, h=h)
    edges = [
        ('A','B',4), ('A','C',3), ('B','D',5), ('C','E',4),
        ('D','G',9), ('E','G',6), ('C','F',7), ('F','G',5)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G

# --- Best-First Search with Step Tracing ---
def best_first_search(graph, start, goal):
    open_set = [(graph.nodes[start]['h'], start)]
    came_from = {}
    visited = set()
    history = []

    while open_set:
        open_set.sort(key=lambda x: x[0])
        h, current = open_set.pop(0)
        visited.add(current)
        history.append({
            'current': current,
            'frontier': [n for _, n in open_set],
            'visited': list(visited)
        })
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, history
        for neigh in sorted(graph.neighbors(current)):
            if neigh not in visited and neigh not in (n for _, n in open_set):
                came_from[neigh] = current
                open_set.append((graph.nodes[neigh]['h'], neigh))
    return None, history

# --- Explanation Page ---
if mode == "📚 Explanation":
    st.title("📖 Greedy Best-First Search Explained")
    st.markdown("""
Greedy Best-First Search prioritizes nodes based solely on a heuristic `h(n)`, striving to reach the target quickly but without guaranteed optimality.

- **Heuristic (h):** Estimate from a node to the goal.
- **Open Set:** Frontier nodes in a priority queue ordered by `h(n)`.
- **Visited Set:** Avoids re-expansion.

**Advantages:** Fast exploration on many graphs.  
**Drawbacks:** May bypass shorter paths or get stuck.
""")
    with st.expander("📜 Pseudocode", expanded=True):
        st.code(
"""
GREEDY-BFS(start, goal):
  open_set = priority queue by h(n)
  open_set.insert(start)
  came_from = {}
  visited = set()
  while open_set not empty:
    current = open_set.pop()  # smallest heuristic
    if current == goal:
      return reconstruct_path(came_from, current)
    for neighbor in neighbors(current):
      if neighbor not in visited:
        came_from[neighbor] = current
        open_set.insert(neighbor)
  return failure
""", language='python')
    st.header("🔧 Complexity & Properties")
    st.markdown("""
- **Time Complexity:** O(b^m) worst-case  
- **Space Complexity:** O(b^m)  
- **Complete?:** No  
- **Optimal?:** No
""")
    st.header("🗺️ Example Graph")
    G0 = load_default_graph()
    pos0 = nx.spring_layout(G0, seed=42)
    fig, ax = plt.subplots(figsize=(6,4))
    nx.draw(G0, pos0, with_labels=True, node_color='skyblue', node_size=700, ax=ax)
    nx.draw_networkx_edge_labels(G0, pos0, edge_labels=nx.get_edge_attributes(G0,'weight'), ax=ax)
    for n in G0.nodes:
        x, y = pos0[n]
        ax.text(x, y-0.15, f"h={G0.nodes[n]['h']}", ha='center', fontsize=10)
    ax.set_title("Sample Graph (Heuristics)")
    ax.axis('off')
    st.pyplot(fig)

# --- Interactive Demo Page ---
else:
    st.title("🎯 Live Best-First Search Demo")
    st.markdown("Paste your graph below (CSV columns: u,v,weight,h_u,h_v).\nExample CSV provided below for testing.")
    example_csv = """
u,v,weight,h_u,h_v
A,B,4,14,12
A,C,3,14,11
B,D,5,12,10
C,E,4,11,7
C,F,7,11,6
D,G,9,10,0
E,G,6,7,0
F,G,5,6,0
"""
    with st.expander("📄 Example CSV", expanded=False):
        st.code(example_csv)

    raw = st.text_area("Graph CSV input:", height=200, value=example_csv)
    try:
        df = pd.read_csv(StringIO(raw))
        G = nx.Graph()
        # Build nodes
        for _, row in df.iterrows():
            if row.u not in G:
                G.add_node(row.u, h=row.h_u)
            if row.v not in G:
                G.add_node(row.v, h=row.h_v)
            G.add_edge(row.u, row.v, weight=row.weight)
    except Exception as e:
        st.error(f"Invalid CSV format: {e}")
        st.stop()

    # Precompute layout for consistent plots
    pos = nx.spring_layout(G, seed=42)
    nodes = list(G.nodes)
    start = st.sidebar.selectbox("Start node", nodes, index=0)
    goal = st.sidebar.selectbox("Goal node", nodes, index=len(nodes)-1)

    if st.sidebar.button("▶️ Run Search"):
        path, history = best_first_search(G, start, goal)
        if not history:
            st.warning("No search steps—check your graph.")
        else:
            st.subheader("🔢 Steps Visualization")
            cols = st.columns(4)
            for idx, state in enumerate(history):
                col = cols[idx % 4]
                fig, ax = plt.subplots(figsize=(2,2))
                colors = []
                for n in G.nodes:
                    if n == state['current']:
                        colors.append('red')
                    elif n in state['visited']:
                        colors.append('lightgreen')
                    elif n in state['frontier']:
                        colors.append('orange')
                    else:
                        colors.append('skyblue')
                nx.draw(G, pos, with_labels=True, node_color=colors, node_size=300, font_size=6, ax=ax)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'), font_size=6, ax=ax)
                ax.set_title(f"Step {idx+1}", fontsize=8)
                ax.axis('off')
                col.pyplot(fig)
                if idx % 4 == 3 and idx < len(history)-1:
                    cols = st.columns(4)
            # Results
            if path:
                st.success(f"✅ Path: {' → '.join(path)}")
                st.info(f"Visited order: {' → '.join(history[-1]['visited'])}")
            else:
                st.error("❌ No path found.")
    else:
        st.info("Choose start/goal then click ▶️ Run Search.")
