import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import streamlit as st
from st_cytoscape import cytoscape

num_observations = 10_000
generating_model = """
Z <-- N
C1 <-- N
C21 <-- N
C31 <-- N
C33 <-- N
C22 <-- C21 + N
C32 <-- C31 + C33 + N
X <-- Z + C1 + C21 + C31 + N
M <-- X + N
Y <-- M + C1 + C22 + C33 + N
C4 <-- X + Y + N
"""


def rewrite(x):
    if x == "N":
        return "np.random.randn(num_observations)"
    else:
        return f'd["{x}"]'


@st.cache()
def generate_data():
    np.random.seed(seed=0)
    d = {}
    nodes = set()
    edges = set()
    for line in generating_model.split("\n"):
        if " <-- " in line:
            left, right = line.split(" <-- ")
            right_terms = right.split(" + ")
            nodes.add(left)
            for node in right_terms:
                if node != "N":
                    nodes.add(node)
                    edges.add((node, left))
            formula = f"{rewrite(left)} = {' + '.join(list(map(rewrite, right_terms)))}"
            exec(formula)
    return pd.DataFrame.from_dict(d), nodes, edges


df, nodes, edges = generate_data()
elements = []
for node in nodes:
    elements.append(
        {
            "data": {"id": node},
            "selected": node == "X",
            "selectable": node not in ["X", "Y"],
        }
    )
for edge in edges:
    elements.append(
        {
            "data": {
                "source": edge[0],
                "target": edge[1],
                "id": f"{edge[0]}-{edge[1]}",
            },
            "selectable": False,
        }
    )
stylesheet = [
    {"selector": "node", "style": {"label": "data(id)", "width": 20, "height": 20}},
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
]

layout = {"name": "fcose", "animationDuration": 0}
layout["alignmentConstraint"] = {"horizontal": [["Z", "X", "M", "Y"]]}
layout["relativePlacementConstraint"] = [{"left": "X", "right": "Y"}]
layout["relativePlacementConstraint"].append({"top": "C1", "bottom": "X"})
layout["relativePlacementConstraint"].append({"top": "C21", "bottom": "X"})
layout["relativePlacementConstraint"].append({"top": "X", "bottom": "C4"})
layout["relativePlacementConstraint"].append({"top": "X", "bottom": "C31"})
layout["nodeRepulsion"] = 50000

st.sidebar.title("Causal simulator")

st.sidebar.markdown(
    """
Does the **partial dependence plot** of a certain input feature reflect its **causal effect** on the target variable?

With this demo, you can check whether it's the case for several learning algorithms and various subsets of the input features.

*Inspired by [The Book of Why](http://bayes.cs.ucla.edu/WHY/) by Judea Pearl and Dana Mackenzie and built by [Vivien](https://twitter.com/vivien000000) with [Streamlit](https://streamlit.io/), [Cytoscape.js](https://js.cytoscape.org/) and [scikit-learn](https://scikit-learn.org/stable/)*
"""
)

st.subheader("Data generating process")

st.markdown(
    """
10,000 observations have been generated for the variables mentioned in the causal graph below. The values for each variable were derived from the values of its parents in the causal graph as follows:
"""
)
st.latex(
    "U = \sum_{V \in \mathrm{\ Parents}(U)} V + \epsilon_U  \quad \mathrm{where} \quad \epsilon_U \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)"
)
st.subheader(
    "Partial dependence plot for X with various learning algorithms and input features"
)

order = ["X"] + sorted([n for n in nodes if n not in ["X", "Y"]])

algorithm = st.selectbox(
    "Select a learning algorithm",
    ["Linear regression", "Random forest", "Gradient boosting"],
    index=1,
)

st.markdown(
    '<p class="css-qrbaxs effi0qh0">Select the input features to include in the model</p>',
    unsafe_allow_html=True,
)
selected = cytoscape(
    elements,
    stylesheet,
    height="450px",
    layout=layout,
    selection_type="additive",
    user_panning_enabled=False,
    user_zooming_enabled=False,
    key="graph",
)

try:
    selected_nodes = [n for n in order if n in selected["nodes"]]
    if algorithm == "Linear regression":
        regr = linear_model.LinearRegression()
    elif algorithm == "Random forest":
        regr = RandomForestRegressor()
    elif algorithm == "Gradient boosting":
        regr = GradientBoostingRegressor()
    regr.fit(df[selected_nodes], df[["Y"]])

    fig, ax = plt.subplots(1, 1)
    pdp = PartialDependenceDisplay.from_estimator(
        regr, df[selected_nodes], [0], ax=ax, pd_line_kw={"color": "#999999"}
    )
    sorted_x = np.sort(df[selected_nodes]["X"])
    x_min, x_max = (
        sorted_x[len(sorted_x) * 5 // 100],
        sorted_x[len(sorted_x) * 95 // 100],
    )
    pdp.axes_[0][0].plot(
        [x_min, x_max],
        [x_min, x_max],
        color="#FF5555",
        label="Causal effect of X on Y",
    )
    pdp.axes_[0][0].axis(xmin=x_min, xmax=x_max, ymin=x_min, ymax=x_max)
    pdp.axes_[0][0].legend()
    st.pyplot(fig)
except TypeError:
    pass
