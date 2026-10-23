from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

from task4 import process_store, STORES

# Run the pipeline once at import. Never inside a callback.
DATA = {s: process_store(s) for s in STORES}

app = Dash(__name__, title="Bookstore BI")
server = app.server                      # gunicorn entry point

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px",
           "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H1("Bookstore BI", style={"fontSize": "20px"}),

        # every view reads from here, so the AI can drive it in part 3
        dcc.Store(id="view-state", data={"store": "STORE1"}),

        dcc.Dropdown(
            id="store-picker",
            options=[{"label": s, "value": s} for s in STORES],
            value="STORE1",
            clearable=False,
            style={"maxWidth": "220px"},
        ),

        dcc.Tabs(id="tabs", value="revenue", children=[
            dcc.Tab(label="Revenue", value="revenue"),
            dcc.Tab(label="Customers", value="customers"),
            dcc.Tab(label="Catalogue", value="catalogue"),
        ]),

        html.Div(id="tab-content", style={"paddingTop": "16px"}),
    ],
)


@app.callback(Output("view-state", "data"),
              Input("store-picker", "value"),
              prevent_initial_call=True)
def pick_store(store):
    return {"store": store}


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"),
              Input("view-state", "data"))
def render_tab(tab, state):
    res = DATA[state["store"]]

    if tab == "revenue":
        # TODO: top 5 days (YYYY-MM-dd) + daily revenue line chart
        return html.Div("revenue")

    if tab == "customers":
        # TODO: unique users count + best buyer as [id1, id2, ...]
        return html.Div("customers")

    # TODO: unique author sets count + most popular author(s)
    return html.Div("catalogue")


if __name__ == "__main__":
    app.run(debug=True)