
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd

from task4 import process_store, STORES

DATA = {s: process_store(s) for s in STORES}

app = Dash(__name__, title="Bookstore BI")
server = app.server                  

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
GREY = "#666"
LINE = "#ddd"

LABEL = {"fontSize": "12px", "color": GREY}
VALUE = {"fontFamily": MONO, "fontSize": "24px", "marginTop": "2px"}
H3 = {"fontSize": "14px", "fontWeight": "600", "margin": "0 0 10px 0"}
BLOCK = {"marginBottom": "26px"}

TABLE_KW = dict(
    style_cell={"fontFamily": MONO, "fontSize": "13px", "padding": "7px 24px 7px 0",
                "textAlign": "left", "border": "none",
                "borderBottom": f"1px solid {LINE}"},
    style_header={"fontFamily": FONT, "fontSize": "12px", "fontWeight": "600",
                  "color": GREY, "backgroundColor": "white", "border": "none",
                  "borderBottom": f"1px solid #bbb"},
    style_as_list_view=True,
)


def stat(label, value):
    return html.Div([html.Div(label, style=LABEL), html.Div(value, style=VALUE)],
                    style=BLOCK)


def revenue_tab(res):
    dr = res["daily_revenue"]
    daily = pd.DataFrame({"date": pd.to_datetime(dr.index), "revenue": dr.values})

    # days with no orders earned $0 - fill them so the line doesn't bridge the gaps
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (daily.set_index("date").reindex(idx, fill_value=0.0)
                  .rename_axis("date").reset_index())

    fig = px.line(daily, x="date", y="revenue")
    fig.update_traces(line_width=1.2, line_color="#333",
                      hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=340,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family=FONT, size=12, color=GREY),
                      xaxis_title=None, yaxis_title=None, hovermode="x unified")
    fig.update_yaxes(gridcolor="#eee", tickprefix="$", zeroline=False)
    fig.update_xaxes(showgrid=False)

    top5 = pd.DataFrame({
        "Date": list(res["top_5_days"].index),          # already YYYY-MM-dd
        "Revenue": [f"{v:,.2f}" for v in res["top_5_days"].values],
    })

    return html.Div([
        html.Div([
            html.H3("Top 5 days by revenue", style=H3),
            dash_table.DataTable(
                data=top5.to_dict("records"),
                columns=[{"name": c, "id": c} for c in top5.columns],
                **TABLE_KW,
            ),
        ], style={"marginBottom": "30px", "maxWidth": "340px"}),

        html.Div([
            html.H3("Daily revenue", style=H3),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]),
    ])


def customers_tab(res):
    ids = res["best_buyer_ids"]
    return html.Div([
        stat("Unique users", f"{res['unique_users']:,}"),
        stat("User records in file", f"{res['user_records']:,}"),
        html.Div([
            html.Div("Best buyer", style=LABEL),
            html.Div("[" + ", ".join(str(i) for i in ids) + "]",
                     style={**VALUE, "fontSize": "20px"}),
            html.Div(", ".join(res["best_buyer_names"]),
                     style={"fontSize": "13px", "marginTop": "6px"}),
            html.Div(f"${res['best_buyer_spend']:,.2f} total",
                     style={"fontFamily": MONO, "fontSize": "13px",
                            "color": GREY, "marginTop": "2px"}),
        ], style=BLOCK),
    ])


def catalogue_tab(res):
    authors = res["most_popular_authors"]
    return html.Div([
        stat("Unique sets of authors", f"{res['unique_author_sets']:,}"),
        html.Div([
            html.Div("Most popular author(s)", style=LABEL),
            html.Div([html.Div(a, style={"fontSize": "18px", "marginTop": "3px"})
                      for a in authors]),
            html.Div(f"{res['most_popular_sold']:,} copies sold",
                     style={"fontFamily": MONO, "fontSize": "13px",
                            "color": GREY, "marginTop": "5px"}),
        ], style=BLOCK),
    ])


app.layout = html.Div(
    style={"maxWidth": "920px", "margin": "0 auto", "padding": "28px 20px 60px",
           "fontFamily": FONT, "color": "#111"},
    children=[
        html.Div([
            html.H1("Bookstore BI", style={"fontSize": "17px", "margin": "0",
                                           "fontWeight": "600"}),
            dcc.Dropdown(
                id="store-picker",
                options=[{"label": s, "value": s} for s in STORES],
                value=STORES[0], clearable=False,
                style={"width": "160px", "fontFamily": MONO, "fontSize": "13px"},
            ),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "gap": "16px", "flexWrap": "wrap",
                  "borderBottom": f"1px solid {LINE}", "paddingBottom": "14px"}),

        # every view reads from here, so an assistant can drive the UI in part 3
        dcc.Store(id="view-state", data={"store": STORES[0]}),

        dcc.Tabs(id="tabs", value="revenue", style={"marginTop": "14px"}, children=[
            dcc.Tab(label="Revenue", value="revenue"),
            dcc.Tab(label="Customers", value="customers"),
            dcc.Tab(label="Catalogue", value="catalogue"),
        ]),

        html.Div(id="tab-content", style={"paddingTop": "26px"}),
    ])


@app.callback(Output("view-state", "data"),
              Input("store-picker", "value"))
def pick_store(store):
    return {"store": store}


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"),
              Input("view-state", "data"))
def render_tab(tab, state):
    res = DATA[state["store"]]
    if tab == "revenue":
        return revenue_tab(res)
    if tab == "customers":
        return customers_tab(res)
    return catalogue_tab(res)


if __name__ == "__main__":
    app.run(debug=True)