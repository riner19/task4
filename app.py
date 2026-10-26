import json

from dotenv import load_dotenv
load_dotenv()
from dash import (Dash, html, dcc, Input, Output, State, dash_table, no_update,
                  ALL, callback_context)
import plotly.express as px
import pandas as pd

from task4 import STORES
import tools as T
import llm

WH = T.Warehouse("data")
DATA = {s: WH.by_store[s]["res"] for s in STORES}

app = Dash(__name__, title="Bookstore BI", suppress_callback_exceptions=True)
server = app.server                      # gunicorn entry point

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
                  "borderBottom": "1px solid #bbb"},
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
                **TABLE_KW),
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



EXAMPLES = [
    "How many unique users are there?",
    "Which day had the highest revenue and how much?",
    "Who is the best buyer and what are all their user ids?",
    "How many books in the Fantasy genre?",
    "How many users have an e-mail starting with s or p?",
    "Which store made the most revenue?",
]


def ask_tab(store):
    return html.Div([
        html.Div(f"Ask about the data. Questions default to {store} unless you name "
                 "another store.", style={**LABEL, "marginBottom": "12px"}),

        html.Div([
            dcc.Input(id="chat-input", type="text", debounce=False,
                      placeholder="e.g. how many unique users are there?",
                      style={"flex": "1 1 auto", "padding": "9px 11px",
                             "fontSize": "14px", "fontFamily": FONT,
                             "border": f"1px solid {LINE}", "borderRadius": "4px"}),
            html.Button("Ask", id="chat-send", n_clicks=0,
                        style={"padding": "9px 18px", "fontSize": "14px",
                               "fontFamily": FONT, "cursor": "pointer",
                               "border": f"1px solid {LINE}", "background": "#f5f5f5",
                               "borderRadius": "4px"}),
        ], style={"display": "flex", "gap": "8px"}),

        html.Div([html.Span("Try: ", style=LABEL)] + [
            html.Button(q, id={"type": "example", "index": i}, n_clicks=0,
                        style={"fontSize": "12px", "fontFamily": FONT, "cursor": "pointer",
                               "border": f"1px solid {LINE}", "background": "white",
                               "borderRadius": "12px", "padding": "3px 10px",
                               "margin": "3px 4px 0 0", "color": GREY})
            for i, q in enumerate(EXAMPLES)
        ], style={"marginTop": "10px"}),

        dcc.Loading(html.Div(id="chat-log", style={"marginTop": "22px"}),
                    type="default", color="#888"),
    ])


def render_log(history, traces):
    out = []
    for i, m in enumerate(history):
        if m["role"] == "user":
            out.append(html.Div([
                html.Div("You", style={**LABEL, "fontSize": "11px"}),
                html.Div(m["content"], style={"fontSize": "14px", "marginTop": "2px"}),
            ], style={"marginBottom": "14px"}))
        else:
            tr = traces.get(str(i), [])
            block = [
                html.Div("Answer", style={**LABEL, "fontSize": "11px"}),
                html.Div(m["content"], style={"fontSize": "14px", "marginTop": "2px",
                                              "whiteSpace": "pre-wrap"}),
                html.Details([
                    html.Summary(
                        f"{len(tr)} API call{'s' if len(tr) != 1 else ''}",
                        style={"fontSize": "11.5px", "color": GREY, "cursor": "pointer"}),
                    html.Div([
                        html.Div([
                            html.Div(
                                f"{c['tool']}({', '.join(f'{k}={json.dumps(v)}' for k, v in c['args'].items())})",
                                style={"fontFamily": MONO, "fontSize": "11.5px"}),
                            html.Div(json.dumps(c["result"], default=str)[:400],
                                     style={"fontFamily": MONO, "fontSize": "11px",
                                            "color": GREY, "marginTop": "2px",
                                            "wordBreak": "break-all"}),
                        ], style={"marginBottom": "8px"}) for c in tr
                    ], style={"marginTop": "6px", "paddingLeft": "10px",
                              "borderLeft": f"2px solid {LINE}"}),
                ], style={"marginTop": "6px"})]
            if not tr:
                block = block[:-1]
            out.append(html.Div(block, style={"marginBottom": "22px",
                                              "paddingBottom": "18px",
                                              "borderBottom": "1px solid #eee"}))
    return out or html.Div("No questions yet.", style=LABEL)


# --------------------------------------------------------------------------- layout
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
                style={"width": "160px", "fontFamily": MONO, "fontSize": "13px"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "gap": "16px", "flexWrap": "wrap",
                  "borderBottom": f"1px solid {LINE}", "paddingBottom": "14px"}),

        dcc.Store(id="view-state", data={"store": STORES[0]}),
        dcc.Store(id="chat-history", data=[]),
        dcc.Store(id="chat-traces", data={}),

        dcc.Tabs(id="tabs", value="revenue", style={"marginTop": "14px"}, children=[
            dcc.Tab(label="Revenue", value="revenue"),
            dcc.Tab(label="Customers", value="customers"),
            dcc.Tab(label="Catalogue", value="catalogue"),
            dcc.Tab(label="Ask", value="ask"),
        ]),

        html.Div(id="tab-content", style={"paddingTop": "26px"}),
    ])


# ------------------------------------------------------------------------- callbacks
@app.callback(Output("view-state", "data"), Input("store-picker", "value"))
def pick_store(store):
    return {"store": store}


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"), Input("view-state", "data"))
def render_tab(tab, state):
    res = DATA[state["store"]]
    if tab == "revenue":
        return revenue_tab(res)
    if tab == "customers":
        return customers_tab(res)
    if tab == "catalogue":
        return catalogue_tab(res)
    return ask_tab(state["store"])


@app.callback(Output("chat-input", "value"),
              Input({"type": "example", "index": ALL}, "n_clicks"),
              prevent_initial_call=True)
def fill_example(clicks):
    ctx = callback_context
    if not ctx.triggered or not any(clicks or []):
        return no_update
    idx = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    return EXAMPLES[idx]


@app.callback(Output("chat-history", "data"), Output("chat-traces", "data"),
              Input("chat-send", "n_clicks"), Input("chat-input", "n_submit"),
              State("chat-input", "value"), State("chat-history", "data"),
              State("chat-traces", "data"), State("view-state", "data"),
              prevent_initial_call=True)
def send(n_clicks, n_submit, question, history, traces, state):
    question = (question or "").strip()
    if not question:
        return no_update, no_update

    history = list(history or [])
    traces = dict(traces or {})
    history.append({"role": "user", "content": question})

    try:
        out = llm.ask(WH, question, history=history[:-1],
                      default_store=state["store"])
        answer = out["answer"] or out["error"] or "(no answer)"
        trace = out["trace"]
    except llm.NoProvider as e:
        answer, trace = str(e), []
    except Exception as e:                       # surface failures, never fake an answer
        answer, trace = f"The request failed: {type(e).__name__}: {e}", []

    history.append({"role": "assistant", "content": answer})
    traces[str(len(history) - 1)] = trace
    return history, traces


@app.callback(Output("chat-log", "children"),
              Input("chat-history", "data"), Input("chat-traces", "data"))
def show_log(history, traces):
    return render_log(history or [], traces or {})


if __name__ == "__main__":
    app.run(debug=True)