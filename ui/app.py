import streamlit as st
import uuid
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.graph import build_graph
from src.orchestrator.state import GraphState
from src.ingestion.schema_discovery import discover_schema
from src.preprocessing.profiler import DataProfiler
from src.memory.conversation import ConversationMemory
from src.models.query import RouteType
from src.db.connection import db
from config.settings import settings


st.set_page_config(page_title="BI Agent", layout="wide")
st.title("BI Agent")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "schema" not in st.session_state:
    try:
        with st.spinner("Connecting to database and discovering schema..."):
            from src.startup import initialize
            schema, health_cards = initialize()
            st.session_state.schema = schema
            st.session_state.health_cards = health_cards
    except Exception as e:
        st.error(f"**Database connection failed:** {e}\n\nCheck your `.env` file and ensure PostgreSQL is running.")
        st.code("DATABASE_URL=postgresql://bi_agent:password@localhost:5432/bi_agent_db")
        st.stop()
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "turn" not in st.session_state:
    st.session_state.turn = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Database")
    schema = st.session_state.schema

    if schema.tables:
        for table in schema.tables:
            st.text(f"{table.name}  ({table.row_count:,} rows)")
    else:
        st.info("No tables found. Upload a CSV to get started.")

    st.divider()

    # CSV Upload
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"Preview ({len(df_preview)} rows, {len(df_preview.columns)} columns):")
            st.dataframe(df_preview.head(10))

            suggested_name = uploaded_file.name.lower().replace(".csv", "").replace(" ", "_").replace("-", "_")
            table_name = st.text_input("Table name:", value=suggested_name)

            col1, col2 = st.columns(2)
            with col1:
                overwrite = st.checkbox("Overwrite if exists", value=False)
            with col2:
                confirm = st.button("Upload to DB")

            if confirm and table_name:
                # Sanitize table name: only alphanumeric and underscores
                safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name.lower())
                if not safe_name[0].isalpha():
                    safe_name = "t_" + safe_name

                engine = create_engine(settings.db_url)
                if_exists = "replace" if overwrite else "fail"
                try:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file)
                    df_upload.to_sql(safe_name, engine, if_exists=if_exists, index=False, chunksize=10000)

                    # Refresh schema
                    st.session_state.schema = discover_schema()

                    # Profile new table
                    profiler = DataProfiler()
                    card = profiler.profile_table(safe_name)
                    st.session_state.health_cards[safe_name] = card

                    st.success(f"Uploaded {len(df_upload):,} rows to `{safe_name}`")
                    if card.warnings:
                        for w in card.warnings[:3]:
                            st.warning(w)
                    st.rerun()
                except Exception as e:
                    if "already exists" in str(e).lower():
                        st.error(f"Table `{safe_name}` already exists. Check 'Overwrite if exists' to replace it.")
                    else:
                        st.error(f"Upload failed: {e}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.divider()

    # Data profiling
    st.subheader("Data Profiling")
    if schema.tables:
        selected_table = st.selectbox("Profile table:", [t.name for t in schema.tables])
        if st.button("Run Profiling"):
            profiler = DataProfiler()
            card = profiler.profile_table(selected_table)
            st.session_state.health_cards[selected_table] = card
            st.metric("Quality Score", f"{card.overall_quality_score:.0f}/100")
            for w in card.warnings:
                st.warning(w)

    # API key warning
    if not settings.GROQ_API_KEY:
        st.divider()
        st.error("No GROQ_API_KEY configured. Set it in `.env`. Get a key at https://console.groq.com")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _render_chart(chart_spec: dict, data: list[dict]):
    """Render a Plotly chart from spec + data. Returns None on any failure."""
    try:
        df = pd.DataFrame(data)
        ct = chart_spec.get("chart_type", "bar")
        x = chart_spec.get("x_column")
        y = chart_spec.get("y_column")
        color = chart_spec.get("group_by")

        # Validate columns exist
        if x and x not in df.columns:
            x = None
        if y and y not in df.columns:
            y = None
        if color and color not in df.columns:
            color = None

        if not x or not y:
            # Auto-detect: first string col as x, first numeric col as y
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if str_cols:
                x = str_cols[0]
            if num_cols:
                y = num_cols[0]

        if not x or not y:
            return None

        # Fallback: pie with too many categories → bar
        if ct == "pie" and df[x].nunique() > 20:
            ct = "bar"

        if ct == "bar":
            return px.bar(df, x=x, y=y, color=color)
        elif ct == "line":
            return px.line(df, x=x, y=y, color=color)
        elif ct == "pie":
            return px.pie(df, names=x, values=y)
        elif ct == "scatter":
            return px.scatter(df, x=x, y=y, color=color)
        else:
            return px.bar(df, x=x, y=y, color=color)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("chart"):
            st.plotly_chart(msg["chart"], use_container_width=True)
        if msg.get("data"):
            with st.expander("View data"):
                st.dataframe(pd.DataFrame(msg["data"]))
        if msg.get("trace"):
            with st.expander("Execution trace"):
                _t = msg["trace"]
                st.markdown(f"**Route:** {_t.get('route', '—')}")
                st.markdown(f"**Critique loops:** {_t.get('critique_loops', 0)}")
                st.markdown(f"**Preprocessing:** {'Yes' if _t.get('preprocessing') else 'No'}")
                if _t.get("sql"):
                    st.code(_t["sql"], language="sql")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about your data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.turn += 1

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            initial_state: GraphState = {
                "query": prompt,
                "conversation_context": st.session_state.memory.get_context_string(),
                "schema": st.session_state.schema,
                "health_cards": st.session_state.health_cards,
                "route": None,
                "decomposition": None,
                "sql_results": [],
                "trends": None,
                "anomalies": None,
                "narrative": "",
                "chart_spec": None,
                "critique_count": 0,
                "critique_approved": False,
                "revision_notes": None,
                "verification": None,
                "needs_preprocessing": False,
                "preprocessing_applied": False,
                "confidence": None,
                "trace": None,
                "error": None,
            }

            try:
                final_state = st.session_state.graph.invoke(initial_state)
            except Exception as e:
                st.error(f"Graph execution failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An error occurred: {e}",
                })
                st.stop()

            narrative = final_state.get("narrative", "No results.")
            confidence = final_state.get("confidence")

            # Confidence badge
            if confidence:
                conf_color = (
                    "green" if confidence.score >= 0.8
                    else "orange" if confidence.score >= 0.5
                    else "red"
                )
                st.markdown(f"**Confidence: {confidence.score:.0%}** :{conf_color}_circle:")
                if confidence.uncertain_aspects:
                    st.caption(f"Uncertain about: {', '.join(confidence.uncertain_aspects)}")

            st.write(narrative)

            # Build data for chart
            sql_results = final_state.get("sql_results", [])
            all_data = [row for r in sql_results for row in r.data]

            # Chart rendering with error handling
            chart = None
            chart_spec = final_state.get("chart_spec")
            if chart_spec and chart_spec.get("chart_type") != "none" and all_data:
                chart = _render_chart(chart_spec, all_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.caption("Chart unavailable — showing data table instead.")

            # Data table
            if all_data:
                with st.expander(f"View data ({len(all_data)} rows)"):
                    st.dataframe(pd.DataFrame(all_data))

            # SQL viewer
            if sql_results:
                with st.expander("View SQL"):
                    for r in sql_results:
                        if r.sql:
                            st.code(r.sql, language="sql")
                        if r.error:
                            st.error(r.error)

            # Anomalies
            anomalies = final_state.get("anomalies")
            if anomalies and anomalies.anomaly_count > 0:
                with st.expander(f"Anomalies ({anomalies.anomaly_count})"):
                    st.write(anomalies.summary)

            # Show pipeline errors if present (Issue #6)
            pipeline_error = final_state.get("error")
            if pipeline_error:
                st.warning(f"⚠️ Pipeline issue: {pipeline_error}")

            # Memory — pass structured data (Issue #14)
            sql_queries = [r.sql for r in sql_results if r.sql] if sql_results else []
            conf_score = confidence.score if confidence else 0.0
            st.session_state.memory.add_turn(
                user_message=prompt,
                assistant_response=narrative,
                sql_queries=sql_queries,
                confidence=conf_score,
            )

            # Save to history
            first_sql = sql_results[0].sql if sql_results else None
            st.session_state.messages.append({
                "role": "assistant",
                "content": narrative,
                "chart": chart,
                "data": all_data[:100] if all_data else None,
                "trace": {
                    "route": str(final_state.get("route")),
                    "critique_loops": final_state.get("critique_count", 0),
                    "preprocessing": final_state.get("preprocessing_applied", False),
                    "sql": first_sql,
                },
            })


