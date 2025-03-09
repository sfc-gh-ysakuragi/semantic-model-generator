import streamlit as st
from loguru import logger
from snowflake.connector import ProgrammingError

from app_utils.shared_utils import (
    GeneratorAppScreen,
    format_snowflake_context,
    get_available_databases,
    get_available_schemas,
    get_available_tables,
    input_sample_value_num,
    input_semantic_file_name,
    run_generate_model_str_from_snowflake,
)


def update_schemas_and_tables() -> None:
    """
    Callback to run when the selected databases change. Ensures that if a database is deselected, the corresponding
    schemas and tables are also deselected.
    Returns: None

    """
    databases = st.session_state["selected_databases"]

    # Fetch the available schemas for the selected databases
    schemas = []
    for db in databases:
        try:
            schemas.extend(get_available_schemas(db))
        except ProgrammingError:
            logger.info(
                f"Insufficient permissions to read from database {db}, skipping"
            )

    st.session_state["available_schemas"] = schemas

    # Enforce that the previously selected schemas are still valid
    valid_selected_schemas = [
        schema for schema in st.session_state["selected_schemas"] if schema in schemas
    ]
    st.session_state["selected_schemas"] = valid_selected_schemas
    update_tables()


def update_tables() -> None:
    """
    Callback to run when the selected schemas change. Ensures that if a schema is deselected, the corresponding
    tables are also deselected.
    """
    schemas = st.session_state["selected_schemas"]

    # Fetch the available tables for the selected schemas
    tables = []
    for schema in schemas:
        try:
            tables.extend(get_available_tables(schema))
        except ProgrammingError:
            logger.info(
                f"Insufficient permissions to read from schema {schema}, skipping"
            )
    st.session_state["available_tables"] = tables

    # Enforce that the previously selected tables are still valid
    valid_selected_tables = [
        table for table in st.session_state["selected_tables"] if table in tables
    ]
    st.session_state["selected_tables"] = valid_selected_tables


@st.experimental_dialog("対象テーブルを選択", width="large")
def table_selector_dialog() -> None:
    st.write(
        "セマンティックモデルの構築を開始するには、以下のフィールドに入力してください。"
    )
    model_name = input_semantic_file_name()
    sample_values = input_sample_value_num()
    st.markdown("")

    if "selected_databases" not in st.session_state:
        st.session_state["selected_databases"] = []

    if "selected_schemas" not in st.session_state:
        st.session_state["selected_schemas"] = []

    if "selected_tables" not in st.session_state:
        st.session_state["selected_tables"] = []

    with st.spinner("データベースを読み込み中..."):
        available_databases = get_available_databases()

    st.multiselect(
        label="データベース",
        options=available_databases,
        placeholder="セマンティックモデルに含めたいテーブルを含むデータベースを選択",
        on_change=update_schemas_and_tables,
        key="selected_databases",
        # default=st.session_state.get("selected_databases", []),
    )

    st.multiselect(
        label="スキーマ",
        options=st.session_state.get("available_schemas", []),
        placeholder="セマンティックモデルに含めたいテーブルを含むスキーマを選択",
        on_change=update_tables,
        key="selected_schemas",
        format_func=lambda x: format_snowflake_context(x, -1),
    )

    st.multiselect(
        label="テーブル",
        options=st.session_state.get("available_tables", []),
        placeholder="セマンティックモデルに含めたいテーブルを選択",
        key="selected_tables",
        format_func=lambda x: format_snowflake_context(x, -1),
    )

    st.markdown("<div style='margin: 240px;'></div>", unsafe_allow_html=True)
    experimental_features = st.checkbox(
        "結合を有効にする（任意）",
        help="このボックスにチェックを入れると、セマンティックモデル内の結合パスを追加/編集できるようになります。この設定を有効にする場合は、Snowflakeアカウントに適切なパラメータが設定されていることを確認してください。アクセス方法については、アカウントチームにお問い合わせください。",
    )

    st.session_state["experimental_features"] = experimental_features

    submit = st.button("送信", use_container_width=True, type="primary")
    if submit:
        try:
            run_generate_model_str_from_snowflake(
                model_name,
                sample_values,
                st.session_state["selected_tables"],
                allow_joins=experimental_features,
            )
            st.session_state["page"] = GeneratorAppScreen.ITERATION
            st.rerun()
        except ValueError as e:
            st.error(e)


def show() -> None:
    table_selector_dialog()
