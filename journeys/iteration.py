import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import sqlglot
import streamlit as st
from snowflake.connector import ProgrammingError, SnowflakeConnection
from streamlit import config
from streamlit.delta_generator import DeltaGenerator
from streamlit_extras.row import row
from streamlit_extras.stylable_container import stylable_container
from snowflake.cortex import Translate

from app_utils.chat import send_message
from app_utils.shared_utils import (
    GeneratorAppScreen,
    SnowflakeStage,
    changed_from_last_validated_model,
    download_yaml,
    get_snowflake_connection,
    get_yamls_from_stage,
    init_session_states,
    return_home_button,
    stage_selector_container,
    upload_yaml,
    validate_and_upload_tmp_yaml,
)
from journeys.evaluation import evaluation_mode_show
from journeys.joins import joins_dialog
from semantic_model_generator.data_processing.cte_utils import (
    context_to_column_format,
    expand_all_logical_tables_as_ctes,
    logical_table_name,
    remove_ltable_cte,
)
from semantic_model_generator.data_processing.proto_utils import (
    proto_to_yaml,
    yaml_to_semantic_model,
)
from semantic_model_generator.protos import semantic_model_pb2
from semantic_model_generator.validate_model import validate

# Set minCachedMessageSize to 500 MB to disable forward message cache:
# st.set_config would trigger an error, only the set_config from config module works
config.set_option("global.minCachedMessageSize", 500 * 1e6)


@st.cache_data(show_spinner=False)
def pretty_print_sql(sql: str) -> str:
    """
    Pretty prints SQL using SQLGlot with an option to use the Snowflake dialect for syntax checks.

    Args:
    sql (str): SQL query string to be formatted.

    Returns:
    str: Formatted SQL string.
    """
    # Parse the SQL using SQLGlot
    expression = sqlglot.parse_one(sql, dialect="snowflake")

    # Generate formatted SQL, specifying the dialect if necessary for specific syntax transformations
    formatted_sql: str = expression.sql(dialect="snowflake", pretty=True)
    return formatted_sql


def process_message(_conn: SnowflakeConnection, prompt: str) -> None:
    """Processes a message and adds the response to the chat."""
    user_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("回答を作成中..."):
            # Depending on whether multiturn is enabled, we either send just the user message or the entire chat history.
            request_messages = (
                st.session_state.messages[1:]  # Skip the welcome message
                if st.session_state.multiturn
                else [user_message]
            )
            try:
                response = send_message(
                    _conn=_conn,
                    semantic_model=proto_to_yaml(st.session_state.semantic_model),
                    messages=request_messages,
                )
                content = response["message"]["content"]
                # Grab the request ID from the response and stash it in the chat message object.
                request_id = response["request_id"]
                display_content(conn=_conn, content=content, request_id=request_id)
                st.session_state.messages.append(
                    {"role": "analyst", "content": content, "request_id": request_id}
                )
            except ValueError as e:
                st.error(e)
                # Remove the user message from the chat history if the request fails.
                # We should only save messages to history for successful (user, assistant) turns.
                st.session_state.messages.pop()


def show_expr_for_ref(message_index: int) -> None:
    """Display the column name and expression as a dataframe, to help user write VQR against logical table/columns."""
    tbl_names = list(st.session_state.ctx_table_col_expr_dict.keys())
    # add multi-select on tbl_name
    tbl_options = tbl_names
    selected_tbl = st.selectbox(
        "Select table for the SQL", tbl_options, key=f"table_options_{message_index}"
    )
    if selected_tbl is not None:
        col_dict = st.session_state.ctx_table_col_expr_dict[selected_tbl]
        col_df = pd.DataFrame(
            {"Column Name": k, "Column Expression": v} for k, v in col_dict.items()
        )
        # Workaround for column_width bug in dataframe object within nested dialog
        st.table(col_df.set_index(col_df.columns[1]))


@st.experimental_dialog("Edit", width="large")
def edit_verified_query(
    conn: SnowflakeConnection, sql: str, question: str, message_index: int
) -> None:
    """Allow user to correct generated SQL and add to verfied queries.
    Note: Verified queries needs to be against logical table/column."""

    # When opening the modal, we haven't run the query yet, so set this bit to False.
    st.session_state["error_state"] = None
    st.caption("**CHEAT SHEET**")
    st.markdown(
        "このセクションは、使用可能なカラムと式をチェックするのに便利です。**注意**： SQLで参照するのは `Column Expression`ではなく, `Column Name` を参照ください。"
    )
    show_expr_for_ref(message_index)
    st.markdown("")
    st.divider()

    sql_without_cte = remove_ltable_cte(
        sql, table_names=[t.name for t in st.session_state.semantic_model.tables]
    )
    st.markdown(
        "以下のSQLを編集してください。利用可能なテーブル/カラムについては、上記の**CheetSheet**の`カラム名`カラムを必ず使用してください。"
    )

    with st.container(border=False):
        st.caption("**SQL**")
        with st.container(border=True):
            css_yaml_editor = """
                textarea{
                    font-size: 14px;
                    color: #2e2e2e;
                    font-family:Menlo;
                    background-color: #fbfbfb;
                }
                """
            # Style text_area to mirror st.code
            with stylable_container(
                key="customized_text_area", css_styles=css_yaml_editor
            ):
                user_updated_sql = st.text_area(
                    label="sql_editor",
                    label_visibility="collapsed",
                    value=sql_without_cte,
                )
            run = st.button("実行", use_container_width=True)

            if run:
                try:
                    sql_to_execute = expand_all_logical_tables_as_ctes(
                        user_updated_sql, st.session_state.ctx
                    )

                    connection = get_snowflake_connection()
                    st.session_state["successful_sql"] = False
                    df = pd.read_sql(sql_to_execute, connection)
                    st.code(user_updated_sql)
                    st.caption("**Output data**")
                    st.dataframe(df)
                    st.session_state["successful_sql"] = True

                except Exception as e:
                    st.session_state["error_state"] = (
                        f"Edited SQL not compatible with semantic model provided, please double check: {e}"
                    )

            if st.session_state["error_state"] is not None:
                st.error(st.session_state["error_state"])

            elif st.session_state.get("successful_sql", False):
                # Moved outside the `if run:` block to ensure it's always evaluated
                mark_as_onboarding = st.checkbox(
                    "オンボードの質問としてマーク",
                    key=f"edit_onboarding_idx_{message_index}",
                    help="Mark this question as an onboarding verified query.",
                )
                save = st.button(
                    "検証済クエリとして保存",
                    use_container_width=True,
                    disabled=not st.session_state.get("successful_sql", False),
                )
                if save:
                    sql_no_analyst_comment = user_updated_sql.replace(
                        " /* Generated by Cortex Analyst */", ""
                    )
                    add_verified_query(
                        question,
                        sql_no_analyst_comment,
                        is_onboarding_question=mark_as_onboarding,
                    )
                    st.session_state["editing"] = False
                    st.session_state["confirmed_edits"] = True


def add_verified_query(
    question: str, sql: str, is_onboarding_question: bool = False
) -> None:
    """Save verified question and SQL into an in-memory list with additional details."""
    # Verified queries follow the Snowflake definitions.
    verified_query = semantic_model_pb2.VerifiedQuery(
        name=question,
        question=question,
        sql=sql,
        verified_by=st.session_state["user_name"],
        verified_at=int(time.time()),
        use_as_onboarding_question=is_onboarding_question,
    )
    st.session_state.semantic_model.verified_queries.append(verified_query)
    st.success(
        "検証済クエリが追加されました！YAML を再度検証してアップロードできます; もしくは検証済みのクエリを追加し続けることもできます。"
    )
    st.rerun()


def display_content(
    conn: SnowflakeConnection,
    content: List[Dict[str, Any]],
    request_id: Optional[str],
    message_index: Optional[int] = None,
) -> None:
    """Displays a content item for a message. For generated SQL, allow user to add to verified queries directly or edit then add."""
    message_index = message_index or len(st.session_state.messages)
    question = ""
    for item in content:
        if item["type"] == "text":
            if question == "" and "__" in item["text"]:
                question = item["text"].split("__")[1]
            # If API rejects to answer directly and provided disambiguate suggestions, we'll return text with <SUGGESTION> as prefix.
            if "<SUGGESTION>" in item["text"]:
                suggestion_response = json.loads(item["text"][12:])[0]
                st.markdown(Translate(suggestion_response["explanation"],"en","ja"))
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(
                        suggestion_response["suggestions"]
                    ):
                        if st.button(
                            Translate(suggestion,"en","ja"), key=f"{message_index}_{suggestion_index}"
                        ):
                            st.session_state.active_suggestion = suggestion
            else:
                st.markdown(Translate(item["text"],"en","ja"))
        elif item["type"] == "suggestions":
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(Translate(suggestion,"en","ja"), key=f"{message_index}_{suggestion_index}"):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            with st.container(height=500, border=False):
                sql = item["statement"]
                sql = pretty_print_sql(sql)
                with st.container(height=250, border=False):
                    st.code(item["statement"], language="sql")

                df = pd.read_sql(sql, conn)
                st.dataframe(df, hide_index=True)

                mark_as_onboarding = st.checkbox(
                    "オンボード質問としてマーク",
                    key=f"onboarding_idx_{message_index}",
                    help="Mark this question as an onboarding verified query.",
                )
                left, right = st.columns(2)
                if right.button(
                    "検証済クエリとして保存",
                    key=f"save_idx_{message_index}",
                    use_container_width=True,
                ):
                    sql_no_cte = remove_ltable_cte(
                        sql,
                        table_names=[
                            t.name for t in st.session_state.semantic_model.tables
                        ],
                    )
                    cleaned_sql = sql_no_cte.replace(
                        " /* Generated by Cortex Analyst */", ""
                    )
                    add_verified_query(
                        question, cleaned_sql, is_onboarding_question=mark_as_onboarding
                    )

                if left.button(
                    "編集",
                    key=f"edits_idx_{message_index}",
                    use_container_width=True,
                ):
                    edit_verified_query(conn, sql, question, message_index)

    # If debug mode is enabled, we render the request ID. Note that request IDs are currently only plumbed
    # through for assistant messages, as we obtain the request ID as part of the Analyst response.
    if request_id and st.session_state.chat_debug:
        st.caption(f"Request ID: {request_id}")


def chat_and_edit_vqr(_conn: SnowflakeConnection) -> None:
    messages = st.container(height=600, border=False)

    # Convert semantic model to column format to be backward compatible with some old utils.
    if "semantic_model" in st.session_state:
        st.session_state.ctx = context_to_column_format(st.session_state.semantic_model)
        ctx_table_col_expr_dict = {
            logical_table_name(t): {c.name: c.expr for c in t.columns}
            for t in st.session_state.ctx.tables
        }

        st.session_state.ctx_table_col_expr_dict = ctx_table_col_expr_dict

    FIRST_MESSAGE = "ようこそ！😊 このアプリでは、左側でセマンティックモデルのYAMLを繰り返し編集し、右側のチャット環境でテストすることができます。何かご質問はありますか？"

    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = [
            {
                "role": "analyst",
                "content": [
                    {
                        "type": "text",
                        "text": FIRST_MESSAGE,
                    }
                ],
            }
        ]

    for message_index, message in enumerate(st.session_state.messages):
        with messages:
            # To get the handy robot icon on assistant messages, the role needs to be "assistant" or "ai".
            # However, the Analyst API uses "analyst" as the role, so we need to convert it at render time.
            render_role = "assistant" if message["role"] == "analyst" else "user"
            with st.chat_message(render_role):
                display_content(
                    conn=_conn,
                    content=message["content"],
                    message_index=message_index,
                    request_id=message.get(
                        "request_id"
                    ),  # Safe get since user messages have no request IDs
                )

    chat_placeholder = (
        "質問を入力してください"
        if st.session_state["validated"]
        else "質問を入力する前に, セマンティックモデルを有効化してください"
    )
    if user_input := st.chat_input(
        chat_placeholder, disabled=not st.session_state["validated"]
    ):
        with messages:
            process_message(_conn=_conn, prompt=user_input)

    if st.session_state.active_suggestion:
        with messages:
            process_message(_conn=_conn, prompt=st.session_state.active_suggestion)
        st.session_state.active_suggestion = None


@st.experimental_dialog("Upload", width="small")
def upload_dialog(content: str) -> None:
    def upload_handler(file_name: str) -> None:
        if not st.session_state.validated and changed_from_last_validated_model():
            with st.spinner(
                "あなたのセマンティックモデルは前回の検証以降に変更されました。アップロードする前に再度検証してください。"
            ):
                validate_and_upload_tmp_yaml(conn=get_snowflake_connection())

        st.session_state.semantic_model = yaml_to_semantic_model(content)
        with st.spinner(
            f"@{st.session_state.snowflake_stage.stage_name}/{file_name}.yamlをアップロード中..."
        ):
            upload_yaml(file_name)
        st.success(
            f"@{st.session_state.snowflake_stage.stage_name}/{file_name}.yamlのアップロードが完了しました!"
        )
        st.session_state.last_saved_yaml = content
        time.sleep(1.5)
        st.rerun()

    if "snowflake_stage" in st.session_state:
        # When opening the iteration app directly, we collect stage information already when downloading the YAML.
        # We only need to ask for the new file name in this case.
        with st.form("upload_form_name_only"):
            st.markdown("YAMLを次のSnowflakeステージにアップロードします")
            st.write(st.session_state.snowflake_stage.to_dict())
            new_name = st.text_input(
                key="upload_yaml_final_name",
                label="アップロードするファイル名を入力 (omit .yaml suffix):",
            )

            if st.form_submit_button("アップロード"):
                upload_handler(new_name)
    else:
        # If coming from the builder flow, we need to ask the user for the exact stage path to upload to.
        st.markdown("YAMLファイルの保存先を入力してください")
        stage_selector_container()
        new_name = st.text_input("ファイル名 (omit .yaml suffix)", value="")

        if st.button("アップロード"):
            if (
                not st.session_state["selected_iteration_database"]
                or not st.session_state["selected_iteration_schema"]
                or not st.session_state["selected_iteration_stage"]
                or not new_name
            ):
                st.error("全ての入力欄を埋めてください")
                return

            st.session_state["snowflake_stage"] = SnowflakeStage(
                stage_database=st.session_state["selected_iteration_database"],
                stage_schema=st.session_state["selected_iteration_schema"],
                stage_name=st.session_state["selected_iteration_stage"],
            )
            upload_handler(new_name)


def update_container(
    container: DeltaGenerator, content: str, prefix: Optional[str]
) -> None:
    """
    Update the given Streamlit container with the provided content.

    Args:
        container (DeltaGenerator): The Streamlit container to update.
        content (str): The content to be displayed in the container.
        prefix (str): The prefix to be added to the content.
    """

    # Clear container
    container.empty()

    if content == "success":
        content = "  ·  :green[✅  モデルの更新と検証]"
    elif content == "editing":
        content = "  ·  :gray[✏️  編集中...]"
    elif content == "failed":
        content = "  ·  :red[❌  有効化失敗しました. エラーを修正してください]"

    if prefix:
        content = prefix + content

    container.markdown(content)


@st.experimental_dialog("Error", width="small")
def exception_as_dialog(e: Exception) -> None:
    st.error(f"An error occurred: {e}")


# TODO: how to properly mark fragment back?
# @st.experimental_fragment
def yaml_editor(yaml_str: str) -> None:
    """
    Editor for YAML content. Meant to be used on the left side
    of the app.

    Args:
        yaml_str (str): YAML content to be edited.
    """
    css_yaml_editor = """
    textarea{
        font-size: 14px;
        color: #2e2e2e;
        font-family:Menlo;
        background-color: #fbfbfb;
    }
    """

    # Style text_area to mirror st.code
    with stylable_container(key="customized_text_area", css_styles=css_yaml_editor):
        content = st.text_area(
            label="yaml_editor",
            label_visibility="collapsed",
            value=yaml_str,
            height=600,
        )
    st.session_state.working_yml = content
    status_container_title = "**編集**"
    status_container = st.empty()

    def validate_and_update_session_state() -> None:
        # Validate new content
        try:
            validate(
                content,
                conn=get_snowflake_connection(),
            )
            st.session_state["validated"] = True
            update_container(status_container, "success", prefix=status_container_title)
            st.session_state.semantic_model = yaml_to_semantic_model(content)
            st.session_state.last_saved_yaml = content
        except Exception as e:
            st.session_state["validated"] = False
            update_container(status_container, "failed", prefix=status_container_title)
            exception_as_dialog(e)

    button_row = row(5)
    if button_row.button("検証", use_container_width=True, help=VALIDATE_HELP):
        # Validate new content
        validate_and_update_session_state()

        # Rerun the app if validation was successful.
        # We shouldn't rerun if validation failed as the error popup would immediately dismiss.
        # This must be done outside of the try/except because the generic Exception handling is catching the
        # exception that st.rerun() properly raises to halt execution.
        # This is fixed in later versions of Streamlit, but other refactors to the code are required to upgrade.
        if st.session_state["validated"]:
            st.rerun()

    if content:
        button_row.download_button(
            label="ダウンロード",
            data=content,
            file_name="semantic_model.yaml",
            mime="text/yaml",
            use_container_width=True,
            help=UPLOAD_HELP,
        )

    if button_row.button(
        "アップロード",
        use_container_width=True,
        help=UPLOAD_HELP,
    ):
        upload_dialog(content)
    if st.session_state.get("partner_setup", False):
        from partner.partner_utils import integrate_partner_semantics

        if button_row.button(
            "Integrate Partner",
            use_container_width=True,
            help=PARTNER_SEMANTIC_HELP,
            disabled=not st.session_state["validated"],
        ):
            integrate_partner_semantics()

    if st.session_state.experimental_features:
        # Preserve a session state variable that maintains whether the join dialog is open.
        # This is necessary because the join dialog calls `st.rerun()` from within, which closes the modal
        # unless its state is being tracked.
        if "join_dialog_open" not in st.session_state:
            st.session_state["join_dialog_open"] = False

        if button_row.button(
            "Join編集",
            use_container_width=True,
        ):
            with st.spinner("モデルを検証しています..."):
                validate_and_update_session_state()
            st.session_state["join_dialog_open"] = True

        if st.session_state["join_dialog_open"]:
            joins_dialog()

    # Render the validation state (success=True, failed=False, editing=None) in the editor.
    if st.session_state.validated:
        update_container(status_container, "success", prefix=status_container_title)
    elif st.session_state.validated is not None and not st.session_state.validated:
        update_container(status_container, "failed", prefix=status_container_title)
    else:
        update_container(status_container, "editing", prefix=status_container_title)


@st.experimental_dialog("アプリ「ITERATION（反復）」へようこそ！💬", width="large")
def set_up_requirements() -> None:
    """
    Collects existing YAML location from the user so that we can download it.
    """
    st.markdown(
        "Snowflakeステージの情報を入力して、既存のYAMLファイルをダウンロードします。"
    )

    stage_selector_container()

    # Based on the currently selected stage, show a dropdown of YAML files for the user to pick from.
    available_files = []
    if (
        "selected_iteration_stage" in st.session_state
        and st.session_state["selected_iteration_stage"]
    ):
        # When a valid stage is selected, fetch the available YAML files in that stage.
        try:
            available_files = get_yamls_from_stage(
                st.session_state["selected_iteration_stage"]
            )
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected stage.")
            st.stop()

    file_name = st.selectbox("ファイル名", options=available_files, index=None)

    experimental_features = st.checkbox(
        "結合を許可(オプショナル)",
        help="このボックスにチェックを入れると、セマンティックモデル内の結合パスを追加/編集できるようになります。この設定を有効にする場合は、Snowflakeアカウントに適切なパラメータが設定されていることを確認してください。アクセス方法については、アカウントチームにお問い合わせください。",
    )

    if st.button(
        "送信",
        disabled=not st.session_state["selected_iteration_database"]
        or not st.session_state["selected_iteration_schema"]
        or not st.session_state["selected_iteration_stage"]
        or not file_name,
    ):
        st.session_state["snowflake_stage"] = SnowflakeStage(
            stage_database=st.session_state["selected_iteration_database"],
            stage_schema=st.session_state["selected_iteration_schema"],
            stage_name=st.session_state["selected_iteration_stage"],
        )
        st.session_state["file_name"] = file_name
        st.session_state["page"] = GeneratorAppScreen.ITERATION
        st.session_state["experimental_features"] = experimental_features
        st.rerun()


@st.experimental_dialog("チャット設定", width="small")
def chat_settings_dialog() -> None:
    """
    Dialog that allows user to toggle on/off certain settings about the chat experience.
    """

    debug = st.toggle(
        "デバッグモード",
        value=st.session_state.chat_debug,
        help="デバッグモードを有効にして追加情報を確認する (e.g. request ID).",
    )

    multiturn = st.toggle(
        "マルチターン",
        value=st.session_state.multiturn,
        help="マルチターンモードを有効にして、チャットがコンテキストを記憶できるようにします。この機能を使用するには、アカウントで正しいパラメータが有効になっている必要があります。",
    )

    if st.button("保存"):
        st.session_state.chat_debug = debug
        st.session_state.multiturn = multiturn
        st.rerun()


VALIDATE_HELP = """このアプリでアクティブなセマンティックモデルへの変更を保存し、検証します。
これは役に立つので、右側にあるチャットパネルで操作することができます。"""

DOWNLOAD_HELP = (
    """現在ロードされているセマンティックモデルをローカルマシンにダウンロードします。"""
)

UPLOAD_HELP = """YAMLをSnowflakeステージにアップロードする。セマンティックモデルがうまくいっていて、
prodにプッシュするべきだと思うときはいつでもアップロードします！
セマンティックモデルをアップロードするには検証が必要であることに注意してください。."""

PARTNER_SEMANTIC_HELP = """パートナーツールからセマンティックファイルをアップロードしましたか？
この機能を使用して、パートナーのセマンティックスペックをCortex Analystのスペックに統合します。
パートナーのセマンティクスを統合する前に、Cortex Analystのセマンティックモデルを検証する必要があることに注意してください。"""


def show() -> None:
    init_session_states()

    if "snowflake_stage" not in st.session_state and "yaml" not in st.session_state:
        # If the user is jumping straight into the iteration flow and not coming from the builder flow,
        # we need to collect credentials and load YAML from stage.
        # If coming from the builder flow, there's no need to collect this information until the user wants to upload.
        set_up_requirements()
    else:
        home, mode = st.columns(2)
        with home:
            return_home_button()
        with mode:
            st.session_state["app_mode"] = st.selectbox(
                label="App Mode",
                label_visibility="collapsed",
                options=["チャット", "評価", "YAMLプレビュー"],
            )
        if "yaml" not in st.session_state:
            # Only proceed to download the YAML from stage if we don't have one from the builder flow.
            yaml = download_yaml(
                st.session_state.file_name, st.session_state.snowflake_stage.stage_name
            )
            st.session_state["yaml"] = yaml
            st.session_state["semantic_model"] = yaml_to_semantic_model(yaml)
            if "last_saved_yaml" not in st.session_state:
                st.session_state["last_saved_yaml"] = yaml

        left, right = st.columns(2)
        yaml_container = left.container(height=760)
        chat_container = right.container(height=760)

        with yaml_container:
            # Attempt to use the semantic model stored in the session state.
            # If there is not one present (e.g. they are coming from the builder flow and haven't filled out the
            # placeholders yet), we should still let them edit, so use the raw YAML.
            if st.session_state.semantic_model.name != "":
                editor_contents = proto_to_yaml(st.session_state["semantic_model"])
            else:
                editor_contents = st.session_state["yaml"]

            yaml_editor(editor_contents)

        with chat_container:
            app_mode = st.session_state["app_mode"]
            if app_mode == "YAMLプレビュー":
                st.code(
                    st.session_state.working_yml, language="yaml", line_numbers=True
                )
            elif app_mode == "評価":
                evaluation_mode_show()
            elif app_mode == "チャット":
                if st.button("設定"):
                    chat_settings_dialog()
                # We still initialize an empty connector and pass it down in order to propagate the connector auth token.
                chat_and_edit_vqr(get_snowflake_connection())
            else:
                st.error(f"Unknown App Mode: {app_mode}")
