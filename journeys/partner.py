import streamlit as st


@st.experimental_dialog("Partner Semantic Support", width="large")
def partner_semantic_setup() -> None:
    """
    Renders the partner semantic setup dialog with instructions.
    """
    from partner.partner_utils import configure_partner_semantic

    st.write(
        """
        Snowflakeと統合されているパートナーツールに既存のセマンティックレイヤーをお持ちですか？
パートナーのセマンティックスペックをCortex Analystのセマンティックファイルに統合するには、以下の手順を参照してください。"""
    )
    configure_partner_semantic()


def show() -> None:
    """
    Runs partner setup dialog.
    """
    partner_semantic_setup()
