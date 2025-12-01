from pathlib import Path
import sys

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

import streamlit as st  # noqa
from dashboard.components.page_configurator import PageConfigurator # noqa
from dashboard.translations import TEXT_TRANSLATIONS as tt # noqa

config = PageConfigurator().configure()

if __name__ == "__main__":
    language = config["language"]

    st.markdown(
        f"<h1 style='text-align: center;'>{tt[language]['welcome_to']}</h1>",
        unsafe_allow_html=True,
    )

    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.image("img/logo_title.png", width=3600)

    st.markdown(
        f"""
        <p style='text-align: center; font-size: 20px;'>
        {tt[language]['app_description']}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")
    cols = st.columns([1, 1, 1])
    with cols[1]:
        if st.button(f"{tt[language]['start']}", width="stretch"):
            st.switch_page("pages/artguide.py")
