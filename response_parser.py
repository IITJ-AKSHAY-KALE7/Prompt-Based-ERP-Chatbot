import streamlit as st

class StreamlitResponse:
    def __init__(self, context) -> None:
        self.context = context

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return
