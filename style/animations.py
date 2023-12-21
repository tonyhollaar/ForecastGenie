import streamlit as st
import json
from streamlit_lottie import st_lottie
from style.text import vertical_spacer


def show_lottie_animation(url, key, reverse=False, height=400, width=400, speed=1, loop=True, quality='high',
                          col_sizes=[1, 3, 1], margin_before=0, margin_after=0):
    with open(url, "r") as file:
        animation_url = json.load(file)

    col1, col2, col3 = st.columns(col_sizes)
    with col2:
        vertical_spacer(margin_before)

        st_lottie(animation_url,
                  reverse=reverse,
                  height=height,
                  width=width,
                  speed=speed,
                  loop=loop,
                  quality=quality,
                  key=key
                  )
        vertical_spacer(margin_after)