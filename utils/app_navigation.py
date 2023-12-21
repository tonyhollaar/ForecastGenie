# app_navigation.py
import streamlit as st
from streamlit_option_menu import option_menu


class AppNavigation:
    def __init__(self):
        pass

    @staticmethod
    def create_main_menu():
        # Main menu creation
        menu_item = option_menu(menu_title=None,
                                key="main_menu",  # Add a unique key here
                                options=["Load", "Explore", "Clean", "Engineer", "Prepare", "Select", "Train", "Evaluate",
                                         "Tune", "Forecast", "Review"],
                                icons=['cloud-arrow-up', 'search', 'bi-bezier2', 'gear', 'shuffle', 'bi-sort-down',
                                       "cpu", 'clipboard-check', 'sliders', 'graph-up-arrow', 'bullseye'],
                                menu_icon="cast",
                                default_index=0,
                                orientation="horizontal",
                                styles={
                                    "container": {
                                        "padding": "0.5px",
                                        "background-color": "transparent",
                                        "border-radius": "15px",
                                        "border": "1px solid grey",
                                        "margin": "0px 0px 0px 0px",
                                    },
                                    "icon": {
                                        "color": "#333333",
                                        "font-size": "20px",
                                    },
                                    "nav-link": {
                                        "font-size": "12px",
                                        "font-family": 'Ysabeau SC',
                                        "color": "F5F5F5",
                                        "text-align": "center",
                                        "margin": "0px",
                                        "padding": "8px",
                                        "background-color": "transparent",
                                        "opacity": "1",
                                        "transition": "background-color 0.3s ease-in-out",
                                    },
                                    "nav-link:hover": {
                                        "background-color": "#4715ef",
                                    },
                                    "nav-link-selected": {
                                        "background-color": "transparent",
                                        "color": "black",
                                        "opacity": "0.1",
                                        "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.0)",
                                        "border-radius": "0px",
                                    },
                                }
                                )
        return menu_item

    @staticmethod
    def create_sidebar_menu():
        # Create sidebar menu
        with st.sidebar:
            sidebar_menu_item = option_menu(
                key="sidebar_menu",  # Add a unique key here
                menu_title=None,
                options=["HOME", "ABOUT", "FAQ", "DOC"],
                icons=["house", "file-person", "info-circle", "file-text"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {
                        "padding": "0px",
                        "background": "rgba(255, 255, 255, 0.2)",
                        # ... (other styles)
                    },
                    "icon": {
                        "color": "#CCCCCC",
                        "font-size": "15px",
                        "margin-right": "5px"
                    },
                    "nav-link": {
                        "font-size": "12px",
                        "color": "F5F5F5",
                        "text-align": "center",
                        "margin": "0px",
                        "padding": "8px",
                        "background-color": "transparent",
                        "opacity": "1",
                        "transition": "background-color 0.3s ease-in-out",
                        "font-family": "'Rock Salt', sans-serif !important",
                        # Use '!important' to override existing styles
                    },
                    "nav-link:hover": {
                        "background-color": "rgba(255, 255, 255, 0.1)",
                        # ... (other styles)
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(45deg, #000000, ##f5f5f5)",
                        "background": "-webkit-linear-gradient(45deg, #000000, ##f5f5f5)",
                        "background": "-moz-linear-gradient(45deg, #000000, ##f5f5f5)",
                        "color": "black",
                        "opacity": "1",
                        "font-family": "'Rock Salt', sans-serif !important",
                        # Use '!important' to override existing styles
                        # ... (other styles)
                    }
                }
            )
        return sidebar_menu_item
