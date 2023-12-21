import streamlit as st
from style.text import vertical_spacer


class SocialMediaLinks:

    MARGIN_BEFORE = 20

    GITHUB_PROFILE = "https://github.com/tonyhollaar/ForecastGenie"
    MEDIUM_PROFILE = "https://medium.com/@thollaar"
    LOGO_PROFILE = "https://tonyhollaar.com"
    TWITTER_PROFILE = "https://twitter.com/tonyhollaar"
    BUYMEACOFFEE_PROFILE = "https://www.buymeacoffee.com/tonyhollaar"

    ICON_URLS = {
        'GitHub': 'https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/github-mark.png',
        'Medium': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Medium_logo_Monogram.svg/512px'
                  '-Medium_logo_Monogram.svg.png',
        'Logo': 'https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/logo_website.png',
        'Twitter': 'https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/twitter_logo_black.png',
        'BuyMeACoffee': 'https://raw.githubusercontent.com/tonyhollaar/ForecastGenie/main/images/buymeacoffee_logo.png'
    }

    def __init__(self):
        self.margin_before = self.MARGIN_BEFORE
        self.links = {
            'GitHub': self.GITHUB_PROFILE,
            'Medium': self.MEDIUM_PROFILE,
            'Logo': self.LOGO_PROFILE,
            'Twitter': self.TWITTER_PROFILE,
            'BuyMeACoffee': self.BUYMEACOFFEE_PROFILE
        }

    def render(self):
        vertical_spacer(self.margin_before)
        st.divider()
        num_icons = len(self.links)
        columns = self.generate_columns(num_icons)

        cols = st.columns(columns)

        for index, (name, url) in enumerate(self.links.items()):
            col_idx = index * 2
            code = f'<a href="{url}"><img src="{self.ICON_URLS[name]}" alt="{name}" width="32"></a>'
            cols[col_idx].markdown(code, unsafe_allow_html=True)

    @staticmethod
    def generate_columns(num_icons):
        columns = [2]  # Start with the first column as 2
        for _ in range(num_icons - 1):
            columns.extend([1, 2])  # Alternating 1 and 2 for the remaining columns
        return columns