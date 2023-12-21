import streamlit as st

# SET FONT STYLES
font_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Ysabeau+SC:wght@200&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Rubik+Dirt&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Rock+Salt&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Josefin+Slab:wght@200&display=swap');

            /* Set the font family for paragraph elements */
            p {
               font-family: 'Ysabeau SC', sans-serif;
            }

            /* Set the font family, size, and weight for unordered list and ordered list elements */
            ul, ol {
                font-family: 'Ysabeau SC', sans-serif;
                font-size: 16px;
                font-weight: normal;
            }

            /* Set the font family, size, weight, and margin for list item elements */
            li {
                font-family: 'Ysabeau SC', sans-serif;
                font-size: 16px;
                font-weight: normal;
                margin-top: 5px;
            }
            </style>
            """

############################
# FORMATTING TEXT FUNCTIONS
############################
def my_title(my_string, my_background_color="#my_title", gradient_colors=None):
    if gradient_colors is None:
        gradient_colors = f"{my_background_color}, #2CB8A1, #0072B2"
    gradient = f"-webkit-linear-gradient(45deg, {gradient_colors})"
    st.markdown(
        f'<h3 style="background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient}; padding:10px; border-radius: 10px; border: 1px solid {my_background_color}; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25);"> <center> {my_string} </center> </h3>',
        unsafe_allow_html=True)


def my_header(my_string, my_style="#217CD0"):
    st.markdown(f'<h2 style="color:{my_style};"> <center> {my_string} </center> </h2>', unsafe_allow_html=True)


def my_subheader(my_string, my_background_colors=None, my_style="#FFFFFF", my_size=3):
    if my_background_colors is None:
        my_background_colors = ["#45B8AC", "#2CB8A1", "#0072B2"]

    gradient = f"-webkit-linear-gradient(45deg, {', '.join(my_background_colors)})"
    st.markdown(
        f'<h{my_size} style="background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient}; font-family: sans-serif; font-weight: bold; text-align: center; color: {my_style};"> {my_string} </h{my_size}>',
        unsafe_allow_html=True)

def my_text_header(my_string,
                   my_text_align='center',
                   my_font_family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                   my_font_weight=200,
                   my_font_size='36px',
                   my_line_height=1.5):
    text_header = f'<h1 style="text-align:{my_text_align}; font-family: {my_font_family}; font-weight: {my_font_weight}; font-size: {my_font_size}; line-height: {my_line_height};">{my_string}</h1>'
    st.markdown(text_header, unsafe_allow_html=True)


def my_text_paragraph(my_string,
                      my_text_align='center',
                      my_font_family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                      my_font_weight=200,
                      my_font_size='18px',
                      my_line_height=1.5,
                      add_border=False,
                      border_color="#45B8AC"):
    if add_border:
        border_style = f'border: 2px solid {border_color}; border-radius: 10px; padding: 10px; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25);'
    else:
        border_style = ''
    paragraph = f'<p style="text-align:{my_text_align}; font-family:{my_font_family}; font-weight:{my_font_weight}; font-size:{my_font_size}; line-height:{my_line_height}; background-color: rgba(255, 255, 255, 0); {border_style}">{my_string}</p>'
    st.markdown(paragraph, unsafe_allow_html=True)


def vertical_spacer(n):
    for i in range(n):
        st.write("")


def my_bubbles(my_string, my_background_color="#2CB8A1"):
    gradient = f"-webkit-linear-gradient(45deg, {my_background_color}, #2CB8A1, #0072B2)"
    text_style = f"text-align: center; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; font-weight: 200; font-size: 36px; line-height: 1.5; -webkit-background-clip: text; -webkit-text-fill-color: black; padding: 20px; position: relative;"
    st.markdown(f'''
        <div style="position: relative;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: -1; opacity: 0.2;"></div>
            <h1 style="{text_style}">
                <center>{my_string}</center>
                <div style="position: absolute; top: 20px; left: 80px;">
                    <div style="background-color: #0072B2; width: 8px; height: 8px; border-radius: 50%; animation: bubble 3s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; right: 100px;">
                    <div style="background-color: #FF0000; width: 14px; height: 14px; border-radius: 50%; animation: bubble 4s infinite;"></div>
                </div>
                <div style="position: absolute; top: 10px; right: 50px;">
                    <div style="background-color: #0072B2; width: 8px; height: 8px; border-radius: 50%; animation: bubble 5s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 60px;">
                    <div style="background-color: #88466D; width: 8px; height: 8px; border-radius: 50%; animation: bubble 6s infinite;"></div>
                </div>
                <div style="position: absolute; top: 0px; left: -10px;">
                    <div style="background-color: #2CB8A1; width: 12px; height: 12px; border-radius: 50%; animation: bubble 7s infinite;"></div>
                </div>
                <div style="position: absolute; top: 10px; right: -20px;">
                    <div style="background-color: #7B52AB; width: 10px; height: 10px; border-radius: 50%; animation: bubble 10s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 150px;">
                    <div style="background-color: #FF9F00; width: 8px; height: 8px; border-radius: 50%; animation: bubble 20s infinite;"></div>
                </div>
                <div style="position: absolute; top: 25px; right: 170px;">
                    <div style="background-color: #FF6F61; width: 12px; height: 12px; border-radius: 50%; animation: bubble 4s infinite;"></div>
                </div>
                <div style="position: absolute; top: -30px; right: 120px;">
                <div style="background-color: #440154; width: 10px; height: 10px; border-radius: 50%; animation: bubble 5s infinite;"></div>
                </div>
                <div style="position: absolute; top: -20px; left: 150px;">
                <div style="background-color: #2CB8A1; width: 8px; height: 8px; border-radius: 50%; animation: bubble 6s infinite;"></div>
                </div>
                <div style="position: absolute; top: -10px; right: 20px;">
                <div style="background-color: #FFC300; width: 12px; height: 12px; border-radius: 50%; animation: bubble 7s infinite;"></div>
                </div>
                </h1>
                <style>
                @keyframes bubble {{
                0% {{
                transform: translateY(0);
                }}
                50% {{
                transform: translateY(+50px);
                }}
                100% {{
                transform: translateY(0);
                }}
                }}
                .bubble-container div {{
                margin: 10px;
                }}
                </style>
                </div>
                ''', unsafe_allow_html=True)