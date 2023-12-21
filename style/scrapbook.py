import streamlit as st

# =============================================================================
#    _____  _____ _____            _____  ____   ____   ____  _  __
#   / ____|/ ____|  __ \     /\   |  __ \|  _ \ / __ \ / __ \| |/ /
#  | (___ | |    | |__) |   /  \  | |__) | |_) | |  | | |  | | ' /
#   \___ \| |    |  _  /   / /\ \ |  ___/|  _ <| |  | | |  | |  <
#   ____) | |____| | \ \  / ____ \| |    | |_) | |__| | |__| | . \
#  |_____/ \_____|_|  \_\/_/    \_\_|    |____/ \____/ \____/|_|\_\
#
# =============================================================================

# TEST
# =============================================================================
# def create_flipcards_model_cards(num_cards, header_list, paragraph_list_front, paragraph_list_back, font_family, font_size_front, font_size_back):
#     # note removing display: flex; inside the css code for .flashcard -> puts cards below eachother
#     # create empty list that will keep the html code needed for each card with header+text
#     card_html = []
#     # iterate over cards specified by user and join the headers and text of the lists
#     for i in range(num_cards):
#         card_html.append(f"""<div class="flashcard">
#                                 <div class='front'>
#                                     <h1 style='text-align:center;'>{header_list[i]}</h1>
#                                     <p style='text-align:center; font-size: {font_size_front};'>{paragraph_list_front[i]}</p>
#                                 </div>
#                                 <div class="back">
#                                     <p style='text-align:justify; word-spacing: 1px; font-size: {font_size_back}; margin-right: 60px; margin-left: 60px'>{paragraph_list_back[i]}</p>
#                                 </div>
#                             </div>
#                             """)
#     # join all the html code for each card and join it into single html code with carousel wrapper
#     carousel_html = "<div class='carousel'>" + "".join(card_html) + "</div>"
#
#     # Display the carousel in streamlit
#     st.markdown(carousel_html, unsafe_allow_html=True)
#
#     # Create the CSS styling for the carousel
#     st.markdown(
#         f"""
#         <style>
#         /* Carousel Styling */
#         .carousel {{
#           grid-gap: 0px; /* Reduce the gap between cards */
#           justify-content: center; /* Center horizontally */
#           align-items: center; /* Center vertically */
#           overflow-x: auto;
#           scroll-snap-type: x mandatory;
#           scroll-behavior: smooth;
#           -webkit-overflow-scrolling: touch;
#           width: 500px;
#           margin: 0 auto; /* Center horizontally by setting left and right margins to auto */
#           background-color: transparent; /* Remove the background color */
#           padding: 0px; /* Remove padding */
#           border-radius: 0px; /* Add border-radius for rounded edges */
#         }}
#         .flashcard {{
#           display: inline-block; /* Display cards inline */
#           width: 500px;
#           height: 200px;
#           background-color: transparent; /* Remove the background color */
#           border-radius: 0px;
#           border: 2px solid black; /* Add black border */
#           perspective: 0px;
#           margin-bottom: 0px; /* Remove space between cards */
#           padding: 0px;
#           scroll-snap-align: center;
#         }}
#         .front, .back {{
#           position: absolute;
#           top: 0;
#           left: 0;
#           width: 500px;
#           height: 200px;
#           border-radius: 0px;
#           backface-visibility: hidden;
#           font-family: 'Ysabeau SC', sans-serif; /* font-family: font-family: 'Ysabeau SC', sans-serif; */
#           text-align: center;
#           margin: 0px;
#           background: white;
#         }}
#         .back {{
#             /* ... other styles ... */
#             background: none; /* Remove the background */
#             color: transparent; /* Make the text transparent */
#             background-clip: text; /* Apply the background gradient to the text */
#             -webkit-background-clip: text; /* For Safari */
#             -webkit-text-fill-color: transparent; /* For Safari */
#             background-image: linear-gradient(to bottom left, #941c8e, #763a9a, #4e62a3, #2e81ad, #12a9b4); /* Set the background gradient */
#             transform: rotateY(180deg);
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             flex-direction: column;
#         }}
#         .flashcard:hover .front {{
#           transform: rotateY(180deg);
#         }}
#         .flashcard:hover .back {{
#           transform: rotateY(0deg);
#           cursor: default; /* Change cursor to pointer on hover */
#         }}
#         .front h1, .back p {{
#           color: black;
#           text-align: center;
#           margin: 0;
#           font-family: {font_family};
#           font-size: {font_size_front}px;
#         }}
#         .back p {{
#           line-height: 1.5;
#           margin: 0;
#         }}
#         /* Carousel Navigation Styling */
#         .carousel-nav {{
#           margin: 0px 0px;
#           text-align: center;
#         }}
#         </style>
#         """, unsafe_allow_html=True)
# =============================================================================

# =============================================================================
# def create_moon_clickable_btns(button_names = ['MY_BUTTON_TEXT']):
#     """
#     Generates HTML code for creating clickable moon buttons.
#
#     Args:
#         button_names (list): A list of button names.
#
#     Returns:
#         str: The lowercase button name if a moon button is clicked with added lowercase if spaces in the name
#
#     Example:
#         clicked_btn_name = create_moon_clickable_btns(button_names = [my button name'])
#         if clicked_btn_name == 'my_button_name':
#             print(f'you clicked the button {my_button_name}!')
#
#     Requirements:
#         streamlit
#         from st_click_detector import click_detector #source: https://github.com/vivien000/st-click-detector
#
#     """
#     button_colors = [
#                      'radial-gradient(circle at 50% 50%, #000000)',
#                      'conic-gradient(from 180deg at 50% 50%, #f5f5f5 0deg 180deg, #000000 180deg)',
#                      'radial-gradient(circle at 50% 50%, #000000)',
#                      'conic-gradient(from 180deg at 50% 50%, #000000 0deg 180deg, #f5f5f5 180deg)'
#                     ]
#
#     # MOON BUTTONS
#     button_container = """
#     <style>
#         .button-container {{
#             display: flex;
#             flex-direction: row;
#             align-items: center;
#             justify-content: center;
#             padding: 10px;
#             margin-top: -30px;
#             position: relative;
#             perspective: 1000px;
#             transform-style: preserve-3d;
#         }}
#
#         .button {{
#             width: 70px;
#             height: 70px;
#             background: #292929;
#             border-radius: 50%;
#             margin: 0 10px;
#             position: relative;
#             perspective: 1000px;
#             transform-style: preserve-3d;
#             overflow: hidden;
#         }}
#
#         .button:before {{
#             content: "";
#             position: absolute;
#             top: 50%;
#             left: 50%;
#             width: 120%;
#             height: 120%;
#             border-radius: 50%;
#             background: transparent;
#             border: 2px solid rgba(255, 255, 255, 0.3);
#             transform-style: preserve-3d;
#             transform: translate(-50%, -50%) rotateY(-90deg);
#             animation: planet-ring-rotation 8s linear infinite reverse;
#             z-index: -1;
#         }}
#
#         .button:after {{
#             content: "";
#             position: absolute;
#             top: 50%;
#             left: 50%;
#             width: 120%;
#             height: 2px;
#             background: rgba(255, 255, 255, 0.3);
#             transform-style: preserve-3d;
#             transform: translate(-50%, -50%) rotateX(90deg);
#         }}
#
#         .button a {{
#             position: absolute;
#             top: 0;
#             left: 0;
#             width: 100%;
#             height: 100%;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             text-decoration: 'none';
#             color: #f5f5f5;
#             text-transform: uppercase;
#             font-size: 16px;
#             font-weight: bold;
#             white-space: nowrap;
#             animation: text-rotation 8s linear infinite reverse;
#             transform-style: preserve-3d;
#             perspective: 1000px;
#         }}
#
#         @keyframes planet-ring-rotation {{
#             0% {{
#                 transform: translate(100%, -50%) rotateY(0deg);
#                 clip-path: polygon(50%  0%, 0% 0%, 0% 0%, 0% 100%);
#             }}
#             50% {{
#                 transform: translate(-70%, -50%) rotateY(-180deg);
#                 clip-path: polygon(50%  0%, 100% 0%, 100% 100%, 50% 100%);
#             }}
#             100% {{
#                 transform: translate(-50%, -50%) rotateY(-360deg);
#                 clip-path: polygon(50% 0%, 100% 0%, 100% 100%, 50% 100%);
#             }}
#         }}
#
#         @keyframes text-rotation {{
#             0% {{
#                 transform: translateX(-100%) scale(0.5);
#                 opacity: 0;
#             }}
#             10% {{
#                 transform: translateX(-70%) scale(0.7) perspective(1000px);
#                 opacity: 1;
#             }}
#             90% {{
#                 transform: translateX(70%) scale(0.7) perspective(1000px);
#                 opacity: 1;
#             }}
#             100% {{
#                 transform: translateX(100%) scale(0.5) perspective(1000px) rotateY(180deg);
#                 opacity: 0;
#             }}
#         }}
#     </style>
#     <div class="button-container">
#         {buttons_html}
#     </div>
#     """
#
#     # Define the HTML code for a single button
#     button_template = """
#     <div class="button" style="background: {color};">
#         <a href="#" id="{id}">
#             {name}
#         </a>
#     </div>
#     """
#
#     # Generate the HTML code for the buttons
#     buttons_html = ""
#     for i, model_name in enumerate(button_names):
#         button_html = button_template.format(
#             id=model_name.lower().replace(" ", "_"),
#             name=model_name,
#             color=button_colors[i % len(button_colors)]
#         )
#         buttons_html += button_html
#
#     # Combine the buttons HTML with the container HTML
#     full_html = button_container.format(buttons_html=buttons_html)
#
#     clicked_model_btn = click_detector(full_html)
#     return clicked_model_btn
# =============================================================================

# =============================================================================
# # =============================================================================
# #  Show/Hide Button to download dataframe
# # =============================================================================
# # have button available for user and if clicked it expands with the dataframe
# col1, col2, col3 = st.columns([100,50,95])
# with col2:
#     # create empty placeholder for button show/hide
#     placeholder = st.empty()
#
#     # create button (enabled to click e.g. disabled=false with unique key)
#     btn = placeholder.button('Show Details',
#                              disabled=False,
#                              key = "show_naive_trained_model_btn")
#
# # if button is clicked run below code
# if btn == True:
#
#     # display button with text "click me again", with unique key
#     placeholder.button('Hide Details', disabled=False, key = "hide_naive_trained_model_btn")
# =============================================================================

# =============================================================================
# show_lottie_animation(url="./images/89601-solar-system.json", key='solar_system', speed = 1, width=400, reverse=False, height=400, margin_before = 2, margin_after=10)
# =============================================================================

# =============================================================================
#
# # left / right arrow you can create with click event from st-click-detector package
# button_container2 = """
# <style>
# .arrow-container {
#     display: flex;
#     justify-content: center;
#     align-items: center;
# }
#
# .arrow {
#     width: 30px;
#     height: 30px;
#     border-radius: 50%;
#     background-color: black;
#     display: flex;
#     justify-content: center;
#     align-items: center;
#     cursor: pointer;
# }
#
# .left-arrow {
#     margin-right: 10px;
# }
#
# .right-arrow {
#     margin-left: 10px;
# }
#
# .arrow-icon {
#     width: 10px;
#     height: 10px;
#     border-top: 2px solid white;
#     border-right: 2px solid white;
#     transform: rotate(225deg);
# }
# .rarrow-icon {
#     width: 10px;
#     height: 10px;
#     border-top: 2px solid white;
#     border-right: 2px solid white;
#     transform: rotate(45deg);
# }
# </style>
#
# <div class="arrow-container">
#     <a href="#" id="left-arrow" class="arrow left-arrow" onclick="handleClick('left-arrow')">
#         <span class="arrow-icon"></span>
#     </a>
#     <a href="#" id="right-arrow" class="arrow right-arrow" onclick="handleClick('right-arrow')">
#         <span class="rarrow-icon"></span>
#     </a>
# </div>
# """
# =============================================================================

# =============================================================================
#             title = '\"Hi 👋 Welcome to the ForecastGenie app!\"'
#             # set gradient color of letters of title
#             gradient = '-webkit-linear-gradient(left, #F08A5D, #FABA63, #2E9CCA, #4FB99F)'
#             # show in streamlit the title with gradient
#             st.markdown(f'<h1 style="text-align:center; background: none; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-image: {gradient};"> {title} </h1>', unsafe_allow_html=True)
#             # vertical spacer
#             st.write('')
# =============================================================================
# =============================================================================
#                 # Create Carousel Cards
#                 # define for each card the header in the header list
#                 header_list = ["📈",
#                                "🔍",
#                                "🧹",
#                                "🧰",
#                                "🔢"]
#                 # define for each card the corresponding paragraph in the list
#                 paragraph_list = ["Forecasting made easy",
#                                   "Professional data analysis",
#                                   "Automated data cleaning",
#                                   "Intelligent feature engineering",
#                                   "User-friendly interface",
#                                   "Versatile model training"]
#                 # define the font family to display the text of paragraph
#                 font_family = "Trebuchet MS"
#                 # define the paragraph text size
#                 font_size = '18px'
#                 # in streamlit create and show the user defined number of carousel cards with header+text
#                 create_carousel_cards(4, header_list, paragraph_list, font_family, font_size)
# =============================================================================
