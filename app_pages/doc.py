import streamlit as st
from style.text import my_text_header, my_text_paragraph, vertical_spacer
from style.animations import show_lottie_animation
from style.icons import load_icons

# Load icons
icons = load_icons()


class DocPage:
    FRONT_COVER = '-'
    LOAD = 'Step 1: Load Dataset'
    EXPLORE = 'Step 2: Explore Dataset'
    CLEAN = 'Step 3: Clean Dataset'
    ENGINEER = 'Step 4: Engineer Features'
    PREPARE = 'Step 5: Prepare Dataset'
    SELECT = 'Step 6: Select Features'
    TRAIN = 'Step 7: Train Models'
    EVALUATE = 'Step 8: Evaluate Models'
    TUNE = 'Step 9: Tune Models'
    FORECAST = 'Step 10: Forecast'

    # Define constants for file paths and URLs
    IMAGES_PATH = "./images/"
    FRONT_COVER_IMAGE = IMAGES_PATH + "documentation.png"
    LOAD_IMAGE = IMAGES_PATH + "116206-rocket-fly-out-the-laptop.json"
    EXPLORE_IMAGE = IMAGES_PATH + "58666-sputnik-mission-launch.json"
    CLEAN_IMAGE = IMAGES_PATH + "88404-loading-bubbles.json"
    ENGINEER_IMAGE = IMAGES_PATH + "141844-shapes-changing-preloader.json"
    PREPARE_IMAGE = IMAGES_PATH + "141560-loader-v25.json"
    SELECT_IMAGE = IMAGES_PATH + "102149-square-loader.json"
    TRAIN_IMAGE = IMAGES_PATH + "100037-rubiks-cube.json"
    EVALUATE_IMAGE = IMAGES_PATH + "70114-blue-stars.json"
    TUNE_IMAGE = IMAGES_PATH + "95733-loading-20.json"
    FORECAST_IMAGE = IMAGES_PATH + "55298-data-forecast-loading.json"

    PAGES = [FRONT_COVER, LOAD, EXPLORE, CLEAN, ENGINEER, PREPARE, SELECT, TRAIN, EVALUATE, TUNE, FORECAST]

    def __init__(self):
        """Initialize the DocPage class."""
        self.selected_doc_page = None

    def render(self):
        """Render the documentation page."""
        try:
            with st.sidebar:
                self._render_sidebar()

            if self.selected_doc_page:
                self._render_selected_page()

        except Exception as e:
            st.error(f'ForecastGenie Error: "Documentation" in sidebar-menu did not load properly. Error: {e}')

    def _render_sidebar(self):
        """Render the sidebar for the documentation page."""
        vertical_spacer(2)
        with (st.sidebar):
            my_text_paragraph(f'{icons["book_icon"]}')

            # Use st.sidebar.radio for the selected page index
            self.selected_doc_page = st.sidebar.radio(
                label="*Select Page*:",
                options=self.PAGES,
                label_visibility='collapsed',
                index=self.PAGES.index(self.selected_doc_page) if self.selected_doc_page else 0
            )

            # Adjust the alignment of the label in the sidebar to center
            st.sidebar.markdown(
                """
                <style>
                    div[data-testid="stRadio"] {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

    def _render_selected_page(self):
        """Render the selected documentation page."""
        if self.selected_doc_page == self.FRONT_COVER:
            self._render_front_cover_page()
        elif self.selected_doc_page == self.LOAD:
            self._render_load_page()
        elif self.selected_doc_page == self.EXPLORE:
            self._render_explore_page()
        elif self.selected_doc_page == self.CLEAN:
            self._render_clean_page()
        elif self.selected_doc_page == self.ENGINEER:
            self._render_engineer_page()
        elif self.selected_doc_page == self.PREPARE:
            self._render_prepare_page()
        elif self.selected_doc_page == self.SELECT:
            self._render_select_page()
        elif self.selected_doc_page == self.TRAIN:
            self._render_train_page()
        elif self.selected_doc_page == self.EVALUATE:
            self._render_evaluate_page()
        elif self.selected_doc_page == self.TUNE:
            self._render_tune_page()
        elif self.selected_doc_page == self.FORECAST:
            self._render_forecast_page()

    @staticmethod
    def _render_front_cover_page():
        with st.expander('', expanded=True):
            st.image(DocPage.FRONT_COVER_IMAGE)

    @staticmethod
    def _render_load_page():
        with st.expander('', expanded=True):
            # Render Step 1: Load Dataset page
            my_text_header('<b> Step 1: </b> <br> Load Dataset')

            show_lottie_animation(url=DocPage.LOAD_IMAGE, key="rocket_fly_out_of_laptop",
                                  height=200, width=200, speed=1, loop=True, quality='high', col_sizes=[4, 4, 4])

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph('''The <i> ForecastGenie </i> application provides users with a convenient way to 
                upload data files from the sidebar menu. The application supports common file formats such as .csv and 
                .xls. To load a file, users can navigate to the sidebar menu and locate the <i> "Upload Data" </i> 
                option. Upon clicking on this option, a file upload dialog box will appear, allowing users to select a 
                file from their local system. When uploading a file, it is important to ensure that the file meets 
                specific requirements for proper data processing and forecasting. <br> - The first column should have a 
                header named <i> 'date' </i> and dates should be in the mm/dd/yyyy format (12/31/2023), representing the 
                time series data. The dates should be sorted in chronological order, as this is crucial for accurate 
                forecasting. <br> - The second column should contain a header that includes the name of the variable of 
                interest and below it's historical data, for which the user wishes to forecast. <br> Fasten your 
                seatbelt, lean back, and savor the journey as your data blasts off into the realm of forecasting 
                possibilities! Embrace the adventure and enjoy the ride.''', my_text_align='justify')
            vertical_spacer(2)

    @staticmethod
    def _render_explore_page():
        with st.expander('', expanded=True):
            # Render Step 2: Explore Dataset page
            my_text_header('<b> Step 2: </b> <br> Explore Dataset')

            show_lottie_animation(url=DocPage.EXPLORE_IMAGE,
                                  key='test',
                                  width=150,
                                  height=150,
                                  speed=1,
                                  col_sizes=[45, 40, 40],
                                  margin_before=1)

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph('''The <i> Explore </i> tab in the app is designed to provide users with comprehensive 
                tools for data exploration and analysis. It offers valuable insights into the dataset. <br> The <i> Quick 
                Summary </i> section provides an overview of the dataset including the number of rows, start date, 
                missing values, mean, minimum, standard deviation, maximum date, frequency, median, maximum, 
                and mode. <br> The <i> Quick Insights </i>  section is designed to get the summarized observations that 
                highlight important characteristics of the data. This includes indications of dataset size, presence or 
                absence of missing values, balance between mean and median values, variability, symmetry, and the number 
                of distinct values. <br> The <i> Patterns </i> section allows users to visually explore underlying 
                patterns and relationships within the data. Users can select different histogram frequency types to 
                visualize data distribution. The Ljung-Box test is available to assess the presence of white noise. The 
                Augmented Dickey-Fuller (ADF) test can be used to determine stationarity, indicating whether the data 
                exhibits time-dependent patterns. Additionally, users can analyze auto-correlation using the ACF/PACF to 
                understand how data point relates to previous data points. <br> Do not skip exploratory data analysis, 
                because you don't want to end up finding yourself lost in a black hole!''', my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_clean_page():
        # Render Step 3: Clean Dataset page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 3: </b> <br> Clean Dataset')

            show_lottie_animation(url=DocPage.CLEAN_IMAGE, key="loading_bubbles", width=200,
                                  height=200, col_sizes=[2, 2, 2])

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    '''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.''', my_text_align='justify')
                vertical_spacer(2)
    @staticmethod
    def _render_engineer_page():
        # Render Step 4: Engineer Features page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 3: </b> <br> Clean Dataset')

            show_lottie_animation(url=DocPage.ENGINEER_IMAGE, key='shapes_changing_preloader',
                                  width=200, height=200, col_sizes=[2, 2, 2])

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    '''Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere '
                    'cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna vitae felis '
                    'finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum vestibulum. Vestibulum '
                    'pharetra, velit et cursus commodo, erat enim eleifend massa, ac pellentesque velit turpis nec ex. '
                    'Fusce scelerisque, velit non lacinia iaculis, tortor neque viverra turpis, in consectetur diam dui a '
                    'urna. Quisque in velit malesuada, scelerisque tortor vel, dictum massa. Quisque in malesuada libero.''',
                    my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_prepare_page():
        # Render Step 5: Prepare Dataset page
        with st.expander('', expanded=True):
                my_text_header('<b> Step 5: </b> <br> Prepare Dataset')

                show_lottie_animation(url=DocPage.PREPARE_IMAGE, key='prepare', width=200, height=200,
                                      col_sizes=[20, 20, 20])

                col1, col2, col3 = st.columns([2, 8, 2])
                with col2:
                    my_text_paragraph(
                        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris '
                        'congue, non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et '
                        'ultrices posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec '
                        'rutrum magna vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a '
                        'tellus interdum vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend '
                        'massa, ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, '
                        'tortor neque viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, '
                        'scelerisque tortor vel, dictum massa. Quisque in malesuada libero.',
                        my_text_align='justify')
                    vertical_spacer(2)
    @staticmethod
    def _render_select_page():
        # Render Step 6: Select Features page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 6: </b> <br> Select Features')

            show_lottie_animation(url=DocPage.SELECT_IMAGE, key='square_loader', width=200, height=200,
                                  col_sizes=[4, 4, 4])

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.',
                    my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_train_page():
        # Render Step 7: Train Models page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 7: </b> <br> Train Models')

            show_lottie_animation(url=DocPage.TRAIN_IMAGE, key='rubiks_cube', width=200, height=200,
                                  col_sizes=[4, 4, 4])

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.',
                    my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_evaluate_page():
        # Render Step 8: Evaluate Models page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 8: </b> <br> Evaluate Models')

            show_lottie_animation(url=DocPage.EVALUATE_IMAGE, key='blue-stars', width=200, height=200,
                                  col_sizes=[4, 4, 4], speed=1)

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.',
                    my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_tune_page():
        # Render Step 9: Tune Models page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 9: </b> <br> Tune Models')

            show_lottie_animation(url=DocPage.TUNE_IMAGE, key='tune_sliders', width=150, height=150,
                                  col_sizes=[5, 4, 4], speed=1)

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.',
                    my_text_align='justify')
                vertical_spacer(2)

    @staticmethod
    def _render_forecast_page():
        # Render Step 10: Forecast page
        with st.expander('', expanded=True):
            my_text_header('<b> Step 10: </b> <br> Forecast')

            show_lottie_animation(url=DocPage.FORECAST_IMAGE, key='forecast', width=200,
                                  height=200, col_sizes=[4, 4, 4], speed=1)

            col1, col2, col3 = st.columns([2, 8, 2])
            with col2:
                my_text_paragraph(
                    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam pretium nisl vel mauris congue, '
                    'non feugiat neque lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices '
                    'posuere cubilia Curae; Sed ullamcorper massa ut ligula sagittis tristique. Donec rutrum magna '
                    'vitae felis finibus, vitae eleifend nibh commodo. Aliquam fringilla dui a tellus interdum '
                    'vestibulum. Vestibulum pharetra, velit et cursus commodo, erat enim eleifend massa, '
                    'ac pellentesque velit turpis nec ex. Fusce scelerisque, velit non lacinia iaculis, tortor neque '
                    'viverra turpis, in consectetur diam dui a urna. Quisque in velit malesuada, scelerisque tortor '
                    'vel, dictum massa. Quisque in malesuada libero.',
                    my_text_align='justify')
                vertical_spacer(2)
