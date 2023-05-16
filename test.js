<script>
function carousel_previous() {
  var carousel = document.querySelector('.carousel');
  var scrollPos = carousel.scrollLeft;
  var cardWidth = carousel.offsetWidth;
  var cardPos = Math.floor(scrollPos / cardWidth);
  carousel.scrollTo({
    left: (cardPos - 1) * cardWidth,
    behavior: 'smooth'
  });
}

function carousel_next() {
  var carousel = document.querySelector('.carousel');
  var scrollPos = carousel.scrollLeft;
  var cardWidth = carousel.offsetWidth;
  var cardPos = Math.floor(scrollPos / cardWidth);
  carousel.scrollTo({
    left: (cardPos + 1) * cardWidth,
    behavior: 'smooth'
  });
}
</script>

    Add CSS styling for the two buttons:

css

#previous-button, #next-button {
  cursor: pointer;
  border: none;
  background-color: #333333;
  color: #FFFFFF;
  font-family: """ + font_family + """, sans-serif;
  font-size: """ + str(font_size) + """px;
  padding: 10px 20px;
  border-radius: 5px;
  margin: 5px;
}

#previous-button:hover, #next-button:hover {
  background-color: #555555;
}

#previous-button:disabled, #next-button:disabled {
  opacity: 0.5;
  cursor: default;
}

Make sure to add these lines of code within the <style>...</style> and <script>...</script> tags that are already present in your create_carousel_cards() function.
