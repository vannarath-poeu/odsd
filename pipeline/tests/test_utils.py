from utils.text import clean_text

def test_clean_text():
  # Test replace \n with ' ' 
  test_string = "hi\nteam"
  assert clean_text(test_string) == "hi team"