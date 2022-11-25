# copied from https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/text/__init__.py#L48
""" from https://github.com/keithito/tacotron """
from utils_.text import cleaners
from utils_.text.symbols import symbols #, custom_alphabets


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# _symbol_to_id_alpha = {s: i for i, s in enumerate(custom_alphabets)}

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  # sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  keys = _symbol_to_id.keys()
  sequence = []
  for symbol in cleaned_text:
    if symbol in keys:
      sequence.append(_symbol_to_id[symbol])
    else:
      pass

  return sequence

# def custom_alpha_text_to_sequence(text):
#   keys = _symbol_to_id_alpha.keys()
  
#   sequence = []
#   for symbol in text:
#     if symbol in keys:
#       sequence.append(_symbol_to_id_alpha[symbol])
#     else:
#       pass
  
#   return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text