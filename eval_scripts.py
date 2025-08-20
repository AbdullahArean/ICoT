ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
import re
def judge_answer(text, choices, answer):
  if isinstance(answer, int):
  answer = ALPHA_MAP[answer]
  
  pattern = re.compile(r'\(([A-Za-z])\)')
  res = pattern.findall(text)
  if len(res) >= 1:
      pred = res[-1].upper()  # 'A', 'B', ...
  else:
      res = []
      for i, choice in enumerate(choices):
          if choice.lower() in text.lower():
              res.append(ALPHA_MAP[i])
      if len(res) >= 1:
          pred = res[-1]
      else:
          for i, choice in enumerate(choices):
              text = re.sub(r'[\n.,!?]', ' ', text)
              if ALPHA_MAP[i] in text.split(" "):
                  res.append(ALPHA_MAP[i])
          if len(res) >= 1:
              pred = res[-1]
          else:
              for i, choice in enumerate(choices):
                  text = re.sub(r'[\n.,!?]', ' ', text)
                  if ALPHA_MAP[i].lower() in text.split(" "):
                      res.append(ALPHA_MAP[i])
              if len(res) >= 1:
                  pred = res[-1]
              else:
                  pred = "FAILED"
  
                 
  if pred == answer:
      return True
  else:
      return False
